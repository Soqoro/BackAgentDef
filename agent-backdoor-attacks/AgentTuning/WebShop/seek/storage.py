"""Append-only, fsync'ed events; atomic summaries; single writer per row."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

from .schemas import Invalid, canonical, digest


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise Invalid(f"immutable record collision: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Link a complete temporary inode; even a killed writer cannot expose partial JSON.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".immutable-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if read_json(path) != value:
                raise Invalid("concurrent immutable record collision")
    finally:
        os.unlink(tmp)


def events(path):
    path = Path(path)
    if not path.exists():
        return []
    result, seen = [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["id"] in seen:
            raise Invalid("duplicated event ID")
        seen.add(row["id"])
        result.append(row)
    return result


class Journal:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        # Retain a torn final write as evidence, never guess its payload.
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                cut = raw.rfind(b"\n") + 1
                tail = raw[cut:]
                immutable_json(self.root / ("interrupted-tail-" + digest(tail.hex()) + ".json"), {"hex": tail.hex()})
                with self.path.open("r+b") as handle:
                    handle.truncate(cut)
        self.records = events(self.path)

    def emit(self, kind, data):
        row = {"id": digest([len(self.records), kind, data]), "sequence": len(self.records),
               "kind": kind, "data": data, "time": time.time()}
        with self.path.open("a") as handle:
            handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(row)
        return row["id"]

    def completed(self, key):
        found = [r["data"]["result"] for r in self.records
                 if r["kind"] == "call_complete" and r["data"]["key"] == key]
        if len(found) > 1:
            raise Invalid("duplicate successful call")
        return found[0] if found else None

    def call(self, key_parts, category, phase, budget, callback):
        key = digest(key_parts)
        existing = self.completed(key)
        if existing is not None:
            self.emit("cache_hit", {"key": key, "category": category, "phase": phase, "role": key_parts.get("role")})
            return existing
        starts = [r for r in self.records if r["kind"] == "call_attempt" and
                  r["data"]["category"] == category and r["data"]["phase"] == phase]
        if len(starts) >= budget:
            raise Invalid(f"budget_exhausted:{category}:{phase}")
        attempt = self.emit("call_attempt", {"key": key, "category": category, "phase": phase,
                                              "input_hash": digest(key_parts.get("ids", key_parts)),
                                              "role": key_parts.get("role"), "retry": key_parts.get("retry", 0)})
        started = time.monotonic()
        try:
            result = callback()
        except Exception as exc:
            # Exception strings can include credentials, paths and full provider responses.
            self.emit("call_failed", {"attempt": attempt, "key": key, "phase": phase,
                                       "category": category, "error_type": type(exc).__name__})
            raise
        self.emit("call_complete", {"attempt": attempt, "key": key, "phase": phase,
                                     "category": category, "role": key_parts.get("role"), "latency_seconds": time.monotonic() - started,
                                     "result": result})
        return result


@contextmanager
def row_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Invalid("row already has an active worker") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
