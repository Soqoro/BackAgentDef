"""Isolated offline defender process; no torch/Transformers imports in the victim."""
import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess

from .schemas import Invalid, canonical


def validate_local(c):
    x = c["local"]
    if set(x) != {"python", "lock", "lock_sha256", "device", "max_input_tokens", "startup_seconds"}:
        raise Invalid("invalid local defender schema")
    if not all(isinstance(x[k], str) and Path(x[k]).is_absolute() for k in ("python", "lock")):
        raise Invalid("local defender requires absolute Python and lock paths")
    import re
    if not re.fullmatch(r"[a-f0-9]{64}", x["lock_sha256"]):
        raise Invalid("pin local defender lock SHA256")
    if x["device"] != "cuda:1":
        raise Invalid("integrated defender must use visible cuda:1; victim uses cuda:0")
    if type(x["max_input_tokens"]) is not int or not 1 <= x["max_input_tokens"] <= 16384:
        raise Invalid("invalid local input cap")
    if type(x["startup_seconds"]) is not int or not 1 <= x["startup_seconds"] <= 1800:
        raise Invalid("invalid local startup timeout")
    if c["parameters"] != {"temperature": 0} or c["response_format"] != "json_object":
        raise Invalid("local pilot requires greedy JSON output (validated, not grammar constrained)")


class LocalRoles:
    simulated = False

    def __init__(self, config):
        validate_local(config)
        if not os.environ.get("SLURM_JOB_ID"):
            raise Invalid("local defender requires Slurm allocation")
        self.config = config
        self.process = None
        self.runtime = None
        atexit.register(self.close)

    def _read(self, seconds):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(seconds):
                raise Invalid("local defender timeout")
            line = self.process.stdout.readline()
        if not line:
            raise Invalid("local defender exited")
        result = json.loads(line)
        if "error" in result:
            raise Invalid("local defender error: " + result["error"])
        return result

    def call(self, role, payload):
        from .roles import role_messages
        from .qwen_worker import check_lock
        c = self.config
        try:
            if self.process is None:
                check_lock(c["local"]["lock"], c["local"]["lock_sha256"], c["model"])
                env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
                self.process = subprocess.Popen(
                    [c["local"]["python"], "-u", str(Path(__file__).with_name("qwen_worker.py"))],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
                self.process.stdin.write(canonical(c) + "\n")
                self.process.stdin.flush()
                self.runtime = self._read(c["local"]["startup_seconds"])
                if self.runtime.get("status") != "ready":
                    raise Invalid("local defender did not become ready")
            self.process.stdin.write(canonical({"messages": role_messages(role, payload)}) + "\n")
            self.process.stdin.flush()
            result = self._read(c["timeout_seconds"])
            result["runtime"] = self.runtime
            return result
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
