"""Content fingerprints for the live WebShop catalogue and Lucene index."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FINGERPRINT_SCHEMA = "webshop-environment-sha256-v1"
DATA_FILES = (
    Path("data/items_shuffle.json"),
    Path("data/items_ins_v2.json"),
    Path("data/items_human_ins.json"),
)
INDEX_DIRECTORY = Path("search_engine/indexes")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EnvironmentFingerprintError(RuntimeError):
    """Raised when the WebShop environment cannot be frozen or validated."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_environment(webshop_root: str | Path) -> dict[str, Any]:
    """Hash every runtime data file and every regular Lucene-index file."""

    root = Path(webshop_root).resolve()
    paths: list[Path] = []
    missing = [
        relative
        for relative in DATA_FILES
        if not (root / relative).is_file()
    ]
    if missing:
        raise EnvironmentFingerprintError(
            "missing required WebShop data files: "
            + ", ".join(str(path) for path in missing)
        )
    paths.extend(root / relative for relative in DATA_FILES)

    index_root = root / INDEX_DIRECTORY
    if not index_root.is_dir():
        raise EnvironmentFingerprintError(
            f"missing WebShop Lucene index directory: {index_root}"
        )
    index_files = sorted(
        path
        for path in index_root.rglob("*")
        if path.is_file()
    )
    if not index_files:
        raise EnvironmentFingerprintError(
            f"WebShop Lucene index contains no regular files: {index_root}"
        )
    if not any(path.name.startswith("segments_") for path in index_files):
        raise EnvironmentFingerprintError(
            f"WebShop Lucene index contains no segments_* file: {index_root}"
        )
    paths.extend(index_files)

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }

    combined = hashlib.sha256()
    combined.update(FINGERPRINT_SCHEMA.encode("ascii"))
    combined.update(b"\0")
    for relative, record in sorted(files.items()):
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(record["size_bytes"]).encode("ascii"))
        combined.update(b"\0")
        combined.update(record["sha256"].encode("ascii"))
        combined.update(b"\0")

    return {
        "schema": FINGERPRINT_SCHEMA,
        "sha256": combined.hexdigest(),
        "files": files,
    }


def validate_fingerprint_record(
    value: Any,
    *,
    source: str,
) -> str:
    """Validate a serialized fingerprint and return its aggregate digest."""

    if not isinstance(value, Mapping):
        raise EnvironmentFingerprintError(
            f"{source} lacks a WebShop environment fingerprint"
        )
    if value.get("schema") != FINGERPRINT_SCHEMA:
        raise EnvironmentFingerprintError(
            f"{source} has an unsupported environment fingerprint schema"
        )
    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise EnvironmentFingerprintError(
            f"{source} lacks a valid aggregate environment SHA-256"
        )
    files = value.get("files")
    if not isinstance(files, Mapping) or not files:
        raise EnvironmentFingerprintError(
            f"{source} has no environment file fingerprints"
        )

    required_names = {path.as_posix() for path in DATA_FILES}
    observed_names: set[str] = set()
    for name, record in files.items():
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise EnvironmentFingerprintError(
                f"{source} contains an invalid environment path: {name!r}"
            )
        path = Path(name)
        if ".." in path.parts:
            raise EnvironmentFingerprintError(
                f"{source} contains a non-canonical environment path: {name!r}"
            )
        if not isinstance(record, Mapping):
            raise EnvironmentFingerprintError(
                f"{source} contains invalid metadata for {name!r}"
            )
        size = record.get("size_bytes")
        file_digest = record.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(file_digest, str)
            or _SHA256_RE.fullmatch(file_digest) is None
        ):
            raise EnvironmentFingerprintError(
                f"{source} contains an invalid fingerprint for {name!r}"
            )
        observed_names.add(path.as_posix())

    missing = sorted(required_names - observed_names)
    if missing:
        raise EnvironmentFingerprintError(
            f"{source} omits required WebShop data files: {missing}"
        )
    index_names = [
        name
        for name in observed_names
        if name.startswith(INDEX_DIRECTORY.as_posix() + "/")
    ]
    if not index_names or not any(
        Path(name).name.startswith("segments_") for name in index_names
    ):
        raise EnvironmentFingerprintError(
            f"{source} omits the Lucene segments_* index files"
        )
    return digest


def manifest_environment_record(manifest: Any) -> Mapping[str, Any]:
    """Return and validate the environment record frozen in a manifest."""

    metadata = getattr(manifest, "metadata", None)
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    environment = (
        source.get("environment") if isinstance(source, Mapping) else None
    )
    validate_fingerprint_record(
        environment,
        source="benchmark manifest",
    )
    return environment
