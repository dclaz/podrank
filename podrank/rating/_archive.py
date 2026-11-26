"""Posterior archive persistence: JSON manifest plus typed ``.npy`` members.

This module owns the on-disk archive format only. It never uses pickle and does
not accept JSON lists for bulk numeric arrays.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from podrank.errors import PersistenceError

ARCHIVE_SCHEMA_VERSION = 2
MAX_MEMBERS = 256
MAX_MANIFEST_BYTES = 16 * 2**20
MAX_ARRAY_BYTES = 8 * 2**30
MAX_ARCHIVE_BYTES = 16 * 2**30
MAX_TOTAL_UNCOMPRESSED_BYTES = 16 * 2**30
_ALLOWED_DTYPE_KINDS = frozenset("fiubUS")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PersistenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _array_member(name: str) -> str:
    return f"arrays/{name}.npy"


def _array_descriptor(name: str, array: np.ndarray) -> dict[str, object]:
    return {
        "member": _array_member(name),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def write_posterior_archive(
    path: str | Path,
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Write one atomically replaced posterior archive."""

    target = Path(path)
    manifest_payload = dict(payload)
    descriptors: dict[str, object] = {}
    for name, array in arrays.items():
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise PersistenceError(f"invalid array name {name!r}")
        if array.dtype.kind not in _ALLOWED_DTYPE_KINDS:
            raise PersistenceError(f"array {name!r} has unsupported dtype {array.dtype}")
        if array.nbytes > MAX_ARRAY_BYTES:
            raise PersistenceError(f"array {name!r} exceeds the size bound")
        descriptors[name] = _array_descriptor(name, array)
    manifest_payload["arrays"] = descriptors
    manifest = {
        "kind": "podrank.posterior",
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "package_version": str(payload.get("package_version", "")),
        "payload": manifest_payload,
    }
    # The package version belongs in the manifest, not duplicated in payload.
    manifest_payload.pop("package_version", None)
    text = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(text) > MAX_MANIFEST_BYTES:
        raise PersistenceError("posterior manifest exceeds the size bound")

    descriptor = None
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        with os.fdopen(descriptor, "wb") as raw:
            descriptor = None
            with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("manifest.json", text)
                for name, array in arrays.items():
                    buffer = io.BytesIO()
                    np.save(buffer, array, allow_pickle=False)
                    archive.writestr(_array_member(name), buffer.getvalue())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise PersistenceError(f"could not write posterior archive: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def read_posterior_archive(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read and validate one posterior archive."""

    target = Path(path)
    try:
        if target.stat().st_size > MAX_ARCHIVE_BYTES:
            raise PersistenceError("posterior archive exceeds the size bound")
        archive = zipfile.ZipFile(target, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise PersistenceError(f"could not read posterior archive: {exc}") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(infos) > MAX_MEMBERS:
            raise PersistenceError("posterior archive exceeds the member-count bound")
        if sum(info.file_size for info in infos) > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise PersistenceError("posterior archive exceeds the total uncompressed size bound")
        if len(names) != len(set(names)):
            raise PersistenceError("posterior archive has duplicate members")
        if "manifest.json" not in names:
            raise PersistenceError("posterior archive has no manifest.json")
        manifest_info = archive.getinfo("manifest.json")
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise PersistenceError("posterior manifest exceeds the size bound")
        try:
            manifest = json.loads(
                archive.read("manifest.json").decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, PersistenceError) as exc:
            raise PersistenceError(f"invalid posterior manifest: {exc}") from exc
        if not isinstance(manifest, dict) or set(manifest) != {
            "kind",
            "schema_version",
            "package_version",
            "payload",
        }:
            raise PersistenceError("posterior manifest has the wrong keys")
        if manifest["kind"] != "podrank.posterior":
            raise PersistenceError("unsupported posterior kind")
        if manifest["schema_version"] != ARCHIVE_SCHEMA_VERSION:
            raise PersistenceError("unsupported posterior schema version")
        if not isinstance(manifest["payload"], dict):
            raise PersistenceError("posterior payload must be an object")
        payload = cast(dict[str, Any], manifest["payload"])
        descriptors = payload.get("arrays")
        if not isinstance(descriptors, dict):
            raise PersistenceError("posterior payload has no array descriptors")

        expected_members = {"manifest.json"}
        arrays: dict[str, np.ndarray] = {}
        for name, raw_descriptor in descriptors.items():
            if not isinstance(name, str) or not isinstance(raw_descriptor, dict):
                raise PersistenceError("invalid posterior array descriptor")
            if set(raw_descriptor) != {"member", "dtype", "shape"}:
                raise PersistenceError(f"invalid descriptor for array {name!r}")
            member = raw_descriptor["member"]
            if member != _array_member(name):
                raise PersistenceError(f"invalid member name for array {name!r}")
            expected_members.add(member)
            dtype = np.dtype(str(raw_descriptor["dtype"]))
            if dtype.kind not in _ALLOWED_DTYPE_KINDS:
                raise PersistenceError(f"unsupported dtype for array {name!r}")
            shape_raw = raw_descriptor["shape"]
            if not isinstance(shape_raw, list) or any(
                not isinstance(value, int) or value < 0 for value in shape_raw
            ):
                raise PersistenceError(f"invalid shape for array {name!r}")
            shape = tuple(shape_raw)
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if expected_bytes > MAX_ARRAY_BYTES:
                raise PersistenceError(f"array {name!r} exceeds the size bound")
            if member not in names:
                raise PersistenceError(f"missing archive member for array {name!r}")
            info = archive.getinfo(member)
            if info.file_size > MAX_ARRAY_BYTES + 1024 * 1024:
                raise PersistenceError(f"archive member {member!r} exceeds the size bound")
            if info.file_size < expected_bytes or info.file_size > expected_bytes + 4096:
                raise PersistenceError(
                    f"archive member {member!r} size does not match its descriptor"
                )
            try:
                with archive.open(member) as handle:
                    array = np.load(io.BytesIO(handle.read()), allow_pickle=False)
            except (OSError, ValueError, EOFError) as exc:
                raise PersistenceError(f"could not read array {name!r}: {exc}") from exc
            if array.dtype != dtype:
                raise PersistenceError(f"dtype mismatch for array {name!r}")
            if array.shape != shape:
                raise PersistenceError(f"shape mismatch for array {name!r}")
            arrays[name] = array
        if set(names) != expected_members:
            extra = sorted(set(names) - expected_members)
            missing = sorted(expected_members - set(names))
            raise PersistenceError(
                f"posterior archive members do not match manifest: extra={extra}, missing={missing}"
            )
        return payload, arrays


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "MAX_ARRAY_BYTES",
    "MAX_ARCHIVE_BYTES",
    "MAX_MANIFEST_BYTES",
    "MAX_MEMBERS",
    "MAX_TOTAL_UNCOMPRESSED_BYTES",
    "read_posterior_archive",
    "write_posterior_archive",
]
