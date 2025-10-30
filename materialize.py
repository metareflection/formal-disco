#!/usr/bin/env python3
"""
Materialize agenda objects (from a single checkpoint pickle) to the filesystem.

Basically just unpacks the objects in an agenda checkpoint file to disk.

Example usage:
  python materialize.py /path/to/checkpoint.pkl /output/dir --overwrite
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

from patch import apply_patch

logger = logging.getLogger(__name__)


def materialize(checkpoint: Path, out_dir: Path, overwrite: bool) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    with checkpoint.open("rb") as f:
        data = pickle.load(f)

    objects = data.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("Checkpoint does not contain an 'objects' dict.")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for obj_path, obj in objects.items():
        # obj is a dataclass with attributes (path, content, properties, ...)
        path_attr = getattr(obj, "path", None)
        content = getattr(obj, "content", None)
        properties = getattr(obj, "properties", {}) or {}

        if not isinstance(path_attr, str) or not path_attr:
            raise ValueError(f"Invalid object path for entry {obj_path!r}")

        rel_path = Path(path_attr.lstrip("/"))
        base_path = (out_dir / rel_path).resolve()

        # To be sure, ensure we didn't escape out_dir
        if os.path.commonpath([str(base_path), str(out_dir)]) != str(out_dir):
            raise ValueError(f"Refusing to write outside destination: {base_path}")

        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize current bytes
        if content is None:
            current = b""
        elif isinstance(content, (bytes, bytearray, memoryview)):
            current = bytes(content)
        else:
            raise TypeError(
                f"Object content must be bytes; got {type(content).__name__} for {path_attr!r}"
            )

        # Write latest version to the original path
        if base_path.exists() and not overwrite:
            raise FileExistsError(f"File exists (and no --overwrite): {base_path}")
        with base_path.open("wb") as wf:
            wf.write(current)
        logger.info(f"Wrote {base_path}")

        # Reconstruct previous versions using reverse patches
        patch_history = properties.get("patch_history") or []
        # Build list of older versions: prev1 (one step back) ... prevN (earliest)
        prevs = []
        cur_bytes = current
        for patch in reversed(patch_history):
            try:
                cur_bytes = apply_patch(cur_bytes, patch)
            except Exception as e:
                logger.warning(f"Failed to apply patch for {path_attr}: {e}")
                break
            prevs.append(cur_bytes)

        # We want .v0 to be the earliest version (after undoing all patches),
        # .v1 the next, and so on. If there are no patches, .v0 is the current.
        if not prevs:
            versions = [current]
        else:
            versions = list(reversed(prevs))

        for idx, bytes_i in enumerate(versions):
            vi_path = base_path.with_name(base_path.name + f".v{idx}")
            if vi_path.exists() and not overwrite:
                raise FileExistsError(f"File exists (and no --overwrite): {vi_path}")
            with vi_path.open("wb") as wf:
                wf.write(bytes_i)
            logger.info(f"Wrote {vi_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize agenda objects to files.")
    ap.add_argument("checkpoint", type=Path, help="Path to an agenda checkpoint pickle file")
    ap.add_argument("output_dir", type=Path, help="Directory where files will be written")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    materialize(args.checkpoint, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
