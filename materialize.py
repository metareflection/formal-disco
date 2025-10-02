#!/usr/bin/env python3
"""
Materialize agenda objects (from a single checkpoint pickle) to the filesystem.

Basically just unpacks the objects in an agenda checkpoint file to disk.

Example usage:
  python materialize.py /path/to/checkpoint.pkl /output/dir --overwrite
"""

import argparse
import os
import pickle
from pathlib import Path


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
        # obj is a dataclass with attributes (path, content, ...)
        path_attr = getattr(obj, "path", None)
        content = getattr(obj, "content", None)

        if not isinstance(path_attr, str) or not path_attr:
            raise ValueError(f"Invalid object path for entry {obj_path!r}")

        rel_path = Path(path_attr.lstrip("/"))
        full_path = (out_dir / rel_path).resolve()

        # To be sure, ensure we didn't escape out_dir
        if os.path.commonpath([str(full_path), str(out_dir)]) != str(out_dir):
            raise ValueError(f"Refusing to write outside destination: {full_path}")

        full_path.parent.mkdir(parents=True, exist_ok=True)

        if full_path.exists() and not overwrite:
            raise FileExistsError(f"File exists (and no --overwrite): {full_path}")

        if content is None:
            data_bytes = b""
        elif isinstance(content, (bytes, bytearray, memoryview)):
            data_bytes = bytes(content)
        else:
            raise TypeError(
                f"Object content must be bytes; got {type(content).__name__} for {path_attr!r}"
            )

        with full_path.open("wb") as wf:
            wf.write(data_bytes)

        print(f"Wrote {full_path}")


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
