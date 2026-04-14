#!/usr/bin/env python3
"""
Materialize agenda objects (from a single checkpoint pickle) to the filesystem.

Basically just unpacks the objects in an agenda checkpoint file to disk.

Example usage:
  python materialize.py materialize /path/to/checkpoint.pkl /output/dir --overwrite
  python materialize.py /path/to/checkpoint.pkl /output/dir --overwrite    # legacy

Export verified programs as JSON for the web report:
  python materialize.py export-programs /path/to/checkpoint.pkl
  python materialize.py export-dafnybench /path/to/DafnyBench/ground_truth/
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

from patch import apply_patch
from language import Language, Program

logger = logging.getLogger(__name__)

_backend = Language.DAFNY.get_backend()


# ── Export helpers ─────────────────────────────────────────────────────────

def _program_to_entry(source: str) -> dict:
    """Clean a Dafny program and extract per-method stats."""
    prog = Program(source, Language.DAFNY)
    stripped = _backend.strip(prog)
    cleaned = str(stripped)
    methods = _backend.extract_methods(stripped)
    return {
        'source': cleaned,
        'methods': methods,
    }


def _programs_from_agenda(checkpoint: Path) -> list[dict]:
    """Load verified programs from an agenda checkpoint pickle."""
    with checkpoint.open('rb') as f:
        data = pickle.load(f)

    objects = data.get('objects', {})
    programs = []

    for key, obj in objects.items():
        if not key.startswith('dataset/'):
            continue
        content = getattr(obj, 'content', None)
        if content is None:
            continue
        if isinstance(content, (bytes, bytearray, memoryview)):
            content = bytes(content).decode('utf-8', errors='replace')
        programs.append(_program_to_entry(content))

    return programs


def _programs_from_dafnybench(directory: Path) -> list[dict]:
    """Load .dfy files from a DafnyBench ground_truth directory."""
    programs = []
    for dfy_file in sorted(directory.glob('*.dfy')):
        programs.append(_program_to_entry(dfy_file.read_text()))
    return programs


def _write_dataset_json(programs: list[dict], output_path: Path, name: str) -> None:
    """Write the dataset JSON used by the web report."""
    report = {
        'name': name,
        'n_programs': len(programs),
        'programs': programs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(report, f)
    print(f"Wrote {len(programs)} programs to {output_path}")


# ── Original materialize ──────────────────────────────────────────────────

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
        path_attr = getattr(obj, "path", None)
        content = getattr(obj, "content", None)
        properties = getattr(obj, "properties", {}) or {}

        if not isinstance(path_attr, str) or not path_attr:
            raise ValueError(f"Invalid object path for entry {obj_path!r}")

        rel_path = Path(path_attr.lstrip("/"))
        base_path = (out_dir / rel_path).resolve()

        if os.path.commonpath([str(base_path), str(out_dir)]) != str(out_dir):
            raise ValueError(f"Refusing to write outside destination: {base_path}")

        base_path.parent.mkdir(parents=True, exist_ok=True)

        if content is None:
            current = b""
        elif isinstance(content, (bytes, bytearray, memoryview)):
            current = bytes(content)
        else:
            raise TypeError(
                f"Object content must be bytes; got {type(content).__name__} for {path_attr!r}"
            )

        if base_path.exists() and not overwrite:
            raise FileExistsError(f"File exists (and no --overwrite): {base_path}")
        with base_path.open("wb") as wf:
            wf.write(current)
        logger.info(f"Wrote {base_path}")

        patch_history = properties.get("patch_history") or []
        prevs = []
        cur_bytes = current
        for patch in reversed(patch_history):
            try:
                cur_bytes = apply_patch(cur_bytes, patch)
            except Exception as e:
                logger.warning(f"Failed to apply patch for {path_attr}: {e}")
                break
            prevs.append(cur_bytes)

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


# ── CLI ────────────────────────────────────────────────────────────────────

SUBCOMMANDS = {'materialize', 'export-programs', 'export-dafnybench'}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize agenda objects to files, or export dataset JSON.")
    sub = ap.add_subparsers(dest='command')

    mat = sub.add_parser('materialize', help='Materialize agenda objects to disk')
    mat.add_argument('checkpoint', type=Path, help='Path to an agenda checkpoint pickle file')
    mat.add_argument('output_dir', type=Path, nargs='?', default=None,
                     help='Directory where files will be written (default: agendas/materialized/<checkpoint stem>/)')
    mat.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

    exp = sub.add_parser('export-programs', help='Export verified programs from agenda pkl as JSON')
    exp.add_argument('checkpoint', type=Path, help='Path to an agenda checkpoint pickle file')
    exp.add_argument('-o', '--output', type=Path, default=None,
                     help='Output JSON path (default: <checkpoint>.programs.json)')
    exp.add_argument('--name', type=str, default=None,
                     help='Dataset name in JSON (default: checkpoint stem)')

    bench = sub.add_parser('export-dafnybench', help='Export DafnyBench .dfy files as JSON')
    bench.add_argument('directory', type=Path, help='Path to DafnyBench ground_truth directory')
    bench.add_argument('-o', '--output', type=Path, default=None,
                       help='Output JSON path (default: dafnybench.programs.json)')
    bench.add_argument('--name', type=str, default=None,
                       help='Dataset name in JSON (default: DafnyBench)')

    return ap.parse_args()


def main() -> None:
    # Backwards compat: if first arg is not a subcommand, treat as legacy materialize
    if len(sys.argv) > 1 and sys.argv[1] not in SUBCOMMANDS and sys.argv[1] not in ('-h', '--help'):
        checkpoint = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else None
        overwrite = '--overwrite' in sys.argv
        if output_dir is None:
            output_dir = Path("agendas/materialized") / checkpoint.stem
        materialize(checkpoint, output_dir, overwrite)
        return

    args = parse_args()

    if args.command == 'export-programs':
        programs = _programs_from_agenda(args.checkpoint)
        output = args.output or args.checkpoint.with_suffix('.programs.json')
        name = args.name or args.checkpoint.stem
        _write_dataset_json(programs, output, name)

    elif args.command == 'export-dafnybench':
        programs = _programs_from_dafnybench(args.directory)
        output = args.output or Path('dafnybench.programs.json')
        name = args.name or 'DafnyBench'
        _write_dataset_json(programs, output, name)

    elif args.command == 'materialize':
        out_dir = args.output_dir
        if out_dir is None:
            out_dir = Path("agendas/materialized") / args.checkpoint.stem
        materialize(args.checkpoint, out_dir, args.overwrite)

    else:
        parse_args()
        sys.exit(1)


if __name__ == "__main__":
    main()
