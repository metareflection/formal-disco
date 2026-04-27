#!/usr/bin/env python3
"""
Convert microsoft/Verus_Training_Data (SAFE + VeruSyn) into agenda-style pickles.

The HF dataset bundles several files:
  - sft_safe_25k.json         (SAFE,    25K samples)
  - sft_part1_6.9M.json       (VeruSyn, 6.9M samples, 15.7 GB)
  - sft_part2_4557.json       (VeruSyn, 4.6K long-CoT samples)
  - algorithmic_trajectory_9040.jsonl (VeruSyn raw trajectories, 9040)
  - system_trajectory_843.jsonl       (VeruSyn raw trajectories,  843)

Each record has roughly the columns:
  index, status, input_code, verified_code, log, original_item, return_code, error

We turn every record whose `status == 'success'` and which has non-empty
`verified_code` into a `dataset/`-prefixed `verus-program` Object, matching
the format expected by analysis.py / extract.py / distill_common.

Usage:
    # One source -> one pickle (lives next to the script unless --output is given)
    python baselines/verus_hf_to_pkl.py --source safe_sft   --limit 50000
    python baselines/verus_hf_to_pkl.py --source verusyn_sft2

    # Convert everything (separate pickle per file)
    python baselines/verus_hf_to_pkl.py --source all

The 6.9M file is parsed with `ijson` (`pip install ijson`) so we don't have to
load 15 GB into memory.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download

# Make the project root importable when running as `python baselines/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agenda import Object  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


HF_REPO_ID = "microsoft/Verus_Training_Data"

# Logical name -> (HF filename, origin label, format)
SOURCES: dict[str, tuple[str, str, str]] = {
    "safe_sft":      ("sft_safe_25k.json",                "SAFE",    "json"),
    "verusyn_sft1":  ("sft_part1_6.9M.json",              "VeruSyn", "json-stream"),
    "verusyn_sft2":  ("sft_part2_4557.json",              "VeruSyn", "json"),
    "verusyn_algo":  ("algorithmic_trajectory_9040.jsonl", "VeruSyn", "jsonl"),
    "verusyn_sys":   ("system_trajectory_843.jsonl",       "VeruSyn", "jsonl"),
}


def _iter_json_array(path: Path) -> Iterator[dict]:
    """Stream records from a JSON file that holds a single top-level list."""
    try:
        import ijson  # type: ignore
    except ImportError as e:
        raise SystemExit(
            f"This file ({path.name}) is too large to load in one shot; "
            f"please `pip install ijson` for streaming parses."
        ) from e
    with path.open("rb") as f:
        yield from ijson.items(f, "item")


def _iter_json_small(path: Path) -> Iterator[dict]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected top-level JSON list, got {type(data).__name__}")
    yield from data


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_records(path: Path, fmt: str) -> Iterator[dict]:
    if fmt == "jsonl":
        return _iter_jsonl(path)
    if fmt == "json":
        return _iter_json_small(path)
    if fmt == "json-stream":
        return _iter_json_array(path)
    raise ValueError(f"Unknown format: {fmt}")


def convert_source(
    source: str,
    output_dir: Path,
    limit: int | None,
    status_filter: set[str],
    cache_dir: str | None,
) -> Path:
    filename, origin, fmt = SOURCES[source]
    logger.info(f"Downloading {filename} from {HF_REPO_ID} (cached if present)…")
    local_path = Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    ))
    logger.info(f"  → {local_path}")

    objects: dict[str, Object] = {}
    n_seen = 0
    n_kept = 0
    n_no_code = 0

    for rec in _iter_records(local_path, fmt):
        n_seen += 1
        status = str(rec.get("status", "")).lower()
        if status_filter and status not in status_filter:
            continue
        code = rec.get("verified_code") or ""
        if not code or not code.strip():
            n_no_code += 1
            continue

        idx = rec.get("index", n_kept)
        path = f"dataset/{source}_{n_kept:08d}.rs"
        properties = {
            "verification_status": "success",
            "verification_outcome": "SUCCESS",
            "parent_idea": f"idea/{source}_{n_kept:08d}",
            "source_dataset": origin,
            "source_file": filename,
            "source_index": idx,
        }
        # Keep the original status when we accept non-success records.
        if status and status != "success":
            properties["status"] = status

        objects[path] = Object(
            path=path,
            type="verus-program",
            content=code.encode("utf-8"),
            properties=properties,
        )
        n_kept += 1

        if n_kept % 100_000 == 0:
            logger.info(f"  kept {n_kept} programs ({n_seen} seen)…")

        if limit is not None and limit > 0 and n_kept >= limit:
            logger.info(f"Hit --limit {limit}; stopping.")
            break

    logger.info(
        f"{filename}: kept {n_kept}/{n_seen} programs "
        f"(skipped {n_no_code} with no verified_code, "
        f"{n_seen - n_kept - n_no_code} on status filter)"
    )

    out_path = output_dir / f"{source}.pkl"
    pickle_data = {
        "objects": objects,
        "tasks": {},
        "status": {},
        "clock": 0,
    }
    with out_path.open("wb") as f:
        pickle.dump(pickle_data, f)
    logger.info(f"Wrote {out_path} ({n_kept} programs)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=[*SOURCES.keys(), "all", "all-safe", "all-verusyn"],
        required=True,
        help="Which file(s) to convert.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("baselines"),
        help="Directory for the output .pkl files (default: baselines/).",
    )
    parser.add_argument(
        "--limit", type=int, default=50_000,
        help="Cap programs per source (0 or negative = no cap). Default 50K to keep memory sane.",
    )
    parser.add_argument(
        "--status-filter", default="success",
        help="Comma-separated `status` values to keep (default: success). Pass empty to keep all.",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace cache directory (defaults to ~/.cache/huggingface).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "all":
        sources = list(SOURCES.keys())
    elif args.source == "all-safe":
        sources = [s for s, (_, origin, _) in SOURCES.items() if origin == "SAFE"]
    elif args.source == "all-verusyn":
        sources = [s for s, (_, origin, _) in SOURCES.items() if origin == "VeruSyn"]
    else:
        sources = [args.source]

    status_filter = {s.strip().lower() for s in args.status_filter.split(",") if s.strip()}

    for src in sources:
        convert_source(
            src,
            output_dir=args.output_dir,
            limit=args.limit,
            status_filter=status_filter,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
