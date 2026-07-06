#!/usr/bin/env python3
"""Extract verified Verus programs from the microsoft/Verus_Training_Data HF
datasets (SAFE and VeruSyn) into directories of .rs files, so they can be fed
to `analysis.py plot-rarefaction` as plain source directories.

Each dataset entry is `{"prompt", "input", "output"}` where `output` is the
verified solution wrapped in a ```rust ... ``` code block. We pull that block.

- SAFE  (sft_safe_25k.json, ~59 MB): downloaded in full and parsed.
- VeruSyn (sft_part1_6.9M.json, ~15.7 GB): streamed over HTTP and parsed
  incrementally, stopping after --max programs so we never pull the whole file.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

REPO = "microsoft/Verus_Training_Data"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

_RUST_BLOCK = re.compile(r"```(?:rust|verus)?\s*\n(.*?)```", re.DOTALL)


def extract_code(output: str) -> str | None:
    """Pull the (first) rust code block out of an `output` field."""
    m = _RUST_BLOCK.search(output)
    if m is None:
        return None
    code = m.group(1).strip()
    return code or None


def write_programs(codes, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for code in codes:
        (out_dir / f"prog_{n:06d}.rs").write_text(code, encoding="utf-8")
        n += 1
    return n


def iter_safe(max_n: int):
    """Yield verified programs from the (small) SAFE file, loaded in full."""
    path = hf_hub_download(REPO, "sft_safe_25k.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    for entry in data:
        if max_n is not None and max_n >= 0 and getattr(iter_safe, "_n", 0) >= max_n:
            break
        code = extract_code(entry.get("output", ""))
        if code:
            iter_safe._n = getattr(iter_safe, "_n", 0) + 1
            yield code


def iter_streamed_array(url: str, max_n: int):
    """Stream a JSON array of objects over HTTP, decoding top-level objects
    incrementally and yielding each entry's extracted rust block. Stops once
    `max_n` programs have been yielded (so only a prefix of the file is read).
    """
    dec = json.JSONDecoder()
    buf = ""
    started = False  # have we consumed the opening '['
    yielded = 0

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            buf += chunk.decode("utf-8", errors="replace")

            while True:
                # Trim leading whitespace / structural chars between objects.
                i = 0
                while i < len(buf) and buf[i] in " \t\r\n,":
                    i += 1
                if not started:
                    if i < len(buf) and buf[i] == "[":
                        i += 1
                        started = True
                        while i < len(buf) and buf[i] in " \t\r\n,":
                            i += 1
                    else:
                        buf = buf[i:]
                        break
                buf = buf[i:]
                if not buf or buf[0] == "]":
                    break
                try:
                    obj, end = dec.raw_decode(buf)
                except json.JSONDecodeError:
                    break  # need more bytes
                buf = buf[end:]
                code = extract_code(obj.get("output", "")) if isinstance(obj, dict) else None
                if code:
                    yield code
                    yielded += 1
                    if yielded >= max_n:
                        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["safe", "verusyn"])
    ap.add_argument("--out", required=True, help="Output directory for .rs files")
    ap.add_argument("--max", type=int, default=100_000,
                    help="Max programs to extract (default 100000)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.dataset == "safe":
        codes = iter_safe(args.max)
    else:
        url = f"{RESOLVE}/sft_part1_6.9M.json"
        codes = iter_streamed_array(url, args.max)

    # Stream straight to disk so VeruSyn never accumulates in memory.
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for code in codes:
        (out_dir / f"prog_{n:06d}.rs").write_text(code, encoding="utf-8")
        n += 1
        if n % 10_000 == 0:
            print(f"  ... {n} programs written", flush=True)
    print(f"Wrote {n} programs to {out_dir}")


if __name__ == "__main__":
    main()
