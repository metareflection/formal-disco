#!/usr/bin/env python3
"""
Build a Frama-C benchmark pickle from a local checkout of ACSL by Example.

Walks ``StandardAlgorithms/<group>/`` for each group listed in
``StandardAlgorithms/subdirs.list``, picks up every ``<name>.c`` paired with
``<name>.h``, and attaches the upstream verification metadata from
``StandardAlgorithms/Results/<name>.json`` when present.

Each example becomes one ``Object(type="framac-program")`` whose content is
``<name>.h`` followed by ``<name>.c`` concatenated. Properties carry the WP
result counts and the source location.

Usage:
    python acsl_pkl.py --src ../acsl-by-example --output acsl_by_example.pkl
"""

import argparse
import json
import logging
import pickle
import re
import subprocess
import tempfile
from pathlib import Path

from agenda import Object

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_subdirs(std_algo: Path) -> list[str]:
    return [
        line.strip()
        for line in (std_algo / "subdirs.list").read_text().splitlines()
        if line.strip()
    ]


def read_examples(group_dir: Path) -> list[str]:
    f = group_dir / "examples.list"
    if not f.exists():
        return []
    return [line.strip() for line in f.read_text().splitlines() if line.strip()]


def load_result(results_dir: Path, name: str) -> dict | None:
    f = results_dir / f"{name}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    if isinstance(data, list) and data:
        data = data[0]
    return data if isinstance(data, dict) else None


def build_content(group_dir: Path, name: str) -> bytes | None:
    c_file = group_dir / f"{name}.c"
    h_file = group_dir / f"{name}.h"
    if not c_file.exists():
        return None
    parts = []
    if h_file.exists():
        parts.append(f"// === {name}.h ===\n")
        parts.append(h_file.read_text())
        parts.append("\n")
    parts.append(f"// === {name}.c ===\n")
    parts.append(c_file.read_text())
    return "".join(parts).encode("utf-8")


# Body headings look like "4.1. The find algorithm" or
# "4.2. The find2 algorithm—reuse of specification elements".
_TOP_HEADING_RE = re.compile(r"^(\d+)\.(\d+)\.\s+(.+)$", re.MULTILINE)
_NAME_FROM_TITLE_RE = re.compile(r"\bThe\s+([A-Za-z_][A-Za-z0-9_]*)\s+algorithm\b")


def extract_pdf_sections(pdf: Path) -> dict[str, str]:
    """Return {example_name: section_text} for every "The <name> algorithm" section."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["pdftotext", str(pdf), str(tmp_path)],
            check=True, capture_output=True,
        )
        # pdftotext emits a form-feed at every page break; replace with newline
        # so `^` in MULTILINE mode reliably matches headings at top of page.
        text = tmp_path.read_text().replace("\f", "\n")
    finally:
        tmp_path.unlink(missing_ok=True)

    headings = list(_TOP_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(headings):
        title = m.group(3).strip()
        name_m = _NAME_FROM_TITLE_RE.search(title)
        if not name_m:
            continue
        name = name_m.group(1)
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections[name] = text[m.start():end].strip()
    return sections


def build(src: Path, output: Path, pdf: Path | None) -> None:
    std_algo = src / "StandardAlgorithms"
    results_dir = std_algo / "Results"
    pdf_sections = extract_pdf_sections(pdf) if pdf else {}
    if pdf:
        logger.info(f"extracted {len(pdf_sections)} PDF sections from {pdf}")
    objects: dict[str, Object] = {}

    for group in read_subdirs(std_algo):
        group_dir = std_algo / group
        for name in read_examples(group_dir):
            content = build_content(group_dir, name)
            if content is None:
                logger.warning(f"skipping {group}/{name}: no .c file")
                continue

            props: dict = {
                "group": group,
                "source": "fraunhoferfokus/acsl-by-example",
                "frama_c_version": "32.0",
            }
            if name in pdf_sections:
                props["pdf_section"] = pdf_sections[name]

            result = load_result(results_dir, name)
            if result is not None:
                props.update({
                    "wp_proved": result.get("proved"),
                    "wp_total": result.get("total"),
                    "wp_percent": result.get("percent"),
                    "qed": result.get("qed"),
                    "alt_ergo": result.get("alt_ergo"),
                    "cvc5": result.get("cvc5"),
                    "z3": result.get("z3"),
                    "coq": result.get("coq"),
                })
                props["verification_status"] = (
                    "success" if result.get("percent") == 100 else "goal_unproven"
                )
            else:
                props["verification_status"] = "goal_unproven"

            path = f"dataset/acsl-by-example/{group}/{name}.c"
            objects[path] = Object(
                path=path,
                type="framac-program",
                content=content,
                properties=props,
            )

    pickle_data = {
        "objects": objects,
        "tasks": {},
        "status": {},
        "clock": 0,
    }
    with output.open("wb") as f:
        pickle.dump(pickle_data, f)
    logger.info(f"Wrote {output} with {len(objects)} programs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("../acsl-by-example"),
                        help="Path to the local acsl-by-example checkout.")
    parser.add_argument("--output", type=Path, default=Path("acsl_by_example.pkl"),
                        help="Where to write the pickle.")
    parser.add_argument("--pdf", type=Path, default=None,
                        help="Path to ACSL-by-Example.pdf (default: <src>/ACSL-by-Example.pdf). "
                             "Pass --no-pdf to skip.")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF section extraction.")
    args = parser.parse_args()
    pdf = None if args.no_pdf else (args.pdf or args.src / "ACSL-by-Example.pdf")
    build(args.src, args.output, pdf)


if __name__ == "__main__":
    main()
