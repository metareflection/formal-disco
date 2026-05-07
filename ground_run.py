#!/usr/bin/env python3
"""Cross-reference alignment results against proved theorems in a pickle.

Run after `align_run.py --mode vs-seed` to classify each proved theorem as:
  - seed_only: stated entirely in seed vocabulary
  - fully_aliased: uses invented concepts but all of them alias back to seed
  - partially_novel: uses ≥1 invented concept that did NOT align to any seed

Usage:
    python ground_run.py \\
        --pickle agenda-discovery-matroid.pkl \\
        --alignment outputs/align-matroid/alignment.json \\
        --seed data/seed_matroid_theory.json \\
        --domain matroid_theory \\
        --output outputs/align-matroid/grounding.md
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from align.grounding import ground_theorems, write_grounding_report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pickle", required=True, type=Path)
    p.add_argument("--alignment", required=True, type=Path,
                   help="Path to alignment.json output by align_run.py")
    p.add_argument("--seed", required=True, type=Path)
    p.add_argument("--domain", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    seed = json.loads(args.seed.read_text())
    seed_names = {c["name"] for c in seed}

    groundings = ground_theorems(args.pickle, args.alignment, args.domain, seed_names)
    write_grounding_report(groundings, args.output)

    cats = Counter(g.category for g in groundings)
    print(f"Wrote {args.output}")
    print("=" * 60)
    print(f"Total proved theorems: {len(groundings)}")
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
