#!/usr/bin/env python3
"""
Generate distillation data for the implementer worker.

Takes verified Dafny programs with idea prompts and creates training examples
for generating programs from specifications.

Usage:
    python implementer_distill.py generate --output implementer.pkl
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from agenda import Object


def extract_idea_prompt(program: str) -> Optional[str]:
    """Extract the idea prompt from the program comment."""
    match = re.search(r'//\s*Idea prompt:\s*(.+)', program)
    if match:
        return match.group(1).strip()
    return None


def remove_idea_comment(program: str) -> str:
    """Remove the idea prompt comment from the program."""
    return re.sub(r'//\s*Idea prompt:[^\n]*\n?', '', program).strip()


def generate_implement_examples(
    input_dir: Path,
    max_files: Optional[int] = None,
) -> list[dict]:
    """Generate implementation examples from verified programs."""

    dfy_files = sorted(input_dir.glob("*.dfy"))
    if max_files:
        dfy_files = dfy_files[:max_files]

    examples = []
    skipped_no_idea = 0

    for dfy_file in tqdm(dfy_files, desc="Processing files"):
        program = dfy_file.read_text()

        idea = extract_idea_prompt(program)
        if not idea:
            skipped_no_idea += 1
            continue

        # The response is the full program (without the idea comment)
        clean_program = remove_idea_comment(program)

        examples.append({
            "prompt": "implement",
            "arguments": {
                "idea": idea,
            },
            "response": clean_program,
            "outcome": "success",
            "metadata": {
                "source": "free_autogen",
                "file": dfy_file.name,
            }
        })

    if skipped_no_idea:
        print(f"Skipped {skipped_no_idea} files (no idea prompt)")

    return examples


def create_agenda_pickle(examples: list[dict], output_path: Path) -> None:
    """Create a pickle file in the format expected by distill.py."""
    objects = {}

    for i, ex in enumerate(examples):
        path = f"distil/implement_{i:04d}"
        content = json.dumps(ex, ensure_ascii=False).encode("utf-8")
        obj = Object(
            path=path,
            type="distil-example",
            content=content,
        )
        objects[path] = obj

    pickle_data = {
        "objects": objects,
        "tasks": {},
        "task_statuses": {},
    }

    with output_path.open("wb") as f:
        pickle.dump(pickle_data, f)


def cmd_generate(args):
    """Generate implementer distillation data."""
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    print(f"Generating examples from {input_dir}")

    examples = generate_implement_examples(
        input_dir,
        max_files=args.max_files,
    )

    print(f"Generated {len(examples)} implement examples")

    create_agenda_pickle(examples, output_path)
    print(f"Saved to {output_path}")

    if examples:
        print("\n--- Sample example ---")
        sample = examples[0]
        print(f"File: {sample['metadata']['file']}")
        print(f"Idea: {sample['arguments']['idea']}")
        print(f"\nProgram (first 300 chars):\n{sample['response'][:300]}...")


def cmd_list(args):
    """List idea prompts from files."""
    input_dir = Path(args.input_dir)

    for dfy_file in sorted(input_dir.glob("*.dfy"))[:args.max_files or 10]:
        program = dfy_file.read_text()
        idea = extract_idea_prompt(program)
        if idea:
            print(f"{dfy_file.name}: {idea[:70]}...")
        else:
            print(f"{dfy_file.name}: (no idea prompt)")


def main():
    parser = argparse.ArgumentParser(description="Generate implementer distillation data")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_p = sub.add_parser("generate", help="Generate implementer distillation data")
    gen_p.add_argument("--input-dir", type=str, default="../dafny-free-autogen",
                       help="Directory containing verified .dfy files with idea prompts")
    gen_p.add_argument("--output", "-o", type=str, default="implementer.pkl",
                       help="Output pickle file path")
    gen_p.add_argument("--max-files", type=int, default=None,
                       help="Maximum number of files to process")
    gen_p.set_defaults(func=cmd_generate)

    list_p = sub.add_parser("list", help="List idea prompts from files")
    list_p.add_argument("--input-dir", type=str, default="../dafny-free-autogen",
                        help="Directory containing .dfy files")
    list_p.add_argument("--max-files", type=int, default=10,
                        help="Maximum files to show")
    list_p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
