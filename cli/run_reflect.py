"""One-shot reflection: exercise the Phase 2 soundness gate end-to-end.

Loads a (proved theorem, origin heuristic) pair from a seed file (and optionally
some failed conjectures), calls the reflection LLM with the same system prompt
the live ReflectionWorker uses, parses the output via ``parse_reflection_output``,
and — if a new heuristic was proposed — runs ``HeuristicSoundnessCheck`` against
the seed as the held-out harness. Prints every step.

Existence rationale: the live system fires reflection rarely (gated on a prove
landing) and the reflection prompt only proposes new heuristics
probabilistically. This tool lets you iterate on the reflection prompt and the
soundness gate in seconds rather than waiting hours for one to fire in the wild.

Example:
    python -m cli.run_reflect \\
      --domain modular_arithmetic \\
      --seed data/seed_modular_arithmetic.json \\
      --proved mod_op_self_zero gcd_op_zero_right \\
      --origin-heuristic boundary_cases
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from agenda import LocalAgenda, Object
from discovery import parse_reflection_output
from discovery.checks.soundness import (
    check_heuristic_soundness,
    collect_existing_heuristics,
    collect_harness,
)
from discovery.heuristic import (
    AppliesToSpec,
    CheckSpec,
    CommitSpec,
    Heuristic,
    ProposeSpec,
)
from discovery.heuristics_seed import INITIAL_HEURISTICS
from discovery.prompts import format_reflect_user, system_reflect


DEFAULT_LLM_CONFIG = {
    "_target_": "langchain_aws.ChatBedrock",
    "model_id": "us.anthropic.claude-sonnet-4-6",
    "region_name": "${oc.env:AWS_DEFAULT_REGION,us-east-1}",
    "max_tokens": 8000,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cli.run_reflect",
        description="One reflection LLM call + Phase 2 soundness gate verdict, end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--domain", required=True)
    p.add_argument("--seed", required=True, help="path to seed JSON file (also used as harness)")
    p.add_argument(
        "--proved",
        nargs="+",
        required=True,
        help="one or more concept names from the seed to feed in as proved theorems",
    )
    p.add_argument(
        "--failed",
        nargs="*",
        default=[],
        help="optional list of concept names to feed in as failed conjectures",
    )
    p.add_argument(
        "--origin-heuristic",
        required=True,
        help=(
            "name of the heuristic the proved theorems trace back to "
            f"(one of: {', '.join(h['name'] for h in INITIAL_HEURISTICS)})"
        ),
    )
    p.add_argument("--llm-yaml", default=None)
    p.add_argument(
        "--no-soundness-check",
        action="store_true",
        help="skip running the Phase 2 gate (just print the parsed reflection output)",
    )
    p.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the full reflection prompt sent to the LLM",
    )
    p.add_argument(
        "--show-response",
        action="store_true",
        help="print the full raw LLM response (otherwise only the parsed structure)",
    )
    p.add_argument("--language", default="lean")
    return p.parse_args()


def _resolve_llm(args: argparse.Namespace):
    from hydra.utils import instantiate

    cfg = (
        OmegaConf.load(args.llm_yaml)
        if args.llm_yaml
        else OmegaConf.create(DEFAULT_LLM_CONFIG)
    )
    OmegaConf.resolve(cfg)
    return instantiate(cfg)


async def _seed_agenda(agenda: LocalAgenda, seeds: list[dict], domain: str) -> None:
    for s in seeds:
        await agenda.create_object(Object(
            path=f"concept/{domain}/{s['name']}",
            type="concept",
            content=s.get("description", "").encode("utf-8"),
            properties={
                "name": s["name"],
                "kind": s["kind"],
                "domain": domain,
                "description": s.get("description", ""),
                "lean_statement": s["lean_statement"],
                "lean_imports": s.get("lean_imports", []),
                "tags": s.get("tags", []),
                "related_concepts": s.get("related", []),
                "origin_heuristic": None,
                "proof_attempts": 0,
                "proof_strategy": None,
            },
        ))
    # Also seed the heuristic templates so HeuristicSoundnessCheck has them for similarity dedup.
    for h in INITIAL_HEURISTICS:
        await agenda.create_object(Object(
            path=f"heuristic/{h['name']}",
            type="heuristic",
            content=h["template"].encode("utf-8"),
            properties={
                "name": h["name"],
                "heuristic_kind": h["heuristic_kind"],
                "input_concept_kinds": list(h.get("input_concept_kinds", [])),
                "input_tags": list(h.get("input_tags", [])),
                "eurisclo_origin": h.get("eurisclo_origin"),
                "attempts": 0,
                "successes": 0,
            },
        ))


def _hr(c: str = "-", n: int = 78) -> None:
    print(c * n)


def _truncate(s: str, n: int = 200) -> str:
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "…"


async def _run(args: argparse.Namespace) -> int:
    # Validate inputs
    seed_path = Path(args.seed)
    if not seed_path.exists():
        print(f"error: seed file not found: {seed_path}", file=sys.stderr)
        return 2
    seeds = json.loads(seed_path.read_text())
    by_name = {s["name"]: s for s in seeds}

    missing = [n for n in args.proved if n not in by_name]
    if missing:
        print(f"error: not in seed: {missing}\nAvailable: {list(by_name)}", file=sys.stderr)
        return 2
    missing_failed = [n for n in args.failed if n not in by_name]
    if missing_failed:
        print(f"error: not in seed: {missing_failed}", file=sys.stderr)
        return 2

    h_spec = next((h for h in INITIAL_HEURISTICS if h["name"] == args.origin_heuristic), None)
    if h_spec is None:
        avail = ", ".join(h["name"] for h in INITIAL_HEURISTICS)
        print(f"error: heuristic '{args.origin_heuristic}' not found.\nAvailable: {avail}", file=sys.stderr)
        return 2

    # In-memory agenda with seed + heuristics
    agenda = LocalAgenda(language=args.language, checkpoint_path=None, checkpoint_interval=0)
    await _seed_agenda(agenda, seeds, args.domain)

    # Build proved/failed payloads as ReflectionWorker would
    def _to_payload(name: str) -> dict:
        s = by_name[name]
        return {
            "name": s["name"],
            "description": s.get("description", ""),
            "lean_statement": s.get("lean_statement", ""),
            "lean_proof": s.get("lean_proof", ""),  # mostly empty in seeds; live system fills it
        }

    proved = [_to_payload(n) for n in args.proved]
    failed = [_to_payload(n) for n in args.failed]

    # Header
    _hr("=")
    print(f"reflection on:")
    print(f"  origin heuristic: {h_spec['name']} ({h_spec.get('eurisclo_origin', '-')})")
    print(f"  proved:           {args.proved}")
    if args.failed:
        print(f"  failed:           {args.failed}")
    _hr("=")

    # Build prompt
    sys_msg = system_reflect()
    usr_msg = format_reflect_user(h_spec["name"], h_spec["template"], proved, failed)
    if args.show_prompt:
        _hr()
        print("[system prompt]")
        print(sys_msg)
        _hr()
        print("[user prompt]")
        print(usr_msg)
        _hr()

    # Invoke LLM
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _resolve_llm(args)
    t0 = time.time()
    try:
        response = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=usr_msg)])
    except Exception as e:  # noqa: BLE001
        print(f"error: LLM invoke failed: {e}", file=sys.stderr)
        return 3
    response_text = response.content if hasattr(response, "content") else str(response)
    t_llm = time.time() - t0

    print(f"[reflection LLM] {t_llm:.1f}s")
    if args.show_response:
        _hr()
        print("[raw LLM response]")
        print(response_text)
        _hr()

    # Parse
    parsed = parse_reflection_output(response_text)

    print()
    print(f"analysis: {_truncate(parsed.get('analysis', '<none>'), 400)}")
    print()
    concepts = parsed.get("concepts", [])
    print(f"extracted concepts ({len(concepts)}):")
    for c in concepts:
        print(f"  - {c.get('name','?'):35s} kind={c.get('kind','?')}  → {_truncate(c.get('lean_statement',''), 100)}")

    new_h = parsed.get("new_heuristic")
    if not new_h:
        print()
        print("new_heuristic: <none proposed>")
        print()
        _hr("=")
        print("reflection produced no new-heuristic proposal — Phase 2 gate not exercised.")
        return 0

    print()
    print("new_heuristic proposed:")
    for k, v in new_h.items():
        if k == "template":
            print(f"  template:")
            for line in str(v).splitlines():
                print(f"    {line}")
        else:
            print(f"  {k}: {v}")

    if args.no_soundness_check:
        return 0

    # Run the Phase 2 gate
    print()
    _hr()
    print("[soundness gate]")

    proposed = Heuristic(
        name=str(new_h.get("name", "")),
        heuristic_kind=str(new_h.get("kind", "concept")),
        applies_to=AppliesToSpec(
            concept_kinds=list(new_h.get("input_concept_kinds", []) or []),
            tags=list(new_h.get("input_tags", []) or []),
        ),
        propose=ProposeSpec(kind="prompt_template", template=str(new_h.get("template", ""))),
        check=CheckSpec(),
        commit=CommitSpec(),
        born_from_reflection=True,
    )
    # Exclude the proved theorems from the harness so the new heuristic doesn't
    # match its own input — same as ReflectionWorker.
    excludes = tuple(f"concept/{args.domain}/{n}" for n in args.proved)
    harness = collect_harness(agenda, exclude_paths=excludes, max_size=30)
    existing = collect_existing_heuristics(agenda)

    t0 = time.time()
    verdict = check_heuristic_soundness(
        proposed=proposed, harness=harness, existing_heuristics=existing,
    )
    t_gate = time.time() - t0

    status = "PASS" if verdict.passed else "FAIL"
    print(f"  verdict:  {status}  ({verdict.verdict})  in {t_gate*1000:.0f}ms")
    print(f"  reason:   {verdict.reason}")
    if verdict.witness:
        print(f"  witness:  {verdict.witness}")
    if verdict.details:
        for k, v in verdict.details.items():
            print(f"  {k}: {v}")

    print()
    _hr("=")
    if verdict.passed:
        print("→ this heuristic would be ADMITTED into the agenda by Phase 2.")
    else:
        print("→ this heuristic would be REJECTED by Phase 2.")
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
