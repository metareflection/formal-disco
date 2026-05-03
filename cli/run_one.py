"""One-shot: apply ONE heuristic to ONE concept and inspect the propose/check pipeline.

Phase A of PROPOSAL_EURISKO_CLI.md. No agenda persistence, no scheduler, no
queueing — every step runs synchronously with structured stdout output so
you can see exactly what the LLM proposed, what each check decided, and how
long each part took.

Example:
    python -m cli.run_one \\
      --domain modular_arithmetic \\
      --seed data/seed_modular_arithmetic.json \\
      --heuristic specialize \\
      --concept mod_op \\
      --num-conjectures 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from agenda import LocalAgenda, Object
from discovery import (
    format_concepts_for_prompt,
    gather_definitions,
    is_duplicate_statement,
    parse_conjecture_output,
    resolve_imports,
)
from discovery.checks.counterexample import check_counterexample
from discovery.checks.novelty import (
    NoveltyIndex,
    check_novelty,
    collect_agenda_corpus,
    collect_leandisco_corpus,
)
from discovery.heuristics_seed import INITIAL_HEURISTICS
from discovery.prompts import format_conjecture_user, system_conjecture
from language import Language, Program, VerificationOutcome


DEFAULT_LLM_CONFIG = {
    "_target_": "langchain_aws.ChatBedrock",
    "model_id": "us.anthropic.claude-sonnet-4-6",
    "region_name": "${oc.env:AWS_DEFAULT_REGION,us-east-1}",
    "max_tokens": 8000,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cli.run_one",
        description="Apply one heuristic to one concept and run the full check pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--domain", required=True, help="domain tag, e.g. modular_arithmetic")
    p.add_argument("--seed", required=True, help="path to a seed JSON file")
    p.add_argument(
        "--heuristic",
        required=True,
        help=f"heuristic name (one of: {', '.join(h['name'] for h in INITIAL_HEURISTICS)})",
    )
    p.add_argument("--concept", required=True, help="target concept name (must be present in the seed)")
    p.add_argument("--num-conjectures", type=int, default=5, help="max candidates from the LLM output to process")
    p.add_argument(
        "--llm-yaml",
        default=None,
        help="path to a Hydra-style yaml describing the LLM (default: AWS Bedrock Sonnet 4-6)",
    )
    p.add_argument("--no-novelty", action="store_true", help="skip the novelty check")
    p.add_argument("--no-counterexample", action="store_true", help="skip the plausible counterexample probe")
    p.add_argument(
        "--lean-project-dir",
        default=None,
        help="LEAN_PROJECT_DIR to use for typecheck/probe (default: $LEAN_PROJECT_DIR)",
    )
    p.add_argument("--language", default="lean")
    p.add_argument(
        "--counterexample-num-inst",
        type=int,
        default=50,
        help="how many random samples plausible takes per probe",
    )
    p.add_argument(
        "--counterexample-timeout",
        type=float,
        default=60.0,
        help="per-probe timeout (seconds)",
    )
    p.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.92,
        help="cosine similarity threshold above which novelty rejects",
    )
    return p.parse_args()


def _resolve_llm(args: argparse.Namespace):
    """Build the LLM client from the yaml (or the default). Lazy-imports hydra."""
    from hydra.utils import instantiate

    cfg = (
        OmegaConf.load(args.llm_yaml)
        if args.llm_yaml
        else OmegaConf.create(DEFAULT_LLM_CONFIG)
    )
    OmegaConf.resolve(cfg)
    return instantiate(cfg)


async def _seed_agenda(agenda: LocalAgenda, seeds: list[dict], domain: str) -> None:
    """Materialize each seed concept as an agenda Object (mirrors ConceptSeeder)."""
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


def _strip_proof_to_sorry(statement: str) -> str:
    """Replace any proof body with ``:= sorry`` so we can typecheck the signature only."""
    cleaned = re.sub(r":=\s*by\b.*", ":= sorry", statement, flags=re.DOTALL)
    cleaned = re.sub(r":=\s*(?!sorry).*", ":= sorry", cleaned, flags=re.DOTALL)
    if ":= sorry" not in cleaned:
        cleaned = cleaned.rstrip() + " := sorry"
    return cleaned


async def _typecheck_with_sorry(
    agenda: LocalAgenda,
    domain: str,
    backend,
    entry: dict,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Compile a stub of the candidate (with sorry as proof). Returns (ok, error_excerpt, preamble).

    Preamble is returned alongside so the counterexample step can reuse it
    rather than recompute gather_definitions a second time.
    """
    statement = entry.get("lean_statement", "")
    if not statement:
        return False, "empty statement", None

    imports = entry.get("lean_imports", [])
    imports_str = "\n".join(f"import {imp}" for imp in resolve_imports(imports))
    preamble_defs = await gather_definitions(
        agenda, domain, statement, entry.get("related_concepts", [])
    )
    preamble = "\n\n".join(preamble_defs)
    stub_src = f"{imports_str}\n\n{preamble}\n\n{_strip_proof_to_sorry(statement)}"

    try:
        ver = backend.verify(Program(stub_src, Language.LEAN, name="typecheck"), timeout=30)
    except Exception as e:  # noqa: BLE001
        return False, f"verify_error: {e}", preamble

    if ver.outcome == VerificationOutcome.FAIL:
        excerpt = (ver.stdout or ver.stderr or "")[:400]
        return False, excerpt, preamble
    return True, None, preamble


def _heuristic_template(name: str) -> Optional[dict]:
    for h in INITIAL_HEURISTICS:
        if h["name"] == name:
            return h
    return None


# ---------- pretty printing ----------------------------------------------------

def _hr(char: str = "-", n: int = 78) -> None:
    print(char * n)


def _box(label: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{label:<14}{value}")


def _truncate(text: str, n: int = 200) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + "…"


# ---------- main pipeline ------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    if args.lean_project_dir:
        os.environ["LEAN_PROJECT_DIR"] = args.lean_project_dir
    if "LEAN_PROJECT_DIR" not in os.environ and not (args.no_counterexample):
        print(
            "warning: LEAN_PROJECT_DIR is unset; typecheck and counterexample probes "
            "will fail. Pass --lean-project-dir or export the env var.",
            file=sys.stderr,
        )

    seed_path = Path(args.seed)
    if not seed_path.exists():
        print(f"error: seed file not found: {seed_path}", file=sys.stderr)
        return 2

    h_spec = _heuristic_template(args.heuristic)
    if h_spec is None:
        avail = ", ".join(h["name"] for h in INITIAL_HEURISTICS)
        print(f"error: heuristic '{args.heuristic}' not found.\nAvailable: {avail}", file=sys.stderr)
        return 2

    seeds = json.loads(seed_path.read_text())
    if not any(s["name"] == args.concept for s in seeds):
        names = ", ".join(s["name"] for s in seeds)
        print(f"error: concept '{args.concept}' not in {seed_path}.\nAvailable: {names}", file=sys.stderr)
        return 2

    # ---------- setup ----------
    agenda = LocalAgenda(language=args.language, checkpoint_path=None, checkpoint_interval=0)
    await _seed_agenda(agenda, seeds, args.domain)

    target = await agenda.get_object(f"concept/{args.domain}/{args.concept}")
    assert target is not None  # checked above

    backend = Language[args.language.upper()].get_backend()

    # Build novelty index (corpus = seeds + LeanDisco library).
    novelty_idx: Optional[NoveltyIndex] = None
    if not args.no_novelty:
        t0 = time.time()
        novelty_idx = NoveltyIndex(model_name=NoveltyIndex.DEFAULT_MODEL)
        entries = collect_agenda_corpus(agenda)
        entries += collect_leandisco_corpus(os.environ.get("LEAN_PROJECT_DIR"))
        novelty_idx.build(entries)
        print(f"[novelty] index built: {len(entries)} entries in {time.time() - t0:.1f}s")

    # ---------- propose ----------
    related_names = target.properties.get("related_concepts", [])
    context_concepts = [target]
    for n in related_names[:5]:
        rel = await agenda.get_object(f"concept/{args.domain}/{n}")
        if rel is not None:
            context_concepts.append(rel)

    concepts_text = format_concepts_for_prompt(context_concepts)
    template = h_spec["template"]

    print()
    _hr("=")
    print(f"heuristic:  {args.heuristic}  (eurisclo origin: {h_spec.get('eurisclo_origin', '-')})")
    print(f"target:     {args.concept}")
    print(f"context:    {[c.properties['name'] for c in context_concepts]}")
    _hr("=")

    llm = _resolve_llm(args)

    from langchain_core.messages import HumanMessage, SystemMessage  # local import to keep top tidy

    msgs = [
        SystemMessage(content=system_conjecture(args.domain)),
        HumanMessage(content=format_conjecture_user(template, concepts_text, args.domain)),
    ]

    t0 = time.time()
    try:
        response = llm.invoke(msgs)
    except Exception as e:  # noqa: BLE001
        print(f"error: LLM invoke failed: {e}", file=sys.stderr)
        return 3
    response_text = response.content if hasattr(response, "content") else str(response)
    t_llm = time.time() - t0

    candidates = parse_conjecture_output(response_text)[: args.num_conjectures]
    print(f"[propose]   LLM call: {t_llm:.1f}s   parsed candidates: {len(candidates)}")

    # ---------- per-candidate check pipeline ----------
    admits = 0
    rejects: dict[str, int] = {}
    total_t0 = time.time()

    for i, entry in enumerate(candidates, 1):
        name = entry.get("name", f"<unnamed_{i}>")
        kind = entry.get("kind", "conjecture")
        statement = entry.get("lean_statement", "")

        print()
        _hr("-")
        print(f"candidate {i}: {name}   (kind: {kind})")
        _box("statement:", _truncate(statement, 200))

        # 1. dedup by name
        existing = await agenda.get_object(f"concept/{args.domain}/{name}")
        if existing is not None:
            _box("dedup:", "REJECT  duplicate_name")
            rejects["duplicate_name"] = rejects.get("duplicate_name", 0) + 1
            continue

        # 2. dedup by signature
        if statement and await is_duplicate_statement(agenda, args.domain, statement):
            _box("dedup:", "REJECT  duplicate_signature")
            rejects["duplicate_signature"] = rejects.get("duplicate_signature", 0) + 1
            continue
        _box("dedup:", "PASS")

        preamble: Optional[str] = None

        # 3. typecheck (with sorry stub)
        if statement:
            t0 = time.time()
            tc_ok, tc_err, preamble = await _typecheck_with_sorry(agenda, args.domain, backend, entry)
            tc_dt = time.time() - t0
            if tc_ok:
                _box("typecheck:", f"PASS  ({tc_dt:.1f}s)")
            else:
                _box("typecheck:", f"REJECT  typecheck_failed  ({tc_dt:.1f}s)")
                if tc_err:
                    _box("", f"error: {_truncate(tc_err, 250)}")
                rejects["typecheck_failed"] = rejects.get("typecheck_failed", 0) + 1
                continue

        # 4. novelty
        if novelty_idx is not None and statement:
            t0 = time.time()
            nov = check_novelty(statement=statement, index=novelty_idx, threshold=args.novelty_threshold)
            nov_dt = time.time() - t0
            sim = nov.details.get("similarity", 0.0)
            neighbor = nov.details.get("nearest", "")
            if not nov.passed:
                _box("novelty:", f"REJECT  too_similar  cos={sim:.3f} → {neighbor}  ({nov_dt:.2f}s)")
                rejects["too_similar"] = rejects.get("too_similar", 0) + 1
                continue
            _box("novelty:", f"PASS  cos={sim:.3f}  (nearest: {neighbor})  ({nov_dt:.2f}s)")

        # 5. counterexample (conjectures only)
        if (
            not args.no_counterexample
            and kind == "conjecture"
            and statement
            and preamble is not None
        ):
            t0 = time.time()
            cex = check_counterexample(
                statement=statement,
                imports=entry.get("lean_imports", []),
                preamble=preamble,
                backend=backend,
                num_inst=args.counterexample_num_inst,
                timeout=args.counterexample_timeout,
            )
            cex_dt = time.time() - t0
            if cex.verdict == "refuted":
                _box("counterexample:", f"REJECT  refuted  ({cex_dt:.1f}s)")
                witness = (cex.witness or "").replace("\n", " | ")[:300]
                _box("", f"witness: {witness}")
                rejects["counterexample_found"] = rejects.get("counterexample_found", 0) + 1
                continue
            _box("counterexample:", f"{cex.verdict.upper()}  ({cex_dt:.1f}s)")

        admits += 1
        _box("verdict:", "ADMIT")

    # ---------- summary ----------
    total_t = time.time() - total_t0
    print()
    _hr("=")
    rej_total = sum(rejects.values())
    print(f"summary: {admits} admit / {rej_total} reject  out of {len(candidates)} candidates")
    if rejects:
        for r, n in sorted(rejects.items(), key=lambda kv: -kv[1]):
            print(f"  {n}× {r}")
    print(f"total wall-clock (checks): {total_t:.1f}s")
    print(f"LLM time: {t_llm:.1f}s")
    _hr("=")
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
