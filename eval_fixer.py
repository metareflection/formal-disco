#!/usr/bin/env python3
"""
Evaluation script for the LLM Fixer on DafnyBench.

This script faithfully extracts the core llm_fixer logic (without the agenda
abstraction) and evaluates it on adding annotations to DafnyBench programs.

The logic is copied directly from worker/llm_fixer.py.

Usage:
    # Using a simple model name (convenience)
    python eval_fixer.py --model gpt-4o --num-programs 50 --max-attempts 3

    # Using a Hydra LLM config (from config/llm/)
    python eval_fixer.py --llm-config openai --num-programs 50
    python eval_fixer.py --llm-config aws --num-programs 50
    python eval_fixer.py --llm-config vllm --num-programs 50

    # Using a custom config file path
    python eval_fixer.py --llm-config config/llm/openai.yaml --num-programs 50

    # Hydra overrides
    python eval_fixer.py --llm-config openai --llm-override "model=gpt-4-turbo"
"""

import os
import json
import argparse
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
from tqdm import tqdm

# Import from existing formal-disco modules (same as llm_fixer.py)
from langchain_core.prompts import ChatPromptTemplate
from code_output_parser import CodeOutputParser
from prompt import format_repair_user, system_repair
from dafny import DafnyProgram, VerificationOutcome
from patch import apply_text_diff, TEXT_DIFF_EXAMPLE, TEXT_BEFORE_EXAMPLE, TEXT_AFTER_EXAMPLE

# Note: langchain_openai and langchain_anthropic are imported lazily in create_llm()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FixerResult:
    """Result of a single fixer run on a program."""
    program_name: str
    original_program: str
    final_program: str
    success: bool
    num_attempts: int
    verification_outcome: str
    verification_stdout: str = ""
    verification_stderr: str = ""
    error: Optional[str] = None
    diffs_applied: list = field(default_factory=list)


class DafnyFixer:
    """
    Core fixer logic extracted directly from worker/llm_fixer.py.

    This is a faithful copy of the LLMFixer class, but without the
    Agenda/Task/Object abstractions - it operates directly on program strings.
    """

    def __init__(
        self,
        llm: Any,
        max_attempts: int = 3,
        prompt_template: Optional[ChatPromptTemplate] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the fixer.

        Args:
            llm: A LangChain LLM/ChatModel instance
            max_attempts: Maximum number of repair attempts
            prompt_template: Optional custom prompt template
            verbose: Enable verbose logging
        """
        self._llm = llm
        self._max_attempts = max_attempts
        self._verbose = verbose

        # Build prompt exactly as in llm_fixer.py lines 45-61
        if prompt_template is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_repair(
                            example_before="{example_before}",
                            example_diff="{example_diff}",
                            example_after="{example_after}",
                        ),
                    ),
                    (
                        "human",
                        format_repair_user(program="{program}", notes="{notes}"),
                    ),
                ]
            )
        else:
            self._prompt = prompt_template

        # Build the chain exactly as in llm_fixer.py line 65
        self._chain = self._prompt | self._llm | CodeOutputParser()

    def fix(
        self,
        prog_text: str,
        program_name: str = "program",
        initial_stdout: str = "",
        initial_stderr: str = "",
    ) -> FixerResult:
        """
        Attempt to fix a Dafny program through iterative LLM-guided repair.

        This follows the exact logic from llm_fixer.py work() method,
        but without the agenda/task management.

        Args:
            prog_text: The Dafny program source code
            program_name: Name for logging purposes
            initial_stdout: Initial verification stdout (if already computed)
            initial_stderr: Initial verification stderr (if already computed)

        Returns:
            FixerResult with the outcome
        """
        current_text = prog_text
        diffs_applied = []

        # Get initial verification output if not provided
        ver_stdout = initial_stdout
        ver_stderr = initial_stderr

        if not ver_stdout and not ver_stderr:
            prog = DafnyProgram(current_text, name=program_name)
            ver = prog.verify()
            ver_stdout = ver.stdout
            ver_stderr = ver.stderr

            # Check if already verified
            if ver.outcome == VerificationOutcome.SUCCESS:
                return FixerResult(
                    program_name=program_name,
                    original_program=prog_text,
                    final_program=current_text,
                    success=True,
                    num_attempts=0,
                    verification_outcome="SUCCESS",
                    verification_stdout=ver_stdout,
                    verification_stderr=ver_stderr,
                    diffs_applied=diffs_applied,
                )

        for attempt in range(self._max_attempts):
            if self._verbose:
                logger.info(f"Attempt {attempt + 1}/{self._max_attempts} for {program_name}")

            # Build prompt notes exactly as in llm_fixer.py lines 104
            prompt_notes = f"Output of dafny verify on this program:\nstdout:\n{ver_stdout}\n\nstderr:\n{ver_stderr}\n"

            # Build llm_args exactly as in llm_fixer.py lines 106-112
            llm_args = {
                "program": current_text,
                "notes": prompt_notes,
                "example_diff": TEXT_DIFF_EXAMPLE,
                "example_before": TEXT_BEFORE_EXAMPLE,
                "example_after": TEXT_AFTER_EXAMPLE,
            }

            # Ask LLM to produce a diff (llm_fixer.py line 115)
            try:
                diff_text = self._chain.invoke(llm_args).strip()
                diffs_applied.append(diff_text)

                if self._verbose:
                    logger.info(f"LLM produced diff:\n{diff_text[:500]}...")
            except Exception as e:
                logger.warning(f"LLM call failed for {program_name}: {e}")
                continue

            # Apply diff; if it fails, continue to next attempt (llm_fixer.py lines 118-127)
            try:
                repaired_text = apply_text_diff(current_text, diff_text)
            except Exception as e:
                logger.warning(f"Failed to apply diff for {program_name}: {e}")
                if self._verbose:
                    logger.info(f"Diff that failed:\n{diff_text}")
                continue

            # Verify repaired program (llm_fixer.py lines 129-150)
            repaired_prog = DafnyProgram(repaired_text, name=program_name)

            if self._verbose:
                short_repaired = repaired_text if len(repaired_text) < 2000 else repaired_text[:2000] + "..."
                logger.info(f"Verifying repaired program: {short_repaired}")

            ver = repaired_prog.verify()

            if self._verbose:
                logger.info(f"Verification outcome: {ver.outcome.name}")
                logger.info(f"Program before fix:\n{current_text}")
                logger.info(f"Dafny output before fix:\n{prompt_notes}")
                logger.info(f"Diff produced by LLM:\n{diff_text}")
                logger.info(f"Repaired program:\n{repaired_text}")
                logger.info(f"Dafny stdout after fix:\n{ver.stdout}")
                logger.info(f"Dafny stderr after fix:\n{ver.stderr}")

            # Update current state
            current_text = repaired_text
            ver_stdout = ver.stdout
            ver_stderr = ver.stderr

            # Check if successful (llm_fixer.py lines 168-174)
            if ver.outcome == VerificationOutcome.SUCCESS:
                return FixerResult(
                    program_name=program_name,
                    original_program=prog_text,
                    final_program=current_text,
                    success=True,
                    num_attempts=attempt + 1,
                    verification_outcome="SUCCESS",
                    verification_stdout=ver_stdout,
                    verification_stderr=ver_stderr,
                    diffs_applied=diffs_applied,
                )

        # All attempts exhausted - return final state
        return FixerResult(
            program_name=program_name,
            original_program=prog_text,
            final_program=current_text,
            success=False,
            num_attempts=self._max_attempts,
            verification_outcome=ver.outcome.name if ver else "UNKNOWN",
            verification_stdout=ver_stdout,
            verification_stderr=ver_stderr,
            diffs_applied=diffs_applied,
        )


def load_dafnybench_programs(
    benchmark_path: str,
    use_hints_removed: bool = True,
    seed: str = 'dafny-fixer-eval',
) -> list[tuple[str, str]]:
    """
    Load programs from DafnyBench.

    Args:
        benchmark_path: Path to DafnyBench directory
        use_hints_removed: If True, load from hints_removed/; else from ground_truth/
        seed: Random seed for shuffling

    Returns:
        List of (program_name, program_text) tuples
    """
    benchmark_path = Path(benchmark_path)

    # Try different directory structures
    if use_hints_removed:
        candidates = [
            benchmark_path / "DafnyBench" / "dataset" / "hints_removed",
            benchmark_path / "dataset" / "hints_removed",
            benchmark_path / "hints_removed",
        ]
    else:
        candidates = [
            benchmark_path / "DafnyBench" / "dataset" / "ground_truth",
            benchmark_path / "dataset" / "ground_truth",
            benchmark_path / "ground_truth",
        ]

    programs_dir = None
    for candidate in candidates:
        if candidate.exists():
            programs_dir = candidate
            break

    if programs_dir is None:
        raise FileNotFoundError(f"Could not find programs directory. Tried: {candidates}")

    programs = []
    for dfy_file in programs_dir.glob("*.dfy"):
        program_text = dfy_file.read_text()
        programs.append((dfy_file.stem, program_text))

    # Sort and shuffle deterministically (matching dafny-annotator's approach)
    programs.sort(key=lambda x: x[0])
    random.seed(seed)
    random.shuffle(programs)

    logger.info(f"Loaded {len(programs)} programs from {programs_dir}")
    return programs


def filter_nontrivial_programs(
    programs: list[tuple[str, str]],
    cache_path: Optional[str] = None,
) -> list[tuple[str, str]]:
    """
    Filter to programs that don't already verify.

    Args:
        programs: List of (name, text) tuples
        cache_path: Optional path to cache verification outcomes

    Returns:
        Filtered list of programs that need fixing
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    nontrivial = []

    for name, text in tqdm(programs, desc="Filtering trivial programs"):
        if name in cache:
            needs_fixing = cache[name] != "SUCCESS"
        else:
            prog = DafnyProgram(text, name=name)
            ver = prog.verify()
            cache[name] = ver.outcome.name
            needs_fixing = (ver.outcome != VerificationOutcome.SUCCESS)

            if cache_path:
                with open(cache_path, 'w') as f:
                    json.dump(cache, f, indent=2)

        if needs_fixing:
            nontrivial.append((name, text))

    return nontrivial


def create_llm(model_name: str, temperature: float = 0.0):
    """
    Create a LangChain LLM instance based on the model name.

    Supports:
    - OpenAI models (gpt-4o, gpt-4-turbo, etc.)
    - Anthropic models (claude-3-5-sonnet-*, etc.)
    """
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported prefixes: gpt-*, o1*, o3*, claude-*")


def load_llm_from_config(
    config_name: str,
    overrides: Optional[list[str]] = None,
    llm_key: str = "code",
):
    """
    Load a LangChain LLM from a Hydra config file.

    Args:
        config_name: Either a config name (e.g., "openai", "aws", "vllm")
                     or a path to a YAML file (e.g., "config/llm/openai.yaml")
        overrides: Optional list of Hydra overrides (e.g., ["model=gpt-4-turbo"])
        llm_key: Which LLM to use from the config ("code" or "write")

    Returns:
        A LangChain LLM instance
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    overrides = overrides or []

    # Check if it's a file path or a config name
    if config_name.endswith(".yaml") or config_name.endswith(".yml"):
        # Load directly from file path
        config_path = Path(config_name)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_name}")
        cfg = OmegaConf.load(config_path)
    else:
        # Load from config/llm/ directory
        config_dir = Path(__file__).parent / "config" / "llm"
        config_path = config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"LLM config '{config_name}' not found. "
                f"Available configs: {[p.stem for p in config_dir.glob('*.yaml')]}"
            )
        cfg = OmegaConf.load(config_path)

    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        # Apply overrides to the specific llm_key
        if llm_key in cfg:
            cfg[llm_key] = OmegaConf.merge(cfg[llm_key], override_cfg)

    # Get the specific LLM config
    if llm_key not in cfg:
        available_keys = list(cfg.keys())
        raise KeyError(
            f"LLM key '{llm_key}' not found in config. Available: {available_keys}"
        )

    llm_cfg = cfg[llm_key]
    logger.info(f"Loading LLM from config: {config_name} (key={llm_key})")
    logger.info(f"LLM config: {OmegaConf.to_yaml(llm_cfg)}")

    # Instantiate the LLM
    return instantiate(llm_cfg)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLM Fixer on DafnyBench (faithful extraction of llm_fixer logic)'
    )

    # LLM configuration (mutually exclusive: --model or --llm-config)
    llm_group = parser.add_mutually_exclusive_group(required=True)
    llm_group.add_argument('--model', type=str,
                           help='LLM model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)')
    llm_group.add_argument('--llm-config', type=str,
                           help='Hydra LLM config name (e.g., openai, aws, vllm) or path to YAML file')

    parser.add_argument('--llm-override', type=str, action='append', default=[],
                        help='Hydra-style overrides for LLM config (e.g., "model=gpt-4-turbo")')
    parser.add_argument('--llm-key', type=str, default='code',
                        help='Which LLM to use from config: "code" or "write" (default: code)')

    parser.add_argument('--benchmark-path', type=str, default='DafnyBench',
                        help='Path to DafnyBench directory')
    parser.add_argument('--num-programs', type=int, default=None,
                        help='Number of programs to evaluate (default: all)')
    parser.add_argument('--fraction', type=float, default=None,
                        help='Fraction of dataset to use, e.g. 0.1 for 10%% (applied after filtering)')
    parser.add_argument('--max-attempts', type=int, default=3,
                        help='Maximum repair attempts per program')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--cache-path', type=str, default='.fixer_outcome_cache.json',
                        help='Path to verification outcome cache')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM temperature (only used with --model)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip first N programs (for train/test split)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip filtering of trivial programs')
    args = parser.parse_args()

    # Load programs
    logger.info(f"Loading programs from {args.benchmark_path}")
    programs = load_dafnybench_programs(args.benchmark_path, use_hints_removed=True)

    # Skip first N (for train/test split)
    if args.skip > 0:
        programs = programs[args.skip:]
        logger.info(f"Skipped first {args.skip} programs, {len(programs)} remaining")

    # Filter to nontrivial (programs that don't already verify)
    if not args.no_filter:
        programs = filter_nontrivial_programs(programs, args.cache_path)
        logger.info(f"Found {len(programs)} nontrivial programs")

    # Limit by fraction or num_programs
    if args.fraction is not None:
        num_to_use = max(1, int(len(programs) * args.fraction))
        programs = programs[:num_to_use]
        logger.info(f"Using {args.fraction*100:.0f}% of dataset: {len(programs)} programs")
    elif args.num_programs is not None:
        programs = programs[:args.num_programs]
        logger.info(f"Evaluating on {len(programs)} programs")
    else:
        logger.info(f"Evaluating on all {len(programs)} programs")

    if not programs:
        logger.error("No programs to evaluate!")
        return

    # Create LLM - either from model name or from config
    if args.model:
        llm = create_llm(args.model, args.temperature)
        model_desc = args.model
    else:
        llm = load_llm_from_config(
            args.llm_config,
            overrides=args.llm_override if args.llm_override else None,
            llm_key=args.llm_key,
        )
        model_desc = f"{args.llm_config}:{args.llm_key}"

    fixer = DafnyFixer(llm, max_attempts=args.max_attempts, verbose=args.verbose)

    # Run evaluation
    results = []
    success_count = 0

    for name, text in tqdm(programs, desc="Fixing programs"):
        try:
            result = fixer.fix(text, program_name=name)
            results.append({
                "program_name": result.program_name,
                "success": result.success,
                "num_attempts": result.num_attempts,
                "verification_outcome": result.verification_outcome,
                "original_program": result.original_program,
                "final_program": result.final_program,
                "diffs_applied": result.diffs_applied,
            })
            if result.success:
                success_count += 1
                logger.info(f"✓ {name} verified after {result.num_attempts} attempts")
            else:
                logger.info(f"✗ {name} failed ({result.verification_outcome})")
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "program_name": name,
                "success": False,
                "error": str(e),
            })

    # Print summary
    print("\n" + "="*60)
    print(f"RESULTS: {success_count}/{len(programs)} programs verified")
    print(f"Success rate: {100*success_count/len(programs):.1f}%")
    print("="*60)

    # Save results
    if args.output:
        output_data = {
            "model": model_desc,
            "llm_config": args.llm_config,
            "llm_overrides": args.llm_override,
            "max_attempts": args.max_attempts,
            "num_programs": len(programs),
            "success_count": success_count,
            "success_rate": success_count / len(programs) if programs else 0,
            "results": results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
