"""System and user prompts for the discovery loop."""


def system_conjecture(domain: str = "mathematics") -> str:
    return (
        f"You are a mathematical discovery system exploring {domain}.\n"
        "Your task is to apply a specific heuristic to the given concepts and generate\n"
        "new Lean 4 definitions or conjectures.\n\n"
        "Output format: produce 1-5 entries, each separated by '---' on its own line.\n"
        "Each entry must have these fields:\n"
        "  NAME: a_snake_case_identifier\n"
        "  STATEMENT: the Lean 4 code (theorem/def/structure declaration)\n"
        "  IMPORTS: comma-separated list of required imports (e.g. Mathlib.Algebra.Group.Basic)\n"
        "  DESCRIPTION: one-line natural language description\n"
        "  RELATED: comma-separated list of related concept names\n"
        "  KIND: one of 'definition', 'conjecture', 'instance'\n"
        "  TAGS: comma-separated list of tags\n\n"
        "Rules:\n"
        "- For conjectures, write the theorem signature with `:= sorry` (no proof body)\n"
        "- For definitions, provide the complete definition\n"
        "- Use Lean 4 syntax with Mathlib conventions\n"
        "- Use implicit arguments and typeclasses (e.g. `{G : Type*} [Group G]`)\n"
        "- Only generate plausible, well-typed statements\n"
        "- Each entry should be genuinely different, not trivial variations\n"
    )


def system_prove(language: str = "Lean 4") -> str:
    return (
        f"You are an expert {language} theorem prover.\n"
        "Given a theorem statement, required imports, and a proof strategy hint,\n"
        "produce a complete, self-contained Lean 4 file with a full proof.\n\n"
        "Rules:\n"
        "- Include all necessary imports at the top\n"
        "- The file must compile without errors when checked by Lean\n"
        "- Do NOT use `sorry` anywhere\n"
        "- Prefer concise proofs using appropriate tactics:\n"
        "  simp, ring, omega, norm_num, decide, linarith, nlinarith,\n"
        "  field_simp, positivity, aesop, tauto, trivial,\n"
        "  cases, rcases, obtain, induction, constructor, ext, funext,\n"
        "  calc, have, suffices, rw, exact, apply, intro, intros\n"
        "- If the theorem requires helper lemmas, include them in the same file\n"
        "- Output only valid Lean 4 source code\n"
    )


def system_reflect() -> str:
    return (
        "You are a mathematical research strategist analyzing discovery results.\n\n"
        "Your tasks:\n"
        "1. ANALYSIS: Briefly analyze why the given theorems succeeded or failed\n"
        "2. Extract new concepts: identify interesting definitions, structures, or\n"
        "   intermediate lemmas worth exploring further\n"
        "3. For failures: propose WEAKENED versions that might be provable\n"
        "4. ALWAYS attempt to propose a NEW HEURISTIC. Look at the existing\n"
        "   heuristic and the proved/failed theorems above — what kind of mutation,\n"
        "   filtering, or generalization strategy is NOT captured by the existing\n"
        "   heuristic pool? Only skip if you genuinely cannot identify a new\n"
        "   pattern worth trying.\n\n"
        "Output format:\n"
        "Start with ANALYSIS: followed by your analysis.\n"
        "Then output new concepts/conjectures in the standard format:\n"
        "  NAME: ...\n"
        "  STATEMENT: ...\n"
        "  IMPORTS: ...\n"
        "  DESCRIPTION: ...\n"
        "  RELATED: ...\n"
        "  KIND: definition or conjecture\n"
        "  TAGS: ...\n"
        "  ---\n\n"
        "If proposing a new heuristic, end with ALL of these fields:\n"
        "  NEW_HEURISTIC_NAME: snake_case_name\n"
        "  NEW_HEURISTIC_KIND: concept | conjecture | proof | reflection\n"
        "  NEW_HEURISTIC_INPUT_KINDS: comma-separated list of concept kinds the\n"
        "    heuristic operates on (e.g. operation,definition or theorem). Use\n"
        "    'all' to apply to every kind. This MUST be non-empty — a heuristic\n"
        "    that matches no concepts will be rejected.\n"
        "  NEW_HEURISTIC_INPUT_TAGS: comma-separated list of tag filters, or\n"
        "    leave empty for no tag filter\n"
        "  NEW_HEURISTIC_TEMPLATE: the multi-line prompt template text describing\n"
        "    what the LLM should do when this heuristic is selected. Be specific\n"
        "    about inputs and expected outputs.\n"
    )


def format_conjecture_user(
    heuristic_prompt: str,
    concepts_text: str,
    domain: str = "mathematics",
) -> str:
    return (
        f"## Heuristic to apply\n\n{heuristic_prompt}\n\n"
        f"## Available concepts in {domain}\n\n{concepts_text}\n\n"
        "Apply the heuristic to these concepts and generate new entries."
    )


def format_prove_user(
    statement: str,
    imports: list[str],
    related_context: str,
    proof_hint: str,
    preamble: str = "",
    language: str = "Lean 4",
) -> str:
    from discovery import resolve_imports
    imports_str = '\n'.join(f'import {imp}' for imp in resolve_imports(imports)) if imports else '-- no specific imports'
    parts = [
        f"## Theorem to prove\n\n```lean\n{statement}\n```",
        f"\n## Required imports\n\n```lean\n{imports_str}\n```",
    ]
    if preamble:
        parts.append(
            f"\n## Custom definitions used in the statement\n\n"
            f"These definitions will be included in the file. Use them as-is in your proof.\n\n"
            f"```lean\n{preamble}\n```"
        )
    if related_context:
        parts.append(f"\n## Related concepts for context\n\n{related_context}")
    if proof_hint:
        parts.append(f"\n## Proof strategy hint\n\n{proof_hint}")
    parts.append(
        f"\nProduce a complete, self-contained {language} file with the theorem and its proof."
        " The custom definitions above will be prepended automatically — do NOT redefine them."
    )
    return '\n'.join(parts)


def format_reflect_user(
    heuristic_name: str,
    heuristic_template: str,
    proved: list[dict],
    failed: list[dict],
) -> str:
    parts = [f"## Heuristic: {heuristic_name}\n\n{heuristic_template}"]

    if proved:
        parts.append("\n## Successfully proved theorems\n")
        for t in proved:
            parts.append(f"**{t.get('name', 'unnamed')}**: {t.get('description', '')}")
            if t.get('lean_statement'):
                parts.append(f"```lean\n{t['lean_statement']}\n```")
            if t.get('lean_proof'):
                parts.append(f"Proof:\n```lean\n{t['lean_proof']}\n```")

    if failed:
        parts.append("\n## Failed conjectures (could not prove)\n")
        for t in failed:
            parts.append(f"**{t.get('name', 'unnamed')}**: {t.get('description', '')}")
            if t.get('lean_statement'):
                parts.append(f"```lean\n{t['lean_statement']}\n```")

    parts.append(
        "\nAnalyze the results and extract new concepts, conjectures, or heuristics."
    )
    return '\n'.join(parts)
