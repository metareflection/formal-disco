"""Discovery utilities: concept formatting, output parsing, heuristic matching."""

import hashlib
import re
from typing import Any, Optional

from agenda import Object


def strip_duplicate_decls(llm_code: str, preamble: str) -> str:
    """Remove declarations from LLM output that are already in the preamble.

    The proof/repair workers prepend a preamble with custom definitions, but
    the LLM often re-emits the same definitions.  This strips the duplicates
    to avoid Lean 'already been declared' errors.
    """
    if not preamble:
        return llm_code

    # Collect names declared in the preamble
    decl_re = re.compile(
        r'^(?:noncomputable\s+|partial\s+|private\s+|protected\s+|unsafe\s+)*'
        r'(?:def|structure|class|instance|inductive|abbrev|theorem|lemma)'
        r'\s+([a-zA-Z_][a-zA-Z0-9_\'\.]*)',
        re.MULTILINE,
    )
    preamble_names = {m.group(1) for m in decl_re.finditer(preamble)}
    if not preamble_names:
        return llm_code

    # Walk LLM output and drop blocks that re-declare a preamble name.
    # A block = the declaration line + all following lines until the next
    # top-level declaration or blank-line-then-declaration.
    lines = llm_code.split('\n')
    out: list[str] = []
    skip = False
    for line in lines:
        m = decl_re.match(line)
        if m:
            if m.group(1) in preamble_names:
                skip = True
                continue
            else:
                skip = False
        elif skip:
            # Keep skipping until next top-level construct or non-indented non-blank line
            stripped = line.strip()
            if stripped == '':
                # blank line might end the block — peek logic is hard,
                # so just skip blank lines within a skipped block
                continue
            if not line[0].isspace() and stripped and not stripped.startswith('--'):
                # non-indented, non-blank, non-comment → new block
                skip = False
            else:
                continue
        out.append(line)
    return '\n'.join(out)


async def gather_definitions(agenda, domain: str, statement: str, related_names: list[str]) -> list[str]:
    """Gather custom definition preambles needed for a Lean statement.

    Collects definitions from ``related_names`` and by scanning the statement
    for identifiers that match known concept definitions in the agenda.
    """
    seen: set[str] = set()
    defs: list[str] = []
    for name in related_names[:10]:
        obj = await agenda.get_object(f"concept/{domain}/{name}")
        if obj and obj.properties.get('kind') in ('definition', 'operation'):
            stmt = obj.properties.get('lean_statement', '')
            if stmt and name not in seen:
                seen.add(name)
                defs.append(stmt)

    tokens = set(re.findall(r'\b([a-z][a-z0-9_]*)\b', statement))
    for token in tokens:
        if token in seen:
            continue
        obj = await agenda.get_object(f"concept/{domain}/{token}")
        if obj and obj.properties.get('kind') in ('definition', 'operation'):
            stmt = obj.properties.get('lean_statement', '')
            if stmt:
                seen.add(token)
                defs.append(stmt)

    return defs


def resolve_imports(imports: list[str]) -> list[str]:
    """Collapse granular Mathlib imports into a single `import Mathlib`.

    Individual Mathlib sub-module paths are frequently hallucinated by the LLM.
    Replacing them with the top-level import avoids typecheck failures while
    keeping any non-Mathlib imports intact.
    """
    non_mathlib = [imp for imp in imports if not imp.startswith('Mathlib')]
    has_mathlib = any(imp.startswith('Mathlib') for imp in imports)
    return (['Mathlib'] if has_mathlib else []) + non_mathlib


def format_concepts_for_prompt(concepts: list[Object]) -> str:
    """Format concept Objects into a readable string for LLM prompts."""
    parts = []
    for c in concepts:
        p = c.properties
        lines = [f"### {p.get('name', c.path)}"]
        if p.get('kind'):
            lines.append(f"**Kind:** {p['kind']}")
        if p.get('domain'):
            lines.append(f"**Domain:** {p['domain']}")
        if p.get('description'):
            lines.append(f"**Description:** {p['description']}")
        if p.get('lean_statement'):
            lines.append(f"**Lean statement:**\n```lean\n{p['lean_statement']}\n```")
        if p.get('lean_proof'):
            lines.append(f"**Lean proof:**\n```lean\n{p['lean_proof']}\n```")
        if p.get('tags'):
            lines.append(f"**Tags:** {', '.join(p['tags'])}")
        if p.get('related_concepts'):
            lines.append(f"**Related:** {', '.join(p['related_concepts'])}")
        parts.append('\n'.join(lines))
    return '\n\n'.join(parts)


def parse_conjecture_output(text: str) -> list[dict[str, Any]]:
    """Parse structured LLM output into concept/conjecture definitions.

    Expected format: blocks separated by '---' with fields like
    NAME: ..., STATEMENT: ..., IMPORTS: ..., DESCRIPTION: ..., RELATED: ...
    """
    blocks = re.split(r'\n\s*---\s*\n', text.strip())
    results = []
    for block in blocks:
        if not block.strip():
            continue
        entry = _parse_fields(block)
        if not entry.get('name'):
            continue
        # Sanitize name to valid identifier
        entry['name'] = _sanitize_name(entry['name'])
        # Clean markdown fences from lean code
        if 'lean_statement' in entry:
            entry['lean_statement'] = _strip_code_fences(entry['lean_statement'])
        results.append(entry)
    return results


def parse_reflection_output(text: str) -> dict[str, Any]:
    """Parse reflection LLM output into analysis, concepts, and optional new heuristic."""
    result: dict[str, Any] = {}

    # Normalize: ensure a trailing newline so all the ``...\n$`` lookaheads in
    # the regexes below match the final field. Bedrock / langchain responses
    # often end without one, which previously caused the last NEW_HEURISTIC_*
    # field (and thus the whole proposal) to silently drop.
    if not text.endswith("\n"):
        text = text + "\n"

    # Extract analysis section
    analysis_match = re.search(r'ANALYSIS[:\s]*\n(.*?)(?=\n(?:NAME|NEW_HEURISTIC|$))',
                               text, re.DOTALL)
    if analysis_match:
        result['analysis'] = analysis_match.group(1).strip()

    # Extract new concepts (reuse conjecture parser for concept blocks)
    result['concepts'] = parse_conjecture_output(text)

    # Extract new heuristic proposal. INPUT_KINDS / INPUT_TAGS were added in
    # Phase 2 so the soundness check's applies_to filter has something to bind
    # to — without them every reflection-born heuristic would match no concepts
    # and be rejected as applies_to_empty.
    heuristic: dict[str, Any] = {}
    fields = (
        'NEW_HEURISTIC_NAME',
        'NEW_HEURISTIC_KIND',
        'NEW_HEURISTIC_INPUT_KINDS',
        'NEW_HEURISTIC_INPUT_TAGS',
        'NEW_HEURISTIC_TEMPLATE',
    )
    for field in fields:
        # ``[: \t]*`` instead of ``[:\s]*`` so we don't accidentally consume the
        # terminating newline when the value is empty (e.g. ``INPUT_TAGS:\n``).
        match = re.search(rf'{field}[: \t]*\n?(.*?)(?=\n(?:NEW_HEURISTIC_|$))',
                          text, re.DOTALL)
        if match:
            val = match.group(1).strip()
            # Guard against the lookahead falling through into the next field's
            # body when this field's value is empty.
            if val and val.upper() != 'NONE' and not val.startswith('NEW_HEURISTIC_'):
                key = field.replace('NEW_HEURISTIC_', '').lower()
                heuristic[key] = val

    # Parse the comma-separated lists into the property names the worker uses
    # (worker reads h_spec['input_concept_kinds'] and h_spec['input_tags']).
    if 'input_kinds' in heuristic:
        raw = heuristic.pop('input_kinds')
        items = [s.strip() for s in raw.split(',') if s.strip()]
        # 'all' is shorthand for "no kind filter" — leave concept_kinds empty.
        if items and items != ['all']:
            heuristic['input_concept_kinds'] = items
    if 'input_tags' in heuristic:
        raw = heuristic.pop('input_tags')
        items = [s.strip() for s in raw.split(',') if s.strip()]
        if items:
            heuristic['input_tags'] = items
        else:
            heuristic.pop('input_tags', None)

    if heuristic.get('name') and heuristic.get('template'):
        result['new_heuristic'] = heuristic

    return result


def heuristic_matches_concept(heuristic: Object, concept: Object) -> bool:
    """Check if a heuristic is applicable to a concept based on kind and tag filters."""
    h_props = heuristic.properties
    c_props = concept.properties

    # Filter by concept kind
    input_kinds = h_props.get('input_concept_kinds', [])
    if input_kinds:
        concept_kind = c_props.get('kind', '')
        if concept_kind not in input_kinds:
            return False

    # Filter by tags
    input_tags = h_props.get('input_tags', [])
    if input_tags:
        concept_tags = set(c_props.get('tags', []))
        if not concept_tags.intersection(input_tags):
            return False

    return True


def _parse_fields(block: str) -> dict[str, Any]:
    """Parse a block of KEY: value lines into a dict."""
    field_map = {
        'NAME': 'name',
        'STATEMENT': 'lean_statement',
        'IMPORTS': 'lean_imports',
        'DESCRIPTION': 'description',
        'RELATED': 'related_concepts',
        'KIND': 'kind',
        'TAGS': 'tags',
    }
    result: dict[str, Any] = {}
    current_key = None
    current_lines: list[str] = []

    for line in block.split('\n'):
        matched = False
        for prefix, key in field_map.items():
            if line.strip().startswith(prefix + ':') or line.strip().startswith(prefix + ' :'):
                if current_key is not None:
                    result[current_key] = _finalize_field(current_key, current_lines)
                current_key = key
                # Get value after the colon
                after = line.split(':', 1)[1].strip() if ':' in line else ''
                current_lines = [after] if after else []
                matched = True
                break
        if not matched and current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        result[current_key] = _finalize_field(current_key, current_lines)

    return result


def _finalize_field(key: str, lines: list[str]) -> Any:
    """Convert accumulated lines into the appropriate type for a field."""
    text = '\n'.join(lines).strip()
    if key in ('lean_imports', 'related_concepts', 'tags'):
        # Parse comma-separated list
        items = [item.strip() for item in text.replace('\n', ',').split(',')]
        items = [item for item in items if item]
        # Sanitize related concept names to match stored object paths
        if key == 'related_concepts':
            items = [_sanitize_name(item) for item in items]
        return items
    return text


def _sanitize_name(name: str) -> str:
    """Convert a name to a valid Lean-style snake_case identifier."""
    name = name.strip().strip('`')
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_').lower()
    return name or 'unnamed'


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences, inline backticks, and natural language preambles."""
    text = re.sub(r'^```\w*\n?', '', text.strip())
    text = re.sub(r'\n?```$', '', text.strip())
    text = text.strip()
    # Strip inline backticks wrapping the whole statement
    if text.startswith('`') and text.endswith('`'):
        text = text[1:-1].strip()
    # Strip natural language preamble before a Lean declaration (possibly backtick-wrapped)
    match = re.search(r'`?\s*((?:/--)|(?:(?:theorem|lemma|def|structure|class|instance|abbrev)\b))', text)
    if match and match.start() > 0:
        text = text[match.start():].lstrip('`').strip()
    # Strip any remaining trailing backtick
    if text.endswith('`'):
        text = text[:-1].strip()
    return text.strip()


def normalize_lean_signature(stmt: str) -> str:
    """Normalize a Lean theorem to just its type signature for deduplication.

    Strips doc comments, declaration keyword + name, and proof body, e.g.:
        theorem foo (n : Nat) : n + 0 = n := by simp
    becomes:
        (n : Nat) : n + 0 = n
    """
    s = re.sub(r'/--.*?-/', '', stmt, flags=re.DOTALL).strip()
    s = re.sub(r'^(theorem|lemma)\s+\S+\s*', '', s, count=1).strip()
    s = re.sub(r':=.*', '', s, flags=re.DOTALL).strip()
    s = re.sub(r'\s+', ' ', s).strip()
    return s


async def is_duplicate_statement(agenda, domain: str, lean_statement: str) -> bool:
    """Check whether a theorem with the same type signature already exists.

    Only applies to theorem/lemma declarations; returns False for defs.
    Registers new signatures as _sig/{domain}/{hash} objects.
    """
    stripped = re.sub(r'/--.*?-/', '', lean_statement, flags=re.DOTALL).strip()
    if not (stripped.startswith('theorem ') or stripped.startswith('lemma ')):
        return False
    sig = normalize_lean_signature(lean_statement)
    if not sig:
        return False
    sig_hash = hashlib.md5(sig.encode()).hexdigest()[:16]
    sig_path = f"_sig/{domain}/{sig_hash}"
    if await agenda.get_object(sig_path) is not None:
        return True
    await agenda.create_object(Object(
        path=sig_path, type="signature", content=sig.encode('utf-8'),
    ))
    return False
