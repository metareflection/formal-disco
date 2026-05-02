"""Initial heuristic templates for the Eurisko-like discovery system.

Each heuristic is a prompt template that guides the LLM in generating
new concepts or conjectures. Ported from eurisclo's 29 heuristics,
focusing on the 9 most relevant for Lean 4 mathematical discovery.
"""

INITIAL_HEURISTICS: list[dict] = [
    # --- Concept generation ---
    {
        "name": "specialize",
        "heuristic_kind": "concept",
        "input_concept_kinds": ["definition", "operation", "structure"],
        "input_tags": [],
        "eurisclo_origin": "H1/H6",
        "template": (
            "SPECIALIZE the given concept(s) by adding constraints or restricting the domain.\n"
            "Examples of specialization:\n"
            "  - Group -> Abelian group (add commutativity)\n"
            "  - Homomorphism -> Isomorphism (add bijectivity)\n"
            "  - Monoid -> Group (add inverses)\n"
            "  - Ring -> Field (add multiplicative inverses)\n\n"
            "For each specialization, produce:\n"
            "  1. A new Lean 4 definition or structure with the added constraint\n"
            "  2. One or more conjectures about properties specific to the specialization\n"
        ),
    },
    {
        "name": "generalize",
        "heuristic_kind": "concept",
        "input_concept_kinds": ["definition", "theorem", "structure"],
        "input_tags": [],
        "eurisclo_origin": "H16",
        "template": (
            "GENERALIZE the given concept(s) by relaxing constraints or broadening the domain.\n"
            "Examples of generalization:\n"
            "  - Group -> Monoid (drop inverses)\n"
            "  - Abelian group -> Group (drop commutativity)\n"
            "  - Field -> Ring (drop multiplicative inverses)\n"
            "  - Finite group theorem -> version for arbitrary groups\n\n"
            "For each generalization, produce:\n"
            "  1. A new Lean 4 definition with relaxed constraints\n"
            "  2. Conjectures about which properties of the original still hold\n"
        ),
    },
    {
        "name": "compose_operations",
        "heuristic_kind": "concept",
        "input_concept_kinds": ["operation", "definition"],
        "input_tags": ["operation", "function", "morphism"],
        "eurisclo_origin": "H20",
        "template": (
            "COMPOSE or COMBINE the given operations/concepts to discover new ones.\n"
            "Consider:\n"
            "  - Composition: f(g(x)), g(f(x))\n"
            "  - Commutativity: does f(g(x)) = g(f(x))?\n"
            "  - Fixed points: when does f(x) = x?\n"
            "  - Iteration: f(f(x)), f^n(x)\n"
            "  - Interaction: how do these operations relate to each other?\n\n"
            "For each composition, produce:\n"
            "  1. A new Lean 4 definition for the composed operation\n"
            "  2. Conjectures about the properties of the composition\n"
        ),
    },
    # --- Conjecture formation ---
    {
        "name": "algebraic_identities",
        "heuristic_kind": "conjecture",
        "input_concept_kinds": ["operation", "definition", "structure"],
        "input_tags": [],
        "eurisclo_origin": "H11",
        "template": (
            "Conjecture ALGEBRAIC IDENTITIES involving the given concept(s).\n"
            "Look for laws such as:\n"
            "  - Involution: f(f(x)) = x\n"
            "  - Idempotence: f(f(x)) = f(x)\n"
            "  - Commutativity: f(x, y) = f(y, x)\n"
            "  - Associativity: f(f(x, y), z) = f(x, f(y, z))\n"
            "  - Absorption: f(x, g(x, y)) = x\n"
            "  - Distributivity: f(x, g(y, z)) = g(f(x, y), f(x, z))\n"
            "  - Cancellation: f(x, y) = f(x, z) -> y = z\n\n"
            "Express each identity as a Lean 4 theorem statement (with sorry).\n"
        ),
    },
    {
        "name": "analogy_transfer",
        "heuristic_kind": "conjecture",
        "input_concept_kinds": ["theorem"],
        "input_tags": [],
        "eurisclo_origin": "H21",
        "template": (
            "TRANSFER the given theorem(s) to analogous structures by analogy.\n"
            "Analogies to consider:\n"
            "  - Group <-> Ring <-> Module\n"
            "  - Nat <-> Int <-> Rat <-> Real\n"
            "  - List <-> Multiset <-> Finset\n"
            "  - Monoid <-> Group (what happens without inverses?)\n"
            "  - Additive <-> Multiplicative notation\n\n"
            "For each analogy, produce a Lean 4 theorem statement expressing\n"
            "the transferred result in the new setting.\n"
        ),
    },
    {
        "name": "boundary_cases",
        "heuristic_kind": "conjecture",
        "input_concept_kinds": ["operation", "definition", "theorem"],
        "input_tags": [],
        "eurisclo_origin": "H7/H10",
        "template": (
            "Explore BOUNDARY CASES and edge cases of the given concept(s).\n"
            "Consider:\n"
            "  - Identity element: what happens at e / 1 / 0?\n"
            "  - Inverse: what happens at x⁻¹?\n"
            "  - Equal arguments: f(x, x)\n"
            "  - Trivial structures: the trivial group, empty set, zero ring\n"
            "  - Extreme values: n = 0, n = 1, singleton sets\n"
            "  - Self-application: applying an operation to itself\n\n"
            "Express each boundary case as a Lean 4 theorem statement.\n"
        ),
    },
    # --- Proof strategies ---
    {
        "name": "direct_tactic_proof",
        "heuristic_kind": "proof",
        "input_concept_kinds": ["conjecture"],
        "input_tags": [],
        "eurisclo_origin": None,
        "template": (
            "Try to prove the theorem using direct automation tactics:\n"
            "  simp, ring, ring_nf, norm_num, omega, decide, group_cancel,\n"
            "  field_simp, positivity, norm_cast, push_neg, aesop, tauto\n\n"
            "Prefer short, direct proofs. Combine with `simp [...]` providing\n"
            "relevant lemma names if plain `simp` fails.\n"
        ),
    },
    {
        "name": "structured_proof",
        "heuristic_kind": "proof",
        "input_concept_kinds": ["conjecture"],
        "input_tags": [],
        "eurisclo_origin": None,
        "template": (
            "Try to prove the theorem using structured proof techniques:\n"
            "  - `induction` on a natural number or structural argument\n"
            "  - `cases` / `rcases` / `obtain` for case analysis\n"
            "  - `calc` blocks for equational reasoning chains\n"
            "  - `have` / `suffices` for intermediate goals\n"
            "  - `rw [...]` sequences for rewriting\n"
            "  - `constructor` and `ext` for building structures\n\n"
            "Build the proof step by step. Use `have` to name intermediate results.\n"
        ),
    },
    # --- Reflection ---
    {
        "name": "extract_and_refine",
        "heuristic_kind": "reflection",
        "input_concept_kinds": ["theorem"],
        "input_tags": [],
        "eurisclo_origin": "H2/H12",
        "template": (
            "Analyze the proved theorem(s) and extract new mathematical insights:\n"
            "  1. Identify intermediate lemmas used in the proof that are interesting on their own\n"
            "  2. Extract new definitions or concepts that emerged during the proof\n"
            "  3. Suggest stronger versions of the theorem\n"
            "  4. Identify patterns that could apply to other structures\n\n"
            "For failed conjectures, propose WEAKENED versions that might be provable:\n"
            "  - Add additional hypotheses\n"
            "  - Restrict to a subclass\n"
            "  - Weaken the conclusion\n"
        ),
    },
]
