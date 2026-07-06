# Eurisko-like Discovery System for Lean 4

## Vision

An automated mathematical discovery system that generates, proves, and refines
Lean 4 definitions, conjectures, and theorems. Inspired by Doug Lenat's Eurisko,
the system treats heuristics as first-class objects with worth that evolves over
time -- allowing it to learn which discovery strategies work and to propose new ones.

The ground truth is always compilable Lean 4 code. Natural language serves as
context for the LLM, but verification determines whether a concept is well-formed
and whether a conjecture is proved.

## Concepts as Lean Declarations

In Eurisko, concepts were "units" with many slots. Here, concepts are Lean
declarations stored as agenda Objects:

| Concept kind | Lean form | Lifecycle |
|---|---|---|
| definition | `def`, `structure`, `class`, `abbrev` | Created by seeder or discovery |
| instance | `instance` | Created by discovery |
| conjecture | `theorem ... := sorry` | Created by discovery, awaits proof |
| theorem | `theorem ... := by ...` | Conjecture with completed proof |

Each concept Object has:

```
path:       concept/{domain}/{name}
type:       "concept"
content:    Lean source code (the declaration itself)
properties:
  name:             str               -- identifier
  kind:             str               -- "definition" | "conjecture" | "theorem" | "instance"
  domain:           str               -- e.g. "group_theory"
  description:      str               -- natural language for LLM context
  lean_statement:   str               -- the declaration signature (no proof body)
  lean_proof:       str | None        -- proof body (None for conjectures)
  lean_imports:     list[str]         -- required imports
  tags:             list[str]         -- for heuristic filtering
  related_concepts: list[str]         -- names of related concepts
  origin_heuristic: str | None        -- which heuristic created this
  proof_attempts:   int               -- times proof was attempted
  proof_strategy:   str | None        -- which heuristic succeeded
```

Interestingness (Object.interestingness) serves as Eurisko's "worth": it
determines priority for downstream tasks and decays or grows based on outcomes.

## Heuristics as Objects

Eurisko's key insight: heuristics are themselves concepts that can be created,
evaluated, and killed. Here, heuristics are agenda Objects with success tracking:

```
path:       heuristic/{name}
type:       "heuristic"
content:    prompt template (the LLM instruction for this heuristic)
properties:
  name:                str
  heuristic_kind:      str        -- "concept" | "conjecture" | "proof" | "reflection"
  input_concept_kinds: list[str]  -- which concept kinds this applies to
  input_tags:          list[str]  -- tag filter (empty = all)
  attempts:            int        -- times applied
  successes:           int        -- times it led to a proved theorem
  eurisclo_origin:     str | None -- reference to original Eurisko heuristic
```

Worth is tracked via interestingness + the success rate (successes/attempts).
Heuristics are sorted by worth when selecting which to apply. The Laplace-smoothed
priority for tasks created by a heuristic is `(successes + 1) / (attempts + 2)`.

### Initial Heuristics (9)

Ported from eurisclo's 29 heuristics, focusing on the ones most relevant to
Lean-based mathematical discovery:

**Concept generation (3):**

1. **specialize** (eurisclo H1/H6) -- Add constraints to a definition. E.g.,
   Group -> Abelian group, Homomorphism -> Isomorphism. Produces new definitions
   and conjectures about the specialized concept.

2. **generalize** (eurisclo H16) -- Relax constraints. E.g., finite groups ->
   monoids, integers -> rings. Asks what properties still hold.

3. **compose_operations** (eurisclo H20) -- Study compositions and interactions
   of operations. E.g., f(g(x)), commutativity of operations, fixed points,
   iteration.

**Conjecture formation (3):**

4. **algebraic_identities** (eurisclo H11) -- Conjecture algebraic laws:
   involution, idempotence, commutativity, absorption, distributivity.

5. **analogy_transfer** (eurisclo H21) -- Transfer theorems across structures:
   Group <-> Ring <-> Monoid, Nat <-> Int <-> Rat. If P holds for structure A,
   conjecture a version for structure B.

6. **boundary_cases** (eurisclo H7/H10) -- Explore edge cases: identity element,
   inverse, equal arguments, trivial structures, empty sets.

**Proof strategies (2):**

7. **direct_tactic_proof** -- Try closing the goal with automation: simp, ring,
   omega, norm_num, decide, group_cancel.

8. **structured_proof** -- Use structural tactics: induction, cases, calc blocks,
   have chains, rw sequences.

**Reflection (1):**

9. **extract_and_refine** (eurisclo H2/H12) -- Analyze proved theorems to extract
   intermediate lemmas and new concepts. Analyze failures to propose weakened
   conjectures. Can propose entirely new heuristics.

### Heuristic lifecycle

- **Birth**: Initial heuristics are seeded. ReflectionWorker can propose new
  heuristics from patterns observed in proofs.
- **Worth tracking**: Each application increments `attempts`. Each theorem that
  traces back to a heuristic increments `successes`. Interestingness is boosted
  on success, decayed on failure.
- **Death**: When `attempts >= 10` and `successes / attempts < 0.05`, the
  heuristic's interestingness is set to 0.01 (effectively killed).

## Workers

### ConceptSeeder (task type: `seed`)

Runs once at startup. Seeds the agenda with:
- 9 heuristic Objects from `discovery/heuristics_seed.py`
- Initial concept Objects from a domain-specific seed file (e.g. group theory)
- A `discover` task for each seeded concept

### DiscoveryWorker (task type: `discover`)

The core exploration loop:

1. Claim a `discover` task, load the target concept
2. Load all heuristic Objects, filter by `input_concept_kinds` and `input_tags`
3. Sort heuristics by interestingness (highest first), take top N
4. For each heuristic:
   a. Gather context: target concept + related neighbors
   b. Invoke LLM with the heuristic's prompt template + concept context
   c. Parse output into concept definitions
   d. For each new concept:
      - Deduplicate by name
      - Typecheck the Lean statement (verify with sorry stub)
      - Create concept Object
      - If conjecture: create `prove` task (priority = Laplace-smoothed heuristic rate)
      - If definition: create `discover` task
   e. Update heuristic: increment `attempts`

### ProofWorker (task type: `prove`)

Attempts to prove conjectures:

1. Claim a `prove` task, load the conjecture concept
2. Load proof-strategy heuristics, sort by interestingness, take top N
3. For each strategy:
   a. Invoke LLM with: theorem statement + imports + related concepts + strategy hint
   b. Verify result with `LeanBackend.verify()`
   c. On SUCCESS:
      - Update concept: kind -> "theorem", store lean_proof
      - Boost interestingness (concept + heuristic)
      - Create dataset Object (`dataset/theorem_{name}.lean`)
      - Create `reflect` task (outcome="success")
      - Create `discover` task (to explore the proved theorem further)
   d. On FAIL:
      - Decay heuristic interestingness
      - If max attempts reached: create `reflect` task (outcome="failure")
      - Else: create `repair` task for LLMFixer

### ReflectionWorker (task type: `reflect`)

Meta-learning -- the key Eurisko mechanism:

**On success:**
1. Invoke LLM with proved theorem and context
2. Extract new concepts/conjectures from the proof
3. Create concept Objects and corresponding tasks
4. Boost the origin heuristic's interestingness
5. Check for NEW_HEURISTIC proposals in the LLM output
   - If found: create a new heuristic Object (heuristic birth)

**On failure:**
1. Invoke LLM to propose weakened versions of the failed conjecture
2. Create new conjectures with "weakened" tag
3. Decay the origin heuristic's interestingness
4. **Kill check**: if heuristic has `attempts >= 10` and
   `successes / attempts < 0.05`, set interestingness to 0.01

## Discovery Loop

```
ConceptSeeder (one-shot)
  │
  ├──→ heuristic Objects (9 initial)
  └──→ concept Objects + discover tasks
          │
          ▼
    DiscoveryWorker ◄────────────────────────────────────┐
      │ applies heuristics to concepts                   │
      │ generates new definitions + conjectures          │
      │                                                  │
      ├──→ definitions ──→ discover tasks ───────────────┤
      └──→ conjectures ──→ prove tasks                   │
                             │                           │
                             ▼                           │
                       ProofWorker                       │
                         │                               │
                    ┌────┴────┐                          │
                    ▼         ▼                          │
                SUCCESS    FAILURE                       │
                    │         │                          │
                    │         ├──→ repair tasks (LLMFixer)
                    │         └──→ reflect tasks         │
                    │                  │                 │
                    ├──→ reflect tasks │                 │
                    ├──→ discover tasks ─────────────────┤
                    │                  │                 │
                    ▼                  ▼                 │
              ReflectionWorker                           │
                │                                        │
                ├──→ new concepts ──→ discover tasks ────┘
                ├──→ weakened conjectures ──→ prove tasks
                ├──→ heuristic worth updates
                └──→ new heuristics (birth)
```

## Mapping to Eurisclo

| Eurisclo concept | Our system |
|---|---|
| Unit with slots | agenda Object with properties dict |
| Unit worth (static int) | Object.interestingness (dynamic float) |
| Heuristic if-potentially-relevant | `input_concept_kinds` + `input_tags` filter |
| Heuristic if-truly-relevant | Sort by interestingness, take top N |
| Heuristic then-compute | LLM invocation with heuristic prompt template |
| Heuristic then-define-new-concepts | Parse LLM output -> create concept Objects |
| Heuristic then-add-to-agenda | Create discover/prove/reflect tasks |
| Rarity (freq, true, false) | (successes, attempts) with Laplace smoothing |
| h2 kill mechanism | ReflectionWorker kills heuristics below 5% success |
| h6 specialization | specialize heuristic |
| h16 generalization | generalize heuristic |
| h11 algebraic identities | algebraic_identities heuristic |
| h21 analogy/extension | analogy_transfer heuristic |
| h7/h10 boundary cases | boundary_cases heuristic |
| creditors chain | origin_heuristic property on concepts |

Key differences from eurisclo:
- LLM replaces hand-coded Lisp lambdas for concept manipulation
- Lean verification replaces eurisclo's run-alg/run-defn for testing
- Heuristics are prompt templates rather than code, making birth easier
- Worth is continuous (float) rather than discrete (int)
- No explicit slot-level specialization -- the LLM operates on whole concepts

## Prompts

Four system prompts drive the LLM interactions:

1. **Conjecture prompt** (DiscoveryWorker): Apply a specific heuristic to
   concepts, output 1-5 new Lean 4 definitions or conjectures in a structured
   format (NAME / STATEMENT / IMPORTS / DESCRIPTION / RELATED).

2. **Proof prompt** (ProofWorker): Given a theorem statement, imports, and
   strategy hint, produce a complete Lean 4 file with a full proof (no sorry).

3. **Reflection prompt** (ReflectionWorker): Analyze a proved or failed theorem,
   extract new concepts, propose weakened conjectures, optionally propose new
   heuristics.

4. **Repair prompt** (LLMFixer, existing): Fix a failing Lean program given
   error output.

## Library of Discovered Results

The sibling project `../LeanDisco` is a Lake project with Mathlib that serves
as the library of everything the system discovers. When ProofWorker proves a
theorem, it is written as a `.lean` file into the LeanDisco project so that
future proofs can build on it.

```
LeanDisco/
  LeanDisco/
    Domains/
      GroupRing/
        MulComm.lean          -- discovered theorem
        InvInv.lean           -- discovered theorem
        ...
      GroupRing.lean           -- imports all GroupRing results
      NumberTheory.lean
      ...
  LeanDisco.lean               -- root imports
  lakefile.toml                -- depends on Mathlib
```

**Workflow:**
1. ProofWorker proves a theorem -> writes `LeanDisco/LeanDisco/Domains/{Domain}/{Name}.lean`
2. Updates the domain import file (e.g. `GroupRing.lean`) to include the new module
3. Runs `lake build` to verify the file integrates cleanly
4. Future proof attempts use `import LeanDisco.Domains.{Domain}` to access all
   previously proved results
5. DiscoveryWorker/ProofWorker prompts include a summary of available library
   theorems as context

**Rules:**
- Only `SUCCESS` results (fully proved, no sorry) are added to the library
- Each file is self-contained: it imports what it needs and exports one theorem
- The library is cumulative -- nothing is removed once added

## Configuration

Scheduler config wires all workers together:

```yaml
# config/scheduler/discovery_lean4.yaml
workers:
  - ConceptSeeder      # runs once
  - DiscoveryWorker    # fast model (Sonnet), generates conjectures
  - ProofWorker        # strong model (Opus), proves theorems
  - LLMFixer           # strong model, repairs failed proofs
  - ProofRepairWorker  # strong model, retries failed proof tasks
  - ReflectionWorker   # fast model, meta-learning
```

Workers use `language: lean` to get the Lean backend for verification.

## Running

Prerequisites:
- `LEAN_PROJECT_DIR` pointing to a built Lake project with Mathlib (e.g. `../LeanDisco`)
- AWS credentials configured for Bedrock access

```bash
# First run (or fresh start -- delete checkpoint to reset)
rm -f agenda-discovery.pkl

# Run the discovery system
LEAN_PROJECT_DIR=../LeanDisco python scheduler.py +scheduler=discovery_lean4 +agenda=local language=lean agenda.checkpoint_path=agenda-discovery.pkl agenda.checkpoint_interval=10 agenda/logger=noop
```

Key options:
- `agenda.checkpoint_interval=10` -- checkpoint every 10 ticks (good for expensive runs)
- `agenda/logger=noop` -- skip WandB logging (use `agenda/logger=wandb` for logging)
- `agenda.max_attempts=100` -- stop after N task attempts (omit to run indefinitely)

The system resumes from `agenda-discovery.pkl` on restart.

For Vertex AI (Google Cloud), use the `discovery_lean4_vertex` config:
```bash
GCP_PROJECT_ID=formal-disco LEAN_PROJECT_DIR=../LeanDisco python scheduler.py +scheduler=discovery_lean4_vertex +agenda=local language=lean agenda.checkpoint_path=agenda-discovery.pkl agenda.checkpoint_interval=10 agenda/logger=noop
```

## Web Dashboard

A React/TypeScript frontend with a FastAPI backend for monitoring the
discovery system in real time.

Important: run from the project root so Python can import `agenda` (needed
to unpickle the checkpoint).

```bash
# Build the frontend (once)
cd web-eurisko && npm install && npm run build && cd ..

# Start the server (from project root)
CHECKPOINT_PATH=agenda-discovery.pkl uvicorn web-eurisko.api:app --port 8000
# Open http://localhost:8000

# Development mode (two terminals, both from project root)
CHECKPOINT_PATH=agenda-discovery.pkl uvicorn web-eurisko.api:app --port 8000 --reload
cd web-eurisko && npm run dev   # proxies /api to backend
# Open http://localhost:5173
```

The dashboard auto-refreshes every 5 seconds and has four tabs:
- **Overview** -- stat cards, task queue, heuristic success rate bars
- **Heuristics** -- detailed cards with templates, attempts/successes, interestingness
- **Theorems** -- gallery of proved theorems with full Lean proofs
- **Concepts** -- filterable/searchable table of all concepts with expandable details

## File Structure

```
discovery/
  __init__.py           # format_concepts_for_prompt, parse outputs, heuristic matching
  heuristics_seed.py    # 9 initial heuristic templates
  prompts.py            # system/user prompt construction

worker/
  concept_seeder.py     # ConceptSeeder
  discovery_worker.py   # DiscoveryWorker
  proof_worker.py       # ProofWorker
  reflection_worker.py  # ReflectionWorker

data/
  seed_group_theory.json  # 15 initial group theory concepts

config/scheduler/
  discovery_lean4.yaml         # AWS Bedrock pipeline config
  discovery_lean4_vertex.yaml  # Vertex AI pipeline config

web-eurisko/
  api.py              # FastAPI backend (reads checkpoint, serves JSON)
  src/
    App.tsx           # Main app with tab navigation
    components/
      OverviewPanel.tsx      # Stats, task queue, heuristic bars
      HeuristicDashboard.tsx # Heuristic detail cards
      TheoremGallery.tsx     # Proved theorems with proofs
      ConceptExplorer.tsx    # Filterable concept table
```
