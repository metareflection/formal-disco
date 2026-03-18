# Formal Disco

A distributed, scalable system for synthetically generating diverse and complex formal programs using LLMs.

The core idea: an **agenda** holds tasks and objects. **Workers** of different types claim tasks, call LLMs to produce programs, verify them, and enqueue follow-up tasks. A **scheduler** orchestrates workers in a loop. The system scales from a single laptop to a cluster of nodes.

## Architecture

### Agenda (`agenda.py`, `agenda_distributed.py`)

The agenda is a task queue combined with an object store. Tasks have types (`implement`, `repair`, `extend`), priorities, and a status. Objects store artifacts (ideas, programs, distillation examples) with content and metadata (e.g., a program's verifier output).

There are two implementations of the agenda:

- **Local agenda** (`LocalAgenda`): in-process, pickle-checkpointed. Used for testing in a single machine.
- **Distributed agenda** (`AgendaClient` + `agenda_server.py`): Wraps a LocalAgenda with an RPC interface, so it can be accessed from many machines at once. Since this is just a wrapped LocalAgenda, it essentially has the same features, but is what we use in the cluster.

The agenda periodically checkpoints to disk in a unified pickle file containing both its tasks and objects, and it logs **codebase metrics** (e.g., number of verified programs, lines of code, declaration counts), **diversity/complexity metrics** (entropy of feature distributions, median/p90 of complexity metrics), and **worker success rates** to a logger (see `logger/` for the generic interface). The main logger is the WandB one (`logger/wandb.py`), which we use by default. If you don't have one, you should likely make a wandb.ai account.

### Workers (`worker/`)

Workers implement the main actors of the distributed system, that perform tasks and communicate via the agenda. Each worker claims tasks of certain types from the agenda, calls an LLM, produces objects and updates the tasks. Workers are language agnostic, and they accept a `language` config parameter.

| Worker | File | Task types (takes → creates)| Description |
|---|---|---|---|
| `ReadmeInspiredIdeaGenerator` | `worker/readme_idea_generator.py` | → `implement` | Samples a GitHub README, prompts the LLM for a program idea, enqueues a task with type `implement` |
| `LLMImplementer` | `worker/llm_implementer.py` | `implement` → `extend`|`repair` | Implements an idea as a program, calls verifier, creates an `extend` task on success or a `repair` on failure. |
| `LLMFixer` | `worker/llm_fixer.py` | `repair` → `extend`|`repair` | Applies a diff to fix a failing program, calls verifier again |
| `EditorWorker` | `worker/editor_worker.py` | `extend` → `extend`|`repair` | Extends a working program with new content via a diff |

Workers can optionally collect **distillation examples** (every prompt/response pair with its outcome) by setting `distill: success-only` or `distill: all` in config. We use this both to collect initial fine-tuning data from a strong model (e.g., Claude), and to self-improve the workers by fine-tuning on their own successes as the system runs. 

### Language Backends (`language/`)

Formal Disco is language-agnostic, though right now we only have one language backend. Each backend implements:

- **Verification** by invoking an external command (e.g., `dafny verify prog.dfy`)
- **Prompt builder**: LLM prompts for `implement`, `repair`, `extend`, and `idea` tasks
- **Complexity metrics**: scalar measures per program (method body sizes, loops per method, identifiers in assertions/invariants)
- **Diversity/feature metrics**: counters per program (subject words in identifiers, annotation templates, loop skeletons)

`language/dafny` has the Dafny backend. **Verus** is the next planned backend. Adding a new language means mainly subclassing `LanguageBackend` and `PromptBuilder` in a new sub-package under `language/`.

### Scheduler (`scheduler.py`)

A `RoundRobinScheduler` simply cycles through a list of workers, giving each a specified budget ("fuel") per turn. We do all of our configuration using [Hydra](https://hydra.cc/) - see `config/`.

---

## Typical Workflows

### 1. Run the discovery system locally with Claude/OpenAI models

Download the GitHub README dataset and save it to `data/gh-readmes-100k.jsonl`, then:

```bash
python scheduler.py +scheduler=readme_ideas_aws_opus +agenda=local
```

This runs indefinitely, checkpointing to a pickle file like `agenda.pkl` and logging to W&B by default. LLMs are configured via `config/llm/`; we use LangChain so switching providers is very easy:

```bash
python scheduler.py +scheduler=readme_ideas +agenda=local llm=openai   # OpenAI (will need an OPENAI_API_KEY)
python scheduler.py +scheduler=readme_ideas +agenda=local llm=aws       # Bedrock (will need aws-login)
python scheduler.py +scheduler=readme_ideas +agenda=local llm=vllm      # local vLLM server exposes an Open-API compatible API
```

To collect distillation examples, set `distill: success-only` (or `all`) in the scheduler config. This saves every (prompt, response, outcome) triple as a `distill-example` object in the agenda.

At any point you can dump artifacts to disk:

```bash
python materialize.py agenda.pkl discoveries/
```

### 2. SFT on collected distillation examples

Inspect what's in a checkpoint:

```bash
python distill.py stats -d agenda-run4-opus.pkl
```

Fine-tune a model (config in `config/distill.yaml`):

```bash
python distill.py sft
```

The current default model is **Qwen2.5-Coder-14B-Instruct** (`config/distill.yaml`). We have also experimented with Qwen3-30B-A3B, but training is currently slow and we're investigating why - Qwen2.5 is a whole lot faster but it's a weaker/smaller/older model. See `scripts/sft.sbatch` for a simple slurm script to run this as a job on the cluster.

### 3. Run the discovery system with a local model (distributed run)

1. Start a vLLM server (see `scripts/vllm.sbatch` for the SLURM command)
2. Start the agenda server:
   ```bash
   python agenda_server.py --host 127.0.0.1 --port 9999 --checkpoint agenda.pkl
   # or via slurm: scripts/agenda_server.sbatch
   ```
3. Spawn many workers connecting to the agenda and vLLM:
   ```bash
   python scheduler.py +agenda=distributed +scheduler=readme_ideas llm=vllm
   # or via slurm: scripts/worker.sbatch (spawn many of these)
   ```

### 4. Fine-tune on task-specific data and evaluate

Formal Disco also supports **task-directed SFT** on generated programs — training models to implement, repair, or synthesize lemma bodies. See `tasks/` for the task definitions and `eval.py` for evaluation.

For evaluation with local models, serve them with vLLM (same setup as above). Claude and OpenAI baselines work out of the box.

### 5. Self-distillation loop

Train a model on examples generated by Claude/GPT → run the discovery system using that model → collect its successful examples → fine-tune again. Repeat.

---

## To-Dos / Next Milestones

- [ ] **Implement Verus backend**: `language/verus/` subclassing `LanguageBackend`
- [ ] **Diversity-driven example selection**: measure per-program "uniqueness" against the corpus and use the most unique successful programs as an in-context examples for workers (first planned diversity improvement method - the validation for this will be in whether it improves the diversity/complexity metrics)
- [ ] **Debug task-directed SFT**: fixer/lemma-synth fine-tuning is getting lower-than-expected performance; still need to investigate deeper
