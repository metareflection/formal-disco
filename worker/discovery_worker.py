"""Worker that applies heuristics to concepts to generate new conjectures and definitions."""

import logging
import os
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agenda import Agenda, Object, Task, WorkStatus
from discovery import format_concepts_for_prompt, gather_definitions, parse_conjecture_output, heuristic_matches_concept, is_duplicate_statement, resolve_imports
from discovery.checks.counterexample import check_counterexample
from discovery.checks.novelty import (
    NoveltyIndex,
    check_novelty,
    collect_agenda_corpus,
    collect_leandisco_corpus,
)
from discovery.prompts import system_conjecture, format_conjecture_user
from discovery.trace import Tracer
from discovery.worth import update_heuristic_worth
from language import Language, Program, VerificationOutcome

from . import Worker

logger = logging.getLogger(__name__)


class DiscoveryWorker(Worker):
    """Applies heuristics to concepts to discover new definitions and conjectures.

    For each 'discover' task:
      1. Load the target concept and related neighbors
      2. Select applicable heuristics, sorted by worth
      3. For each heuristic, invoke the LLM to generate new concepts
      4. Typecheck new concepts, create Objects, and enqueue follow-up tasks
    """

    def __init__(
        self,
        llm: Any,
        language: str = 'lean',
        domain: str = 'group_theory',
        max_heuristics_per_concept: int = 4,
        max_conjectures_per_heuristic: int = 5,
        attempt_priority_factor: float = 0.8,
        enable_counterexample_check: bool = True,
        counterexample_num_inst: int = 50,
        counterexample_timeout: float = 60.0,
        enable_novelty_check: bool = True,
        novelty_threshold: float = 0.92,
        novelty_model: str = NoveltyIndex.DEFAULT_MODEL,
        lake_project_dir: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._backend = Language[language.upper()].get_backend()
        self._language = language.lower()
        self._domain = domain
        self._max_heuristics = max_heuristics_per_concept
        self._max_conjectures = max_conjectures_per_heuristic
        self._attempt_priority_factor = attempt_priority_factor
        self._enable_counterexample_check = enable_counterexample_check
        self._counterexample_num_inst = counterexample_num_inst
        self._counterexample_timeout = counterexample_timeout
        self._enable_novelty_check = enable_novelty_check
        self._novelty_threshold = novelty_threshold
        self._novelty_model = novelty_model
        self._lake_project_dir = lake_project_dir
        self._novelty_index: Optional[NoveltyIndex] = None

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            result = await agenda.claim_next_tasks(type="discover")
            if result is None:
                break

            task, status = result[0]
            try:
                await self._process_task(agenda, task, status)
            except Exception as e:
                logger.exception("DiscoveryWorker error on task %s", task.id)
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                         new_notes={"error": str(e)})
            fuel -= 1

    async def _process_task(self, agenda: Agenda, task: Task, status) -> None:
        concept_path = task.properties.get('concept')
        if not concept_path:
            await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                     new_notes={"error": "no concept path"})
            return

        concept_obj = await agenda.get_object(concept_path)
        if concept_obj is None:
            await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                     new_notes={"error": f"concept not found: {concept_path}"})
            return

        # Enumerate ALL heuristic objects in the agenda — including ones born
        # via reflection at runtime — and filter by (a) concept-generation kind
        # (proof/reflection-kind heuristics belong to other workers), and (b)
        # applies_to filter for this concept.
        #
        # Uses agenda._objects directly because the public Agenda protocol only
        # exposes ``get_object(path)``; same Phase 1 expedient as in
        # discovery/checks/{novelty,soundness}.py. Proper query API is Phase 3.
        objects = getattr(agenda, "_objects", None)
        heuristic_objs = []
        if objects:
            for obj in objects.values():
                if obj.type != "heuristic":
                    continue
                kind = obj.properties.get("heuristic_kind", "")
                if kind not in ("concept", "conjecture"):
                    continue
                if heuristic_matches_concept(obj, concept_obj):
                    heuristic_objs.append(obj)

        # Sort by interestingness (worth), take top N
        heuristic_objs.sort(key=lambda h: h.interestingness, reverse=True)
        heuristic_objs = heuristic_objs[:self._max_heuristics]

        if not heuristic_objs:
            await agenda.update_task(task.id, work_status=WorkStatus.DONE,
                                     new_notes={"info": "no applicable heuristics"})
            return

        # Gather related concepts for context
        related_names = concept_obj.properties.get('related_concepts', [])
        context_concepts = [concept_obj]
        for name in related_names[:5]:
            related = await agenda.get_object(f"concept/{self._domain}/{name}")
            if related is not None:
                context_concepts.append(related)

        concepts_text = format_concepts_for_prompt(context_concepts)
        total_created = 0

        for heuristic in heuristic_objs:
            h_name = heuristic.properties.get('name', 'unknown')
            h_template = heuristic.content.decode('utf-8') if heuristic.content else ''

            # Invoke LLM
            messages = [
                SystemMessage(content=system_conjecture(self._domain)),
                HumanMessage(content=format_conjecture_user(h_template, concepts_text, self._domain)),
            ]

            try:
                response = self._llm.invoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.warning("LLM invocation failed for heuristic %s: %s", h_name, e)
                continue

            # Parse output
            entries = parse_conjecture_output(response_text)[:self._max_conjectures]

            # Bump the heuristic's attempts component; worth recomputes automatically.
            await update_heuristic_worth(agenda, h_name, attempts_delta=1)

            tracer = Tracer(agenda=agenda, worker="DiscoveryWorker")
            await tracer.heuristic_apply(
                heuristic=h_name,
                target=concept_obj.properties.get('name', '?'),
                n_candidates=len(entries),
                domain=self._domain,
            )

            for entry in entries:
                created = await self._create_concept(agenda, entry, heuristic, task, tracer)
                if created:
                    total_created += 1

            logger.info("Heuristic %s on %s: %d/%d concepts created",
                        h_name, concept_obj.properties.get('name', '?'),
                        len(entries), total_created)

        await agenda.update_task(task.id, work_status=WorkStatus.DONE,
                                 new_notes={
                                     'heuristics_applied': [h.properties.get('name') for h in heuristic_objs],
                                     'concepts_created': total_created,
                                 })

    async def _create_concept(
        self, agenda: Agenda, entry: dict, heuristic: Object, parent_task: Task,
        tracer: Optional[Tracer] = None,
    ) -> Optional[str]:
        """Create a concept Object and corresponding task. Returns path or None."""
        name = entry.get('name', '')
        h_name = heuristic.properties.get('name', 'unknown')
        if tracer is None:
            tracer = Tracer(agenda=agenda, worker="DiscoveryWorker")
        if not name:
            await tracer.reject(heuristic=h_name, candidate="<unnamed>", reason="missing_name")
            return None

        path = f"concept/{self._domain}/{name}"

        # Deduplicate by name
        existing = await agenda.get_object(path)
        if existing is not None:
            await tracer.reject(heuristic=h_name, candidate=name, reason="duplicate_name")
            return None

        # Deduplicate by normalized type signature
        statement = entry.get('lean_statement', '')
        if statement and await is_duplicate_statement(agenda, self._domain, statement):
            logger.info("Skipping %s: duplicate statement signature", name)
            await tracer.reject(heuristic=h_name, candidate=name, reason="duplicate_signature")
            return None

        kind = entry.get('kind', 'conjecture')

        obj_path = await agenda.create_object(Object(
            path=path,
            type="concept",
            content=entry.get('description', '').encode('utf-8'),
            parents=[parent_task.properties.get('concept', '')],
            properties={
                'name': name,
                'kind': kind,
                'domain': self._domain,
                'description': entry.get('description', ''),
                'lean_statement': entry.get('lean_statement', ''),
                'lean_imports': entry.get('lean_imports', []),
                'tags': entry.get('tags', []),
                'related_concepts': entry.get('related_concepts', []),
                'origin_heuristic': h_name,
                'proof_attempts': 0,
                'proof_strategy': None,
            },
        ))

        # Novelty gate (applies to both definitions and conjectures).
        novelty_verdict, novelty_score, novelty_neighbor = await self._run_novelty_check(
            agenda, entry, name, h_name, tracer,
        )
        if novelty_verdict == "too_similar":
            return obj_path

        # Create follow-up task
        if kind == 'conjecture':
            # Typecheck with sorry stub before queuing a prove task
            if not await self._typecheck_conjecture(agenda, entry):
                logger.info("Conjecture %s failed typecheck, skipping prove task", name)
                await tracer.reject(heuristic=h_name, candidate=name, reason="typecheck_failed", concept_kind=kind)
                return obj_path

            # Counterexample search before committing an Opus call.
            cex_verdict = await self._run_counterexample_check(agenda, entry, name, h_name, tracer)
            if cex_verdict == 'refuted':
                # Concept exists but is marked refuted — no prove task queued.
                return obj_path

            await tracer.admit(
                heuristic=h_name, candidate=name,
                reason="typecheck_passed", concept_kind=kind,
                counterexample_check=cex_verdict,
                novelty_score=novelty_score,
                novelty_neighbor=novelty_neighbor,
            )
            await update_heuristic_worth(
                agenda, h_name,
                admits_delta=1,
                novelty_delta=max(0.0, 1.0 - novelty_score),
            )

            # Add the admitted statement to the novelty index for future checks.
            self._index_admit(name, entry.get('lean_statement', ''))

            # Priority based on heuristic success rate (Laplace smoothing)
            successes = heuristic.properties.get('successes', 0)
            attempts = heuristic.properties.get('attempts', 0)
            priority = (successes + 1) / (attempts + 2)

            task_id = await agenda.add_task(Task(
                id=f"prove-{name}",
                type="prove",
                parents=[parent_task.id],
                properties={'concept': obj_path, 'origin_heuristic': h_name},
                interest_dependencies=[obj_path],
            ))
            # Set priority based on heuristic success rate
            await agenda.update_task(task_id, priority_factor=priority)
        else:
            await tracer.admit(
                heuristic=h_name, candidate=name,
                reason="definition_admitted", concept_kind=kind,
                novelty_score=novelty_score,
                novelty_neighbor=novelty_neighbor,
            )
            await update_heuristic_worth(
                agenda, h_name,
                admits_delta=1,
                novelty_delta=max(0.0, 1.0 - novelty_score),
            )
            self._index_admit(name, entry.get('lean_statement', ''))
            # Definitions get discover tasks
            await agenda.add_task(Task(
                id=f"discover-{name}",
                type="discover",
                parents=[parent_task.id],
                properties={'concept': obj_path},
                interest_dependencies=[obj_path],
            ))

        return obj_path

    async def _ensure_novelty_index(self, agenda: Agenda) -> Optional[NoveltyIndex]:
        if not self._enable_novelty_check:
            return None
        if self._novelty_index is not None:
            return self._novelty_index

        cache_path = None
        ckpt = getattr(agenda, "_checkpoint_path", None)
        if ckpt:
            base = ckpt.rsplit(".", 1)[0]
            cache_path = f"{base}.novelty.npz"

        index = NoveltyIndex(model_name=self._novelty_model, cache_path=cache_path)
        try:
            entries = collect_agenda_corpus(agenda)
            lake = self._lake_project_dir or os.environ.get("LEAN_PROJECT_DIR")
            entries += collect_leandisco_corpus(lake)
            index.build(entries)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to build novelty index: %s", e)
            self._enable_novelty_check = False
            return None

        self._novelty_index = index
        return index

    async def _run_novelty_check(
        self, agenda: Agenda, entry: dict, name: str, h_name: str, tracer: Tracer,
    ) -> tuple[str, float, str]:
        """Returns (verdict, similarity, neighbor)."""
        index = await self._ensure_novelty_index(agenda)
        if index is None:
            return ("skipped", 0.0, "")

        statement = entry.get("lean_statement", "")
        try:
            result = check_novelty(
                statement=statement,
                index=index,
                threshold=self._novelty_threshold,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("novelty check error for %s: %s", name, e)
            return ("error", 0.0, "")

        score = float(result.details.get("similarity", 0.0))
        neighbor = str(result.details.get("nearest", ""))

        if result.verdict == "too_similar":
            await agenda.update_object(
                f"concept/{self._domain}/{name}",
                new_properties={
                    "too_similar": True,
                    "novelty_score": score,
                    "novelty_neighbor": neighbor,
                },
            )
            await tracer.reject(
                heuristic=h_name, candidate=name,
                reason="too_similar",
                witness=neighbor,
                similarity=score,
            )
            logger.info("REJECTED %s as too similar (cosine=%.3f) to %s",
                        name, score, neighbor)
            return ("too_similar", score, neighbor)

        return ("novel", score, neighbor)

    def _index_admit(self, name: str, statement: str) -> None:
        """Add an admitted concept to the in-memory novelty index."""
        if self._novelty_index is None or not statement:
            return
        try:
            self._novelty_index.add(f"concept/{self._domain}/{name}", statement)
        except Exception as e:  # noqa: BLE001
            logger.debug("novelty index add failed: %s", e)

    async def _run_counterexample_check(
        self, agenda: Agenda, entry: dict, name: str, h_name: str, tracer: Tracer,
    ) -> str:
        """Run slim_check on the candidate. Returns verdict tag.

        Side effects: on refutation, marks the concept Object as refuted, emits a
        ``reject`` trace, and records the witness so it can be reviewed later.
        """
        if not self._enable_counterexample_check:
            return "skipped"

        try:
            preamble_defs = await gather_definitions(
                agenda, self._domain,
                entry.get('lean_statement', ''),
                entry.get('related_concepts', []),
            )
            preamble = '\n\n'.join(preamble_defs)

            result = check_counterexample(
                statement=entry.get('lean_statement', ''),
                imports=entry.get('lean_imports', []),
                preamble=preamble,
                backend=self._backend,
                num_inst=self._counterexample_num_inst,
                timeout=self._counterexample_timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("counterexample_check failed for %s: %s", name, e)
            return "error"

        if result.verdict == "refuted":
            await agenda.update_object(
                f"concept/{self._domain}/{name}",
                new_properties={
                    'refuted': True,
                    'counterexample_witness': result.witness or '',
                    'counterexample_method': 'slim_check',
                },
            )
            await tracer.reject(
                heuristic=h_name, candidate=name,
                reason="counterexample_found",
                witness=(result.witness or '')[:600],
            )
            logger.info("REFUTED %s by counterexample: %s",
                        name, (result.witness or '')[:200])
            return "refuted"

        return result.verdict  # 'passed' or 'inconclusive'

    async def _typecheck_conjecture(self, agenda: Agenda, entry: dict) -> bool:
        """Verify that a conjecture typechecks with sorry before queuing it for proof.

        Builds a stub file including imports, definitions of referenced concepts,
        and the conjecture statement with := sorry.
        """
        statement = entry.get('lean_statement', '')
        if not statement:
            return False

        imports = entry.get('lean_imports', [])
        imports_str = '\n'.join(f'import {imp}' for imp in resolve_imports(imports))

        preamble_defs = await gather_definitions(
            agenda, self._domain, statement,
            entry.get('related_concepts', []),
        )
        preamble = '\n\n'.join(preamble_defs)

        # Strip any existing proof body and replace with sorry
        # The LLM sometimes generates full proofs instead of just signatures
        import re
        clean_stmt = re.sub(r':=\s*by\b.*', ':= sorry', statement, flags=re.DOTALL)
        clean_stmt = re.sub(r':=\s*(?!sorry).*', ':= sorry', clean_stmt, flags=re.DOTALL)
        if ':= sorry' not in clean_stmt:
            clean_stmt = clean_stmt.rstrip() + ' := sorry'

        stub = f"{imports_str}\n\n{preamble}\n\n{clean_stmt}"

        try:
            prog = Program(stub, Language[self._language.upper()], name='typecheck')
            result = self._backend.verify(prog)
            if result.outcome == VerificationOutcome.FAIL:
                logger.debug("Typecheck stderr: %s", result.stderr[:200])
            return result.outcome != VerificationOutcome.FAIL
        except Exception as e:
            logger.warning("Typecheck error for conjecture: %s", e)
            return False
