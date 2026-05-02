"""Worker that performs meta-learning: updates heuristic worth, extracts concepts, proposes new heuristics."""

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agenda import Agenda, Object, Task, WorkStatus
from discovery import parse_conjecture_output, parse_reflection_output, is_duplicate_statement
from discovery.prompts import system_reflect, format_reflect_user
from discovery.trace import Tracer

from . import Worker

logger = logging.getLogger(__name__)


class ReflectionWorker(Worker):
    """Meta-learning worker inspired by Eurisko's self-reflection.

    For each 'reflect' task:
      1. Update the origin heuristic's worth based on success/failure
      2. On success: invoke LLM to extract new concepts from the proof
      3. On failure: invoke LLM to propose weakened conjectures
      4. Optionally: create new heuristic Objects (heuristic birth)
      5. Kill underperforming heuristics (heuristic death)
    """

    def __init__(
        self,
        llm: Any,
        domain: str = 'group_theory',
        interest_success_boost: float = 1.3,
        interest_failure_decay: float = 0.8,
        min_attempts_for_kill: int = 10,
        kill_threshold: float = 0.05,
    ) -> None:
        self._llm = llm
        self._domain = domain
        self._interest_success_boost = interest_success_boost
        self._interest_failure_decay = interest_failure_decay
        self._min_attempts_for_kill = min_attempts_for_kill
        self._kill_threshold = kill_threshold

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            result = await agenda.claim_next_tasks(type="reflect")
            if result is None:
                break

            task, status = result[0]
            try:
                await self._process_task(agenda, task, status)
            except Exception as e:
                logger.exception("ReflectionWorker error on task %s", task.id)
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                         new_notes={"error": str(e)})
            fuel -= 1

    async def _process_task(self, agenda: Agenda, task: Task, status) -> None:
        concept_path = task.properties.get('concept')
        outcome = task.properties.get('outcome', 'unknown')
        origin_heuristic = task.properties.get('origin_heuristic')

        concept_obj = await agenda.get_object(concept_path) if concept_path else None

        # Step 1: Update heuristic worth
        if origin_heuristic:
            await self._update_heuristic_worth(agenda, origin_heuristic, outcome)

        # Step 2: Reflect via LLM
        if outcome == 'success' and concept_obj:
            await self._reflect_on_success(agenda, task, concept_obj, origin_heuristic)
        elif outcome == 'failure' and concept_obj:
            await self._reflect_on_failure(agenda, task, concept_obj, origin_heuristic)

        await agenda.update_task(task.id, work_status=WorkStatus.DONE,
                                 new_notes={"outcome": outcome})

    async def _update_heuristic_worth(
        self, agenda: Agenda, heuristic_name: str, outcome: str,
    ) -> None:
        """Update heuristic success tracking and interestingness."""
        h_obj = await agenda.get_object(f"heuristic/{heuristic_name}")
        if h_obj is None:
            return

        h_props = h_obj.properties
        attempts = h_props.get('attempts', 0)
        successes = h_props.get('successes', 0)
        before = h_obj.interestingness
        tracer = Tracer(agenda=agenda, worker="ReflectionWorker")

        if outcome == 'success':
            successes += 1
            await agenda.update_object(h_obj.path,
                                       new_properties={'successes': successes},
                                       interest_factor=self._interest_success_boost)
            logger.info("Heuristic %s: boosted (successes=%d, attempts=%d)",
                        heuristic_name, successes, attempts)
            await tracer.worth_update(
                heuristic=heuristic_name,
                before=before,
                after=before * self._interest_success_boost,
                cause="success",
                attempts=attempts,
                successes=successes,
            )
        else:
            await agenda.update_object(h_obj.path,
                                       interest_factor=self._interest_failure_decay)
            await tracer.worth_update(
                heuristic=heuristic_name,
                before=before,
                after=before * self._interest_failure_decay,
                cause="failure",
                attempts=attempts,
                successes=successes,
            )

        # Kill check
        if (attempts >= self._min_attempts_for_kill and
                attempts > 0 and successes / attempts < self._kill_threshold):
            await agenda.update_object(h_obj.path, interest_factor=0.01)
            logger.info("KILLED heuristic %s: success rate %.2f%% (%d/%d)",
                        heuristic_name, 100 * successes / attempts,
                        successes, attempts)
            await tracer.heuristic_death(
                name=heuristic_name,
                attempts=attempts,
                successes=successes,
                kill_threshold=self._kill_threshold,
            )

    async def _reflect_on_success(
        self, agenda: Agenda, task: Task, concept_obj: Object,
        heuristic_name: Optional[str],
    ) -> None:
        """Extract new concepts from a proved theorem."""
        c_props = concept_obj.properties

        # Build context for reflection
        proved = [{
            'name': c_props.get('name', ''),
            'description': c_props.get('description', ''),
            'lean_statement': c_props.get('lean_statement', ''),
            'lean_proof': c_props.get('lean_proof', ''),
        }]

        h_name = heuristic_name or 'unknown'
        h_obj = await agenda.get_object(f"heuristic/{h_name}")
        h_template = h_obj.content.decode('utf-8') if h_obj and h_obj.content else ''

        messages = [
            SystemMessage(content=system_reflect()),
            HumanMessage(content=format_reflect_user(h_name, h_template, proved, [])),
        ]

        try:
            response = self._llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning("Reflection LLM failed: %s", e)
            return

        await Tracer(agenda=agenda, worker="ReflectionWorker").reflection(
            target=c_props.get('name', '?'),
            prompt=format_reflect_user(h_name, h_template, proved, []),
            response=response_text,
            outcome="success",
            origin_heuristic=h_name,
        )

        result = parse_reflection_output(response_text)

        # Create new concepts from reflection
        concepts_created = 0
        for entry in result.get('concepts', []):
            name = entry.get('name', '')
            if not name:
                continue

            path = f"concept/{self._domain}/{name}"
            existing = await agenda.get_object(path)
            if existing is not None:
                continue

            stmt = entry.get('lean_statement', '')
            if stmt and await is_duplicate_statement(agenda, self._domain, stmt):
                continue

            kind = entry.get('kind', 'conjecture')
            obj_path = await agenda.create_object(Object(
                path=path,
                type="concept",
                parents=[concept_obj.path],
                content=entry.get('description', '').encode('utf-8'),
                properties={
                    'name': name,
                    'kind': kind,
                    'domain': self._domain,
                    'description': entry.get('description', ''),
                    'lean_statement': entry.get('lean_statement', ''),
                    'lean_imports': entry.get('lean_imports', []),
                    'tags': entry.get('tags', []),
                    'related_concepts': entry.get('related_concepts', []),
                    'origin_heuristic': f"reflection-{h_name}",
                    'proof_attempts': 0,
                    'proof_strategy': None,
                },
            ))

            if kind == 'conjecture':
                await agenda.add_task(Task(
                    id=f"prove-{name}",
                    type="prove",
                    parents=[task.id],
                    properties={'concept': obj_path, 'origin_heuristic': h_name},
                    interest_dependencies=[obj_path],
                ))
            else:
                await agenda.add_task(Task(
                    id=f"discover-{name}",
                    type="discover",
                    parents=[task.id],
                    properties={'concept': obj_path},
                    interest_dependencies=[obj_path],
                ))
            concepts_created += 1

        logger.info("Reflection on %s: %d new concepts extracted",
                    c_props.get('name', '?'), concepts_created)

        # Check for new heuristic proposal
        new_h = result.get('new_heuristic')
        if new_h:
            await self._create_heuristic(agenda, new_h, parent_heuristic=h_name)

    async def _reflect_on_failure(
        self, agenda: Agenda, task: Task, concept_obj: Object,
        heuristic_name: Optional[str],
    ) -> None:
        """Propose weakened versions of a failed conjecture."""
        c_props = concept_obj.properties

        failed = [{
            'name': c_props.get('name', ''),
            'description': c_props.get('description', ''),
            'lean_statement': c_props.get('lean_statement', ''),
        }]

        h_name = heuristic_name or 'unknown'
        h_obj = await agenda.get_object(f"heuristic/{h_name}")
        h_template = h_obj.content.decode('utf-8') if h_obj and h_obj.content else ''

        messages = [
            SystemMessage(content=system_reflect()),
            HumanMessage(content=format_reflect_user(h_name, h_template, [], failed)),
        ]

        try:
            response = self._llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning("Reflection LLM failed: %s", e)
            return

        await Tracer(agenda=agenda, worker="ReflectionWorker").reflection(
            target=c_props.get('name', '?'),
            prompt=format_reflect_user(h_name, h_template, [], failed),
            response=response_text,
            outcome="failure",
            origin_heuristic=h_name,
        )

        result = parse_reflection_output(response_text)

        for entry in result.get('concepts', []):
            name = entry.get('name', '')
            if not name:
                continue

            path = f"concept/{self._domain}/{name}"
            existing = await agenda.get_object(path)
            if existing is not None:
                continue

            stmt = entry.get('lean_statement', '')
            if stmt and await is_duplicate_statement(agenda, self._domain, stmt):
                continue

            tags = entry.get('tags', [])
            if 'weakened' not in tags:
                tags.append('weakened')

            obj_path = await agenda.create_object(Object(
                path=path,
                type="concept",
                parents=[concept_obj.path],
                content=entry.get('description', '').encode('utf-8'),
                properties={
                    'name': name,
                    'kind': 'conjecture',
                    'domain': self._domain,
                    'description': entry.get('description', ''),
                    'lean_statement': entry.get('lean_statement', ''),
                    'lean_imports': entry.get('lean_imports', []),
                    'tags': tags,
                    'related_concepts': entry.get('related_concepts', []),
                    'origin_heuristic': f"weakened-{h_name}",
                    'proof_attempts': 0,
                    'proof_strategy': None,
                },
            ))

            await agenda.add_task(Task(
                id=f"prove-{name}",
                type="prove",
                parents=[task.id],
                properties={'concept': obj_path, 'origin_heuristic': h_name},
                interest_dependencies=[obj_path],
            ))

        logger.info("Reflection on failure of %s: %d weakened conjectures proposed",
                    c_props.get('name', '?'), len(result.get('concepts', [])))

    async def _create_heuristic(self, agenda: Agenda, h_spec: dict, *, parent_heuristic: str = "unknown") -> None:
        """Create a new heuristic Object from a reflection proposal."""
        name = h_spec.get('name', '')
        if not name:
            return

        # Sanitize
        import re
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name).strip('_').lower()
        path = f"heuristic/{name}"

        existing = await agenda.get_object(path)
        if existing is not None:
            return

        kind = h_spec.get('kind', 'concept')
        template = h_spec.get('template', '')

        await agenda.create_object(Object(
            path=path,
            type="heuristic",
            content=template.encode('utf-8'),
            properties={
                'name': name,
                'heuristic_kind': kind,
                'input_concept_kinds': [],
                'input_tags': [],
                'eurisclo_origin': None,
                'attempts': 0,
                'successes': 0,
                'born_from_reflection': True,
            },
        ))
        logger.info("NEW HEURISTIC BORN: %s (kind: %s)", name, kind)
        await Tracer(agenda=agenda, worker="ReflectionWorker").heuristic_birth(
            name=name,
            parent_heuristic=parent_heuristic,
            template=template,
            heuristic_kind=kind,
        )
