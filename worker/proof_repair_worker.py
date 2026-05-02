"""Worker that repairs failed proof attempts using LLM with error feedback."""

import logging
import os
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agenda import Agenda, Object, Task, WorkStatus
from discovery import format_concepts_for_prompt, gather_definitions, resolve_imports, strip_duplicate_decls
from language import Language, Program, VerificationOutcome

from . import Worker

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover debugging a failed proof.\n"
    "You will be given:\n"
    "- The theorem statement\n"
    "- A failed proof attempt\n"
    "- The Lean error output from that attempt\n"
    "- Definitions of custom concepts used in the statement\n\n"
    "Your job is to produce a COMPLETE, CORRECTED Lean 4 file with a working proof.\n\n"
    "Rules:\n"
    "- Include all necessary imports at the top\n"
    "- Include any custom definitions provided — do not omit them\n"
    "- The file must compile without errors when checked by Lean\n"
    "- Do NOT use `sorry` anywhere\n"
    "- Read the error messages carefully and fix the specific issues\n"
    "- Common issues: missing imports, wrong tactic, type mismatches, universe errors\n"
    "- Output only valid Lean 4 source code\n"
)


class ProofRepairWorker(Worker):
    """Repairs failed proof attempts using error feedback from Lean.

    For each 'proof_repair' task:
      1. Load the concept and the failed proof + error output
      2. Invoke LLM with error context to produce a corrected proof
      3. Verify the result
      4. On success: promote concept to theorem, create follow-up tasks
      5. On failure: re-queue with updated error if attempts remain
    """

    def __init__(
        self,
        llm: Any,
        language: str = 'lean',
        domain: str = 'natural_numbers',
        max_attempts: int = 3,
        interest_success: float = 3.0,
        interest_fail: float = 0.7,
        interest_recursion_gamma: float = 0.5,
        lake_project_dir: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._backend = Language[language.upper()].get_backend()
        self._language = language.lower()
        self._domain = domain
        self._max_attempts = max_attempts
        self._interest_success = interest_success
        self._interest_fail = interest_fail
        self._interest_recursion_gamma = interest_recursion_gamma
        self._lake_project_dir = lake_project_dir or os.environ.get('LEAN_PROJECT_DIR')

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            result = await agenda.claim_next_tasks(type="proof_repair")
            if result is None:
                break

            task, status = result[0]
            try:
                await self._process_task(agenda, task, status)
            except Exception as e:
                logger.exception("ProofRepairWorker error on task %s", task.id)
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

        c_props = concept_obj.properties

        # Already proved by another path
        if c_props.get('kind') == 'theorem':
            logger.info("Concept %s already proved, skipping repair", c_props.get('name', '?'))
            await agenda.update_task(task.id, work_status=WorkStatus.DONE,
                                     new_notes={"skipped": "already proved"})
            return

        statement = c_props.get('lean_statement', '')
        imports = c_props.get('lean_imports', [])
        failed_code = task.properties.get('failed_code', '')
        error_stdout = task.properties.get('error_stdout', '')
        error_stderr = task.properties.get('error_stderr', '')

        # Gather definition preamble (same as ProofWorker)
        preamble_defs = await self._gather_definitions(agenda, concept_obj)
        preamble = '\n\n'.join(preamble_defs)

        # Combine error output (Lean reports errors on stdout)
        error_text = ''
        if error_stdout:
            error_text += error_stdout.strip()
        if error_stderr:
            if error_text:
                error_text += '\n\n'
            error_text += error_stderr.strip()
        if not error_text:
            error_text = '(no error output captured)'

        # Build the user prompt
        parts = [
            f"## Theorem to prove\n\n```lean\n{statement}\n```",
            f"\n## Required imports\n\n```lean\n{chr(10).join(f'import {imp}' for imp in resolve_imports(imports))}\n```",
        ]
        if preamble:
            parts.append(f"\n## Custom definitions used in the statement\n\n```lean\n{preamble}\n```")
        parts.append(f"\n## Failed proof attempt\n\n```lean\n{failed_code}\n```")
        parts.append(f"\n## Lean error output\n\n```\n{error_text}\n```")
        parts.append(
            "\nFix the errors and produce a complete, self-contained Lean 4 file "
            "with all imports, definitions, and a working proof."
        )
        user_msg = '\n'.join(parts)

        # Invoke LLM
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
            response = self._llm.invoke(messages)
            repaired_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning("LLM invocation failed for proof repair %s: %s", task.id, e)
            await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED,
                                     priority_factor=self._interest_fail)
            return

        repaired_text = self._extract_code(repaired_text)

        # Verify
        prog = Program(repaired_text, Language[self._language.upper()],
                        name=c_props.get('name', 'proof_repair'))
        ver = self._backend.verify(prog)

        logger.info("Proof repair for %s: %s", c_props.get('name', '?'), ver.outcome.name)

        if ver.outcome == VerificationOutcome.SUCCESS:
            await self._handle_success(agenda, task, concept_obj, repaired_text)
            return

        # Failed — re-queue with updated error for another attempt
        repair_attempts = status.worker_notes.get('repair_attempts', 0) + 1
        if repair_attempts >= self._max_attempts:
            await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                     new_notes={"repair_attempts": repair_attempts})
        else:
            # Update the task with the new failed code and error for next attempt
            await agenda.update_task(
                task.id,
                work_status=WorkStatus.ATTEMPTED,
                priority_factor=self._interest_fail,
                new_notes={
                    "repair_attempts": repair_attempts,
                    "last_error_stdout": ver.stdout[:2000],
                    "last_error_stderr": ver.stderr[:2000],
                },
            )
            # Update the task properties so the next attempt sees the latest failure
            task.properties['failed_code'] = repaired_text
            task.properties['error_stdout'] = ver.stdout
            task.properties['error_stderr'] = ver.stderr

    async def _gather_definitions(self, agenda: Agenda, concept_obj: Object) -> list[str]:
        """Gather definitions needed for the proof: from related_concepts + statement scanning."""
        c_props = concept_obj.properties
        return await gather_definitions(
            agenda, self._domain,
            c_props.get('lean_statement', ''),
            c_props.get('related_concepts', []),
        )

    async def _handle_success(
        self, agenda: Agenda, task: Task, concept_obj: Object, proof_text: str,
    ) -> None:
        """Promote a conjecture to theorem after successful repair."""
        c_props = concept_obj.properties
        name = c_props.get('name', 'unnamed')

        # Update concept: conjecture -> theorem
        await agenda.update_object(
            concept_obj.path,
            new_content=proof_text.encode('utf-8'),
            new_properties={
                'kind': 'theorem',
                'lean_proof': proof_text,
                'proof_strategy': 'repair',
            },
            interest_factor=self._interest_success,
            interest_recursion_gamma=self._interest_recursion_gamma,
        )

        # Create dataset entry
        await agenda.create_object(Object(
            path=f"dataset/theorem_{name}.lean",
            type=f"{self._language}-program",
            parents=[concept_obj.path],
            content=proof_text.encode('utf-8'),
            properties={
                'verification_status': 'success',
                'parent_idea': concept_obj.path,
                'origin_heuristic': c_props.get('origin_heuristic'),
            },
        ))

        # Write to LeanDisco library
        self._write_to_library(name, proof_text, c_props.get('domain', 'General'))

        # Create reflect task (success)
        await agenda.add_task(Task(
            id=f"reflect-{name}",
            type="reflect",
            parents=[task.id],
            properties={
                'concept': concept_obj.path,
                'outcome': 'success',
                'origin_heuristic': task.properties.get('origin_heuristic'),
            },
            interest_dependencies=[concept_obj.path],
        ))

        # Create discover task for the proved theorem
        await agenda.add_task(Task(
            id=f"discover-{name}",
            type="discover",
            parents=[task.id],
            properties={'concept': concept_obj.path},
            interest_dependencies=[concept_obj.path],
        ))

        # Boost origin heuristic
        origin = task.properties.get('origin_heuristic')
        if origin:
            h_obj = await agenda.get_object(f"heuristic/{origin}")
            if h_obj:
                s = h_obj.properties.get('successes', 0) + 1
                await agenda.update_object(h_obj.path,
                                           new_properties={'successes': s},
                                           interest_factor=1.1)

        await agenda.update_task(task.id, work_status=WorkStatus.DONE,
                                 new_notes={"proved_with": "repair"})

        logger.info("THEOREM PROVED (via repair): %s", name)

    def _write_to_library(self, name: str, proof_text: str, domain: str) -> None:
        """Write a proved theorem to the LeanDisco library."""
        if not self._lake_project_dir:
            return
        domain_dir = domain.replace('_', '').title().replace(' ', '')
        lib_dir = os.path.join(self._lake_project_dir, 'LeanDisco', 'Domains', domain_dir)
        try:
            os.makedirs(lib_dir, exist_ok=True)
            module_name = ''.join(word.capitalize() for word in name.split('_'))
            filepath = os.path.join(lib_dir, f'{module_name}.lean')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(proof_text)
            logger.info("Wrote theorem to library: %s", filepath)
        except Exception as e:
            logger.warning("Failed to write to library: %s", e)

    def _extract_code(self, text: str) -> str:
        """Extract Lean code from markdown fences if present."""
        match = re.search(r'```lean\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
