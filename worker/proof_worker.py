"""Worker that attempts to prove conjectures using Lean 4."""

import logging
import os
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agenda import Agenda, Object, Task, WorkStatus
from discovery import format_concepts_for_prompt, gather_definitions, resolve_imports, strip_duplicate_decls
from discovery.prompts import system_prove, format_prove_user
from discovery.trace import Tracer
from discovery.worth import difficulty_from_proof, update_heuristic_worth
from language import Language, Program, VerificationOutcome

from . import Worker

logger = logging.getLogger(__name__)


class ProofWorker(Worker):
    """Attempts to prove conjectures using LLM-generated Lean 4 proofs.

    For each 'prove' task:
      1. Load the conjecture concept
      2. Select proof strategy heuristics
      3. For each strategy, invoke LLM and verify the result
      4. On success: update concept to theorem, create reflect/discover tasks
      5. On failure: create repair task or reflect task
    """

    def __init__(
        self,
        llm: Any,
        language: str = 'lean',
        domain: str = 'group_theory',
        max_attempts: int = 3,
        max_strategies: int = 2,
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
        self._max_strategies = max_strategies
        self._interest_success = interest_success
        self._interest_fail = interest_fail
        self._interest_recursion_gamma = interest_recursion_gamma
        self._lake_project_dir = lake_project_dir or os.environ.get('LEAN_PROJECT_DIR')

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            result = await agenda.claim_next_tasks(type="prove")
            if result is None:
                break

            task, status = result[0]
            try:
                used_llm = await self._process_task(agenda, task, status)
            except Exception as e:
                logger.exception("ProofWorker error on task %s", task.id)
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                         new_notes={"error": str(e)})
                used_llm = True
            # Only burn fuel for LLM attempts; embedded proofs are cheap
            if used_llm:
                fuel -= 1

    async def _process_task(self, agenda: Agenda, task: Task, status) -> bool:
        """Process a prove task. Returns True if the LLM was invoked."""
        concept_path = task.properties.get('concept')
        if not concept_path:
            await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                     new_notes={"error": "no concept path"})
            return False

        concept_obj = await agenda.get_object(concept_path)
        if concept_obj is None:
            await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                     new_notes={"error": f"concept not found: {concept_path}"})
            return False

        c_props = concept_obj.properties
        statement = c_props.get('lean_statement', '')
        imports = c_props.get('lean_imports', [])

        # Load related concepts for context and gather definition preamble
        related_names = c_props.get('related_concepts', [])
        context_concepts = []
        for name in related_names[:10]:
            related = await agenda.get_object(f"concept/{self._domain}/{name}")
            if related is not None:
                context_concepts.append(related)

        preamble_defs = await gather_definitions(agenda, self._domain, statement, related_names)
        related_text = format_concepts_for_prompt(context_concepts) if context_concepts else ''
        preamble = '\n\n'.join(preamble_defs)

        # If the statement already contains a proof (from discovery), try it first
        if ':= by' in statement or ':= sorry' not in statement:
            proof_lines = [l for l in statement.split('\n')
                           if not l.strip().startswith('import ')]
            proof_body = strip_duplicate_decls('\n'.join(proof_lines).strip(), preamble)
            imports_str = '\n'.join(f'import {imp}' for imp in resolve_imports(imports))
            candidate = imports_str + '\n\n' + preamble + '\n\n' + proof_body if preamble else imports_str + '\n\n' + proof_body

            prog = Program(candidate, Language[self._language.upper()],
                           name=c_props.get('name', 'proof'))
            ver = self._backend.verify(prog)
            if ver.outcome == VerificationOutcome.SUCCESS:
                logger.info("Embedded proof verified for %s", c_props.get('name', '?'))
                await Tracer(agenda=agenda, worker="ProofWorker").proof_attempt(
                    theorem=c_props.get('name', '?'),
                    strategy='embedded',
                    outcome='success',
                    domain=self._domain,
                )
                await self._handle_success(agenda, task, concept_obj, candidate, 'embedded')
                return False
            else:
                # Embedded proof failed — skip LLM on first attempt, come back later
                proof_attempts = c_props.get('proof_attempts', 0)
                if proof_attempts == 0:
                    logger.info("Embedded proof failed for %s, deferring LLM", c_props.get('name', '?'))
                    await agenda.update_object(concept_path,
                                               new_properties={'proof_attempts': 1})
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED,
                                             priority_factor=self._interest_fail,
                                             new_notes={"proof_attempts": 1, "embedded_failed": True})
                    return False

        # Load proof strategy heuristics
        strategies = []
        for h_name in ['direct_tactic_proof', 'structured_proof']:
            h_obj = await agenda.get_object(f"heuristic/{h_name}")
            if h_obj is not None:
                strategies.append(h_obj)
        strategies.sort(key=lambda h: h.interestingness, reverse=True)
        strategies = strategies[:self._max_strategies]

        # Try each strategy, keep the best failure for repair
        best_failure: dict | None = None
        for strategy in strategies:
            s_name = strategy.properties.get('name', 'unknown')
            proof_hint = strategy.content.decode('utf-8') if strategy.content else ''

            # Update strategy attempts
            s_attempts = strategy.properties.get('attempts', 0) + 1
            await agenda.update_object(strategy.path,
                                       new_properties={'attempts': s_attempts})

            # Invoke LLM
            messages = [
                SystemMessage(content=system_prove()),
                HumanMessage(content=format_prove_user(
                    statement, imports, related_text, proof_hint,
                    preamble=preamble,
                )),
            ]

            try:
                response = self._llm.invoke(messages)
                proof_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.warning("LLM invocation failed for strategy %s: %s", s_name, e)
                await agenda.update_object(strategy.path,
                                           interest_factor=0.95)
                continue

            # Extract code from markdown fences if present
            proof_text = self._extract_code(proof_text)

            # Mechanically assemble the file: our imports + definitions + LLM proof
            # Strip import lines and duplicate declarations from LLM output
            proof_lines = [l for l in proof_text.split('\n')
                           if not l.strip().startswith('import ')]
            proof_body = strip_duplicate_decls('\n'.join(proof_lines).strip(), preamble)

            imports_str = '\n'.join(f'import {imp}' for imp in resolve_imports(imports))
            header = imports_str
            if preamble:
                header += '\n\n' + preamble
            proof_text = header + '\n\n' + proof_body

            # Verify
            prog = Program(proof_text, Language[self._language.upper()],
                           name=c_props.get('name', 'proof'))
            ver = self._backend.verify(prog)

            logger.info("Proof attempt for %s with %s: %s",
                        c_props.get('name', '?'), s_name, ver.outcome.name)

            if ver.outcome == VerificationOutcome.SUCCESS:
                # Update strategy: success
                s_successes = strategy.properties.get('successes', 0) + 1
                await agenda.update_object(strategy.path,
                                           new_properties={'successes': s_successes},
                                           interest_factor=1.1)

                await Tracer(agenda=agenda, worker="ProofWorker").proof_attempt(
                    theorem=c_props.get('name', '?'),
                    strategy=s_name,
                    outcome='success',
                    domain=self._domain,
                )
                await self._handle_success(agenda, task, concept_obj, proof_text, s_name)
                return True

            # Update strategy: failure — keep best attempt for repair
            await agenda.update_object(strategy.path, interest_factor=0.95)
            await Tracer(agenda=agenda, worker="ProofWorker").proof_attempt(
                theorem=c_props.get('name', '?'),
                strategy=s_name,
                outcome='failure',
                domain=self._domain,
                error_excerpt=(ver.stderr or ver.stdout or '')[:600],
            )

            # Prefer the attempt with more error output (more to learn from)
            error_len = len(ver.stdout) + len(ver.stderr)
            if best_failure is None or error_len > best_failure['error_len']:
                best_failure = {
                    'failed_code': proof_text,
                    'error_stdout': ver.stdout,
                    'error_stderr': ver.stderr,
                    'strategy': s_name,
                    'error_len': error_len,
                }

        # All strategies failed — send best attempt to repair
        if best_failure:
            await agenda.add_task(Task(
                id=f"proof_repair-{c_props.get('name', '')}",
                type="proof_repair",
                parents=[task.id],
                properties={
                    'concept': concept_path,
                    'failed_code': best_failure['failed_code'],
                    'error_stdout': best_failure['error_stdout'],
                    'error_stderr': best_failure['error_stderr'],
                    'strategy': best_failure['strategy'],
                    'origin_heuristic': task.properties.get('origin_heuristic'),
                },
                interest_dependencies=[concept_path],
            ))
        proof_attempts = c_props.get('proof_attempts', 0) + 1
        await agenda.update_object(concept_path,
                                   new_properties={'proof_attempts': proof_attempts})

        if proof_attempts >= self._max_attempts:
            await agenda.update_object(concept_path,
                                       interest_factor=self._interest_fail,
                                       interest_recursion_gamma=self._interest_recursion_gamma)
            # Create reflect task for failure analysis
            await agenda.add_task(Task(
                id=f"reflect-fail-{c_props.get('name', '')}",
                type="reflect",
                parents=[task.id],
                properties={
                    'concept': concept_path,
                    'outcome': 'failure',
                    'origin_heuristic': task.properties.get('origin_heuristic'),
                },
                interest_dependencies=[concept_path],
            ))
            await agenda.update_task(task.id, work_status=WorkStatus.FAILED,
                                     new_notes={"proof_attempts": proof_attempts})
        else:
            await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED,
                                     priority_factor=self._interest_fail,
                                     new_notes={"proof_attempts": proof_attempts})
        return True

    async def _handle_success(
        self, agenda: Agenda, task: Task, concept_obj: Object,
        proof_text: str, strategy_name: str,
    ) -> None:
        """Handle a successfully proved theorem."""
        c_props = concept_obj.properties
        name = c_props.get('name', 'unnamed')

        # Update concept: conjecture -> theorem
        await agenda.update_object(
            concept_obj.path,
            new_content=proof_text.encode('utf-8'),
            new_properties={
                'kind': 'theorem',
                'lean_proof': proof_text,
                'proof_strategy': strategy_name,
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

        # Write to LeanDisco library if available
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

        # Credit the origin heuristic with a prove + difficulty (multiplicative worth).
        origin = task.properties.get('origin_heuristic') or c_props.get('origin_heuristic')
        if origin:
            await update_heuristic_worth(
                agenda, origin,
                proves_delta=1,
                difficulty_delta=difficulty_from_proof(proof_text, repaired=False),
            )

        await agenda.update_task(task.id, work_status=WorkStatus.DONE,
                                 new_notes={"proved_with": strategy_name})

        logger.info("THEOREM PROVED: %s (strategy: %s)", name, strategy_name)

    def _write_to_library(self, name: str, proof_text: str, domain: str) -> None:
        """Write a proved theorem to the LeanDisco library."""
        if not self._lake_project_dir:
            return
        # Map domain to directory name
        domain_dir = domain.replace('_', '').title().replace(' ', '')
        lib_dir = os.path.join(self._lake_project_dir, 'LeanDisco', 'Domains', domain_dir)
        try:
            os.makedirs(lib_dir, exist_ok=True)
            # Convert snake_case name to PascalCase for Lean module
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
