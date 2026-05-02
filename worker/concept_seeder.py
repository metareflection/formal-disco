"""One-shot worker that seeds the agenda with initial concepts and heuristics."""

import json
import logging
from typing import Any

from agenda import Agenda, Object, Task
from discovery.heuristics_seed import INITIAL_HEURISTICS

from . import Worker

logger = logging.getLogger(__name__)


class ConceptSeeder(Worker):
    """Seeds the agenda with initial concepts and heuristics, then stops.

    This worker runs once: it checks if seeding has already been done
    (by looking for a sentinel object), and if not, creates:
      - Heuristic Objects from INITIAL_HEURISTICS
      - Concept Objects from a domain-specific seed file
      - A 'discover' task for each concept
    """

    def __init__(
        self,
        domain: str = "group_theory",
        seed_file: str = "data/seed_group_theory.json",
    ) -> None:
        self._domain = domain
        self._seed_file = seed_file

    async def work(self, agenda: Agenda, fuel: int) -> None:
        # Check if already seeded
        sentinel = await agenda.get_object(f"_seeded/{self._domain}")
        if sentinel is not None:
            return

        # Load seed concepts
        with open(self._seed_file, 'r') as f:
            seed_concepts = json.load(f)

        # Create heuristic Objects
        for h in INITIAL_HEURISTICS:
            path = f"heuristic/{h['name']}"
            await agenda.create_object(Object(
                path=path,
                type="heuristic",
                content=h['template'].encode('utf-8'),
                properties={
                    'name': h['name'],
                    'heuristic_kind': h['heuristic_kind'],
                    'input_concept_kinds': h.get('input_concept_kinds', []),
                    'input_tags': h.get('input_tags', []),
                    'eurisclo_origin': h.get('eurisclo_origin'),
                    'attempts': 0,
                    'successes': 0,
                },
            ))
            logger.info("Seeded heuristic: %s", h['name'])

        # Create concept Objects and discover tasks
        for concept in seed_concepts:
            name = concept['name']
            path = f"concept/{self._domain}/{name}"
            await agenda.create_object(Object(
                path=path,
                type="concept",
                content=concept.get('description', '').encode('utf-8'),
                properties={
                    'name': name,
                    'kind': concept.get('kind', 'definition'),
                    'domain': self._domain,
                    'description': concept.get('description', ''),
                    'lean_statement': concept.get('lean_statement', ''),
                    'lean_proof': concept.get('lean_proof'),
                    'lean_imports': concept.get('lean_imports', []),
                    'tags': concept.get('tags', []),
                    'related_concepts': concept.get('related', []),
                    'origin_heuristic': None,
                    'proof_attempts': 0,
                    'proof_strategy': None,
                },
            ))

            # Create discover task for this concept
            await agenda.add_task(Task(
                id=f"discover-{name}",
                type="discover",
                properties={'concept': path},
                interest_dependencies=[path],
            ))
            logger.info("Seeded concept: %s", name)

        # Mark as seeded
        await agenda.create_object(Object(
            path=f"_seeded/{self._domain}",
            type="sentinel",
            content=b"seeded",
        ))
        logger.info("Seeding complete: %d heuristics, %d concepts",
                     len(INITIAL_HEURISTICS), len(seed_concepts))
