#!/usr/bin/env python3

# Add this to worker.py

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

import random

from . import Worker
from agenda import Agenda, Object, Task, WorkStatus  # already used above

from langchain_core.prompts import ChatPromptTemplate

from code_output_parser import CodeOutputParser

from prompt import format_idea_user, system_idea

logger = logging.getLogger(__name__)


class ReadmeInspiredIdeaGenerator(Worker):
    """
    A Worker that generates Dafny program ideas inspired by random GitHub READMEs.

    Constructor:
        jsonl_path: path to a JSONL where each line is {"repo": str, "readme": str}
        llm: a LangChain Runnable/LLM (e.g., ChatOpenAI, ChatAnthropic, etc.)
        rng: optional random.Random for reproducibility
        max_readme_chars: truncate README to this many characters before prompting

    work(agenda, fuel):
        For each unit of fuel:
          - Sample a random row from the dataset
          - Prompt LLM to produce a Dafny program idea/spec
          - Create an 'idea/...txt' object with the LLM output
          - Enqueue an 'implement' task with {'idea': idea_path}
    """

    def __init__(
        self,
        jsonl_path: str,
        llm: Any,
        rng: Optional[random.Random] = None,
        max_readme_chars: int = 2000,
        distill: Optional[Literal['success-only', 'all']] = 'success-only',
    ) -> None:
        self._rng = rng or random.Random()
        self._rows = self._load_jsonl(jsonl_path)
        if not self._rows:
            raise ValueError(f"No valid rows found in JSONL: {jsonl_path}")

        self._llm = llm
        self._max_readme_chars = max_readme_chars
        self._distill = distill

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_idea(),
                ),
                (
                    "human",
                    format_idea_user(repo="{repo}", readme="{readme}"),
                ),
            ]
        )

        # Chain: prompt -> LLM -> text
        self._chain = self._prompt | self._llm | CodeOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        for _ in range(fuel):
            row = self._rng.choice(self._rows)
            repo = row.get("repo", "").strip() or "unknown/repo"
            readme = (row.get("readme") or "").strip()
            readme = self._truncate(readme, self._max_readme_chars)

            # Run the LLM synchronously (LangChain invoke is sync).
            idea_text = self._chain.invoke({"repo": repo, "readme": readme}).strip()

            if self._distill:
                distill_obj = {
                    "prompt": "idea",
                    "arguments": {"repo": repo, "readme": readme},
                    "response": idea_text,
                    "outcome": "success",
                }
                await agenda.create_object(
                    Object(
                        path="distil/example.json",
                        type="distill-example",
                        parents=[],
                        content=json.dumps(distill_obj, ensure_ascii=False).encode("utf-8"),
                    )
                )

            # Persist as an 'idea' object.
            idea_filename = self._unique_idea_filename(repo)
            idea_path = await agenda.create_object(
                Object(
                    path=idea_filename,
                    type="idea",
                    content=idea_text.encode("utf-8"),
                    properties={"repo": repo},
                )
            )

            # Enqueue a corresponding 'implement' task.
            task = Task(id="imp", type="implement", properties={"idea": idea_path})
            await agenda.add_task(task)


    def _load_jsonl(self, path: str) -> list[dict]:
        rows: list[dict] = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "repo" in obj and "readme" in obj:
                    rows.append(obj)

        logger.info(f"Loaded {len(rows)} repositories with README from {path}")

        return rows

    def _truncate(self, s: str, n: int) -> str:
        if len(s) <= n:
            return s
        return s[:n - 3] + "..."

    def _unique_idea_filename(self, repo: str) -> str:
        return f"idea/{self._slugify(repo)}.txt"

    def _slugify(self, s: str) -> str:
        # Replace invalid path characters characters with '-'
        out = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_"):
                out.append(ch)
            else:
                out.append("-")
        # collapse multiple dashes
        slug = "".join(out)
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "github-repo"
