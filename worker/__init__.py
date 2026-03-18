"""Worker abstraction.

Worker is the abstract base for all actors in the discovery system.
_to_langchain_messages converts the language-agnostic ChatMessage list returned
by PromptBuilder methods into LangChain BaseMessage objects.
"""

from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agenda import Agenda
from language import ChatMessage


def _to_langchain_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    """Convert a PromptBuilder message list to LangChain BaseMessage objects."""
    mapping = {"system": SystemMessage, "user": HumanMessage}
    return [mapping[m["role"]](content=m["content"]) for m in messages]


class Worker(ABC):
    """
    These are the main actors in the discovery system:
    workers add and perform tasks from the agenda, and create and update objects.
    """
    @abstractmethod
    async def work(self, agenda: Agenda, fuel: int) -> None:
        """
        Perform some work against the agenda, consuming `fuel` units.

        For instance, fuel can correspond to how many tasks the worker should attempt.
        Specific semantics of "fuel" are worker-specific, but higher fuel should do more work.
        """
        raise NotImplementedError
