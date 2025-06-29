

from abc import ABC
from dataclasses import dataclass
from typing import Any

from evals.evaluators.types import MessageList


@dataclass
class AgentResult:
    """
    Result of running an agent.
    """
    response: str
    info: dict[str, Any] # token usage, critical path, trace


class AgentBase(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, message_list: MessageList) -> AgentResult:
        raise NotImplementedError("Subclasses must implement this method.")
