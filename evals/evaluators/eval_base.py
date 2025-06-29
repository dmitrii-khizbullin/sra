from typing import Callable

from agents.agent_base import AgentBase
from evals.evaluators.types import EvalResult


class EvalBase:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, agent_factory: Callable[[str], AgentBase]) -> EvalResult:
        raise NotImplementedError
