from agents.agent_base import AgentBase
from .types import EvalResult


class EvalBase:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, agent: AgentBase) -> EvalResult:
        raise NotImplementedError
