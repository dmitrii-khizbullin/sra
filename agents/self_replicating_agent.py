from typing import Any

from agents.vllm_like import LLM
from agents.agent_base import AgentBase, AgentResult, MessageList
from agents.fork_manager import ForkManager


class SelfReplicatingAgent(AgentBase):
    def __init__(self,
                 port: int = 8000,
                 extra_tools: list[dict[str, Any]] | None = None,
                 artifact_dir: str | None = None,
                 **kwargs
                 ) -> None:
        """ Supports only vLLM server running on localhost:port/v1 """
        super().__init__()

        self.extra_tools = extra_tools # not used so far
        base_url = f"http://localhost:{port}/v1"
        self.llm = LLM(base_url=base_url)
        self.fork_manager = ForkManager(self.llm, extra_tools, artifact_dir=artifact_dir)

    def __call__(self, message_list: MessageList) -> AgentResult:

        flattened_messages = ""
        for message in message_list:
            if message['role'] in {'system', 'user'}:
                flattened_messages += f"User: {message['content']}\n"
            else:
                raise ValueError(f"Unknown role in message: {message['role']}")

        response_str = self.fork_manager.run_entry(flattened_messages)
        if response_str is None:
            response_str = ""

        agent_result = AgentResult(
            response=response_str,
            info={
                "token_usage": None, # TODO fill
                "critical_path": None,  # TODO fill
                "complete_message_history": None, # does not exist
                "trace": None,  # TODO fill
            }
        )
        return agent_result
