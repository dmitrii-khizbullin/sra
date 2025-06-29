from typing import Any

from .vllm_like import LLM, SamplingParams
from .agent_base import AgentBase, AgentResult, MessageList


class BaselineAgent(AgentBase):
    def __init__(self,
                 port: int = 8000,
                 sampling_params = SamplingParams(max_tokens=10_000, temperature=0.0),
                 extra_tools: list[dict[str, Any]] | None = None,
                 **kwargs
                 ) -> None:
        """ Supports only vLLM server running on localhost:port/v1 """
        super().__init__()

        self.sampling_params = sampling_params
        self.extra_tools = extra_tools # not used so far
        base_url = f"http://localhost:{port}/v1"
        self.llm = LLM(base_url=base_url)

    def __call__(self, message_list: MessageList) -> AgentResult:

        response = self.llm.chat(message_list, self.sampling_params, self.extra_tools)

        response_str = response.choices[0].message.content

        if response_str is None:
            response_str = ""

        complete_message_history = message_list + [{'role': 'assistant', 'content': response_str}]

        agent_result = AgentResult(
            response=response_str,
            info={
                "token_usage": response.usage,
                "critical_path": None,  # The same as the response
                "complete_message_history": complete_message_history,
                "trace": None,  # Not needed
            }
        )
        return agent_result
