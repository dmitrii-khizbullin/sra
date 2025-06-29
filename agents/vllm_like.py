from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any
import warnings

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


class LLM:
    """
    A drop-in replacement for vLLM's LLM class that connects to a standalone vLLM server.
    Maintains API compatibility with the original vLLM LLM interface.
    """
    
    def __init__(self, 
                 model: str | None = None,
                 base_url: str = "http://localhost:8000/v1",
                 api_key: str = "dummy-key",
                 tensor_parallel_size: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 **kwargs):
        """
        Initialize the LLM client to connect to a standalone vLLM server.
        
        Args:
            model: Model name (must match the model loaded on the server)
            base_url: URL of the vLLM server
            api_key: API key (not required for local vLLM servers)
            tensor_parallel_size: Ignored (handled by server configuration)
            gpu_memory_utilization: Ignored (handled by server configuration)
            **kwargs: Additional arguments (mostly ignored for server client)
        """
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        if self.model is not None:
            # warn that the model is ignored and the one that is there in the server will be used
            warnings.warn(
                "Model specified is ignored when using server client. "
                "The model loaded on the server will be used."
            )
            self.model = None
        if self.model is None:
            # load the first model from the vllm server
            try:
                models = self.client.models.list()
            except Exception as e:
                raise RuntimeError("vLLM server is not running or not reachable.") from e
            models = models.data
            self.model = models[0].id if models else None
        if self.model is None:
            raise ValueError("No model specified and no models available on the server.")

        # Warn about ignored parameters
        if tensor_parallel_size is not None or gpu_memory_utilization is not None:
            warnings.warn(
                "tensor_parallel_size and gpu_memory_utilization are ignored when using "
                "server client. These should be configured when starting the server."
            )
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             sampling_params: Optional[Any] = None,
             tools: Optional[List[Dict[str, Any]]] = None) -> ChatCompletion:
        """
        Generate a chat completion with tool support.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            sampling_params: SamplingParams object
            tools: List of tool definitions for function calling
            
        Returns:
            List containing a RequestOutput-like object (for compatibility with notebook)
        """
        # Extract sampling parameters
        max_tokens = 100
        temperature = 0.0
        top_p = 1.0
        
        if sampling_params is not None:
            max_tokens = getattr(sampling_params, 'max_tokens', max_tokens)
            temperature = getattr(sampling_params, 'temperature', temperature)
            top_p = getattr(sampling_params, 'top_p', top_p)
        
        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**api_params)
            
        return response


@dataclass
class SamplingParams:
    """Dataclass to hold sampling parameters."""
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0


# Example usage:
if __name__ == "__main__":
    # This is now a drop-in replacement for your original code:
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", tensor_parallel_size=4, gpu_memory_utilization=0.5)
        
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
    
    # This should work with your notebook's fork_manager
    response = llm.chat(messages, sampling_params=sampling_params)
    print(response.choices[0].message.content)
    print(response.usage)
    print("Done.")
