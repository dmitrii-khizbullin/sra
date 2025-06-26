from dataclasses import dataclass
from openai import OpenAI
from typing import List, Union, Optional, Dict, Any
import warnings
import json

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
            models = self.client.models.list()
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
    
    def generate(self, 
                 prompts: Union[str, List[str]], 
                 sampling_params: Optional[Any] = None,
                 use_tqdm: bool = True) -> List[Any]:
        """
        Generate completions for the given prompts.
        
        Args:
            prompts: Single prompt string or list of prompt strings
            sampling_params: SamplingParams object (will extract relevant parameters)
            use_tqdm: Ignored (no progress bar for API calls)
            
        Returns:
            List of RequestOutput-like objects
        """
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Extract sampling parameters
        max_tokens = 100
        temperature = 1.0
        top_p = 1.0
        top_k = -1
        stop = None
        
        if sampling_params is not None:
            max_tokens = getattr(sampling_params, 'max_tokens', max_tokens)
            temperature = getattr(sampling_params, 'temperature', temperature)
            top_p = getattr(sampling_params, 'top_p', top_p)
            top_k = getattr(sampling_params, 'top_k', top_k)
            stop = getattr(sampling_params, 'stop', stop)
        
        results = []
        
        for prompt in prompts:
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop if stop else None
                )
                
                # Create a RequestOutput-like object
                result = RequestOutput(
                    request_id=len(results),
                    prompt=prompt,
                    outputs=[CompletionOutput(
                        index=0,
                        text=response.choices[0].text,
                        token_ids=[],  # Not available from API
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason=response.choices[0].finish_reason
                    )]
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error generating completion for prompt: {e}")
                # Return empty result on error
                result = RequestOutput(
                    request_id=len(results),
                    prompt=prompt,
                    outputs=[CompletionOutput(
                        index=0,
                        text="",
                        token_ids=[],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="error"
                    )]
                )
                results.append(result)
        
        return results
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             sampling_params: Optional[Any] = None,
             tools: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
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
        
        try:
            response = self.client.chat.completions.create(**api_params)
            
            # Extract the response content and tool calls
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Handle tool calls if present
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                # Format tool calls in the expected format for the notebook
                for tool_call in choice.message.tool_calls:
                    tool_call_json = {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    }
                    content += f"\n<tool_call>{json.dumps(tool_call_json)}</tool_call>"
            
            # Create a RequestOutput-like object that matches the notebook's expectations
            result = RequestOutput(
                request_id=0,
                prompt="",  # Not used in chat mode
                outputs=[CompletionOutput(
                    index=0,
                    text=content,
                    token_ids=[],  # Not available from API
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason=choice.finish_reason
                )]
            )
            
            return [result]  # Return as list for compatibility
            
        except Exception as e:
            print(f"Error in chat completion: {e}")
            # Return empty result on error
            result = RequestOutput(
                request_id=0,
                prompt="",
                outputs=[CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="error"
                )]
            )
            return [result]


class RequestOutput:
    """Mock RequestOutput class to maintain compatibility with vLLM's interface."""
    
    def __init__(self, request_id: int, prompt: str, outputs: List[Any]):
        self.request_id = request_id
        self.prompt = prompt
        self.outputs = outputs


class CompletionOutput:
    """Mock CompletionOutput class to maintain compatibility with vLLM's interface."""
    
    def __init__(self, index: int, text: str, token_ids: List[int], 
                 cumulative_logprob: Optional[float], logprobs: Optional[Any], 
                 finish_reason: Optional[str]):
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason


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
    
    # Test with messages and tools (as used in the notebook)
    from vllm.sampling_params import SamplingParams
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
    
    # This should work with your notebook's fork_manager
    outputs = llm.chat(messages, sampling_params=sampling_params)
    print(outputs[0].outputs[0].text)
