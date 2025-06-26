import json
from typing import Any
import string
import random
import re
import concurrent.futures

from vllm_like import LLM, SamplingParams
from utils import suppress_tqdm


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fork",
            "description": (
                "Fork a conversation with the provided messages and makes it run asynchronously. "
                "The forked agent will not automatically inherit your message history. "
                "Instead, you must explicitly provide the forked agent with the messages "
                "that you think are essential to the task you give to it. "
                "First, form message_ids as a list if Message IDs (MIDs). An MID is a alphanumerical tag "
                "in square brackets that is prepended to every system, user and assistant message. "
                "Example of an MID: [A1B2]. "
                "Second, pass your message/task/query to the forked agent in the message field. "
                "The tool returns a Thread ID (TID) to be later joined."
                "Do not forget to join the threads with the TIDs you own before exiting, "
                "to get the information you queried from them or the results of the task given. "
                ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_ids": {
                        "type": "array",
                        "description": (
                            "A list of message IDs to reference previous messages. "
                            "The selected messages will be included into the forked conversation."),
                        "items": {
                            "type": "string"
                        }
                    },
                    "message": {
                        "type": "string",
                        "description": "The message or an instruction to the forked agent",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "join",
            "description": "Join the conversation with the provided thread ID (TID)",
            "parameters": {
                "type": "object",
                "properties": {
                    "tids": {
                        "type": "array",
                        "description": (
                            "A list of thread IDs (TIDs) to join. "
                            "The agent will wait for the results of the forked conversations."),
                    },
                },
                "required": ["tids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "status",
            "description": "Gets the status of all threads launched by this thread",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finishes the agent or the forked agent operation when the result is ready.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_ids": {
                        "type": "array",
                        "description": (
                            "A list of message IDs to reference previous messages. "
                            "The selected messages will be included into the response."),
                        "items": {
                            "type": "string"
                        }
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to the caller agent",
                    },
                },
                "required": [],
            },
        },
    },
]


SYSTEM_MESSAGE = """
You are a helpful assistant that can solve problems.
You must parallelize your operation as much as possible to minimize the time to the final answer.
Use the tools at your disposal. Do not try using any other tools that are not listed.
If the set of fork, join and status tools are available, you are highly encouraged to use them.
The way fork-join works is it launches a separate LLm conversation (a thread) that will be running
in parallel with your main conversation. Thus, if you want to try multiple ideas or approaches,
or you want to run errands, you can fork yourself, shape the message history as you want with
the message_ids and message parameters, and then join the threads to get the results of the forked conversations.
Remember that when you call a fork tool, your message history is not automatically inherited by the forked agent.
You must explicitly provide the forked agent with the list of Message IDs (MIDs) that you think are essential to the task you give to it via the message_ids parameter. The task itself can be given in the message parameter.
You can also use the status tool to get the status of all threads launched by this thread. You can thus check if one of the LLM threads you launched is already finished, you can join it and leave the others running.
When you are done with the task, you can use the finish tool to return the results of your operation.
"""


def generate_random_id(length: int) -> str:
    characters = string.ascii_letters + string.digits
    random_id = "".join(random.choice(characters) for _ in range(length))
    return random_id


def extract_tool_calls(response_text: str) -> list[dict[str, Any]]:
    # Look for tool call patterns in the response
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
    
    tool_calls = []
    for match in matches:
        try:
            tool_data = json.loads(match.strip())
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": tool_data,
            })
        except:
            continue
    
    return tool_calls


def remove_tool_calls(response_text: str) -> str:
    # Remove tool call patterns from the response
    tool_call_pattern = r'<tool_call>.*?</tool_call>'
    cleaned_text = re.sub(tool_call_pattern, '', response_text, flags=re.DOTALL)
    return cleaned_text.strip()


def tag_message(message: dict[str, Any], mid_length: int) -> dict[str, Any]:
    if message["role"] in ["system", "user", "assistant"]:
        message = message.copy()
        random_id = generate_random_id(mid_length)
        message["content"] = f"[{random_id}] {message['content']}"
    return message


def tag_messages(messages: list[dict[str, Any]], mid_length: int) -> list[dict[str, Any]]:
    # for every system, user and assistant message, prepend the content with [random_id]. do not alter messages as a variable
    out_messages = []
    for message in messages:
        message = tag_message(message, mid_length)
        out_messages.append(message)
    return out_messages


def extract_mid(message: dict[str, Any]) -> str | None:
    # Extract the Message ID (MID) from the message content
    if "content" in message:
        match = re.search(r'^\[([A-Za-z0-9]+)\]', message["content"])
        if match:
            return match.group(1)
    return None


def remove_forking_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool for tool in tools
            if tool["function"]["name"] not in {"fork", "join", "status"}]


class ForkManager:
    def __init__(self, 
                 llm: LLM,
                 tools: list[dict[str, Any]],
                 ):
        self.llm = llm
        # self.tools = tools
        self.tools = remove_forking_tools(tools) # TEMPORARY

        self.sampling_params = SamplingParams(max_tokens=1000, temperature=0.0)

        self.tools_functions = {
            "fork": self.fork,
            "join": self.join,
            "status": self.status,
            "finish": self.finish,
        }

        self.thread_pool = concurrent.futures.ThreadPoolExecutor()

        self.thread_records: dict[str, dict[str, Any]] = {}

        self.tid_length = 3 # Thread ID, chars
        self.tcid_length = 2 # Tool Call ID, chars
        self.mid_length = 4 # Message ID, chars

        self.max_turns = 10

        self.max_forking_level = 1

    def fork(
        self,
        my_tid: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        message: str | None = None,
        message_ids: list[str] | None = None,
    ) -> str:
        selected_messages = []
        if len(message_ids) > 0:
            for message in messages:
                mid = extract_mid(message)
                if mid is not None and mid in message_ids:
                    selected_messages.append(message)
        if message is not None:
            selected_messages.append({"role": "user", "content": message})
        if self.thread_records[my_tid]['level'] >= self.max_forking_level:
            tools = remove_forking_tools(tools)
        future_tid = self.submit_task(my_tid, selected_messages, tools)
        return f"Forked TID: {future_tid}"

    def join(self, tids: list[str]) -> str:
        results: list[str] = []
        for tid in tids:
            if tid in self.thread_records:
                future: concurrent.futures.Future
                future = self.thread_records[tid]['future']
                if future is not None:
                    result = future.result()
                    results.append(f"Thread {tid} result:\n\n {result}")
                else:
                    raise ValueError(f"Thread {tid} has no future associated with it.")
            else:
                raise ValueError(f"Thread {tid} does not exist.")
        results_str = "Successfully joined threads.\n\n"
        for result in results:
            results_str += f"Thread {result}\n"
            results_str += f"{result}\n\n"
        return results_str

    def status(self) -> str:
        status_lines = []
        for tid, record in self.thread_records.items():
            parent_tid = record['parent_tid']
            child_tids = record['child_tids']
            future: concurrent.futures.Future = record['future']
            if future is not None:
                status = "Running" if not future.done() else "Finished"
                result = future.result() if future.done() else "Not yet completed"
            else:
                status = "Not started"
                result = "No result available"
            status_lines.append(
                f"Thread ID: {tid}, Parent TID: {parent_tid}, "
                f"Child TIDs: {', '.join(child_tids) if child_tids else 'None'}, "
                f"Status: {status}, Result: {result}"
            )
        if not status_lines:
            return "No threads have been created yet."
        return "\n".join(status_lines)

    def finish(self,
               messages: list[dict[str, Any]],
               message: str | None = None,
               message_ids: list[str] | None = None,
               ) -> str:

        selected_messages = []
        if message_ids is not None and len(message_ids) > 0:
            for message in messages:
                mid = extract_mid(message)
                if mid is not None and mid in message_ids:
                    selected_messages.append(message)

        result_str = ""
        for message in selected_messages:
            if message["role"] in ["system", "user", "assistant"]:
                result_str += f"{message['role'].capitalize()}: {message['content']}\n"
            elif message["role"] == "tool":
                tool_call_id = message.get("tool_call_id", "unknown")
                result_str += f"Tool Call [{tool_call_id}]: {message['content']}\n"
        
        if message is not None:
            result_str += f"Message: {message}\n"
        
        return result_str

    def submit_task(self,
                    parent_tid: str,
                    messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]],
                    ) -> str:
        new_tid = generate_random_id(self.tid_length)
        future = self.thread_pool.submit(
            self.run_agent, new_tid, messages, tools
        )
        self.thread_records[parent_tid]['child_tids'].append(new_tid)
        self.thread_records[new_tid] = {
            "future": future,
            "parent_tid": parent_tid,
            "child_tids": [],
            "level": self.thread_records[parent_tid]['level'] + 1,
        }
        return new_tid

    def run_agent(self,
                  my_tid: str,
                  messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]],
                  ) -> str | None:

        for i in range(self.max_turns):
            print(f"MyTID={my_tid}, Turn {i+1}/{self.max_turns}")

            with suppress_tqdm():
                outputs = self.llm.chat(
                    messages,
                    sampling_params=self.sampling_params,
                    tools=tools,
                    )
            output = outputs[0].outputs[0].text.strip()
            content = remove_tool_calls(output)
            messages.append({"role": "assistant", "content": content})
            tool_calls = extract_tool_calls(output)
            print(f"MyTID={my_tid}, Assistant: {content}, Tool calls: {tool_calls}")
            tool_answers = []
            is_finish = False
            result_str: str | None = None
            for tool_call in tool_calls:
                func = tool_call["function"]
                tool_name = func["name"]
                if tool_name == "finish":
                    is_finish = True
                    tool_args = func["arguments"]
                    unexpected_args = set(tool_args.keys()) - {"message_ids", "message"}
                    if len(unexpected_args) == 0:
                        result_str = self.finish(messages=messages, **tool_args)
                    else:
                        result_str = "Invalid arguments for finish. Returning no result."
                    break
            if is_finish:
                break

            for tool_call in tool_calls:
                if is_finish and tool_call["function"]["name"] != "finish":
                    continue
                if 'role' not in tool_call:
                    tool_call['role'] = 'tool'
                if 'id' not in tool_call:
                    tool_call['id'] = generate_random_id(self.tcid_length) # patch
                func = tool_call["function"]
                tool_name = func["name"]
                tool_args = func["arguments"]
                if tool_name in self.tools_functions:
                    tool_fn = self.tools_functions[tool_name]
                    print(f"MyTID={my_tid}, Tool call: {tool_name}, args: {tool_args}")
                    if tool_name == 'fork':
                        messages_forking = messages.copy()
                        messages_forking.append(tool_call)
                        unexpected_args = set(tool_args.keys()) - {"message_ids", "message"}
                        if len(unexpected_args) == 0:
                            tool_answer = self.fork(
                                my_tid=my_tid,
                                messages=messages_forking,
                                tools=tools,
                                **tool_args)
                        else:
                            tool_answer = "Invalid arguments for fork."
                    else:
                        tool_answer = tool_fn(**tool_args)
                else:
                    tool_answer = f"Tool {tool_name} does not exist."
                tool_answers.append(tool_answer)
            for tool_call, tool_answer in zip(tool_calls, tool_answers):
                tool_answer_message = {
                    "role": "tool",
                    "content": tool_answer,
                    "tool_call_id": tool_call['id'],
                }
                messages.append(tool_answer_message)
        return result_str

    def run_entry(self, messages):
        messages = tag_messages(messages, self.mid_length)
        new_tid = generate_random_id(self.tid_length)
        self.thread_records[new_tid] = {
            "future": None,
            "parent_tid": None,
            "child_tids": [],
            "level": 0,
        }
        whatever = self.run_agent(new_tid, messages, self.tools) # start with all tools

        for tid, record in self.thread_records.items():
            print(f"Joining thread {tid} with parent {record['parent_tid']}, level {record['level']}")
            if record['future'] is not None:
                record['future'].result()
        return whatever


if __name__ == "__main__":
    llm = LLM()
    
    sampling_params = SamplingParams(max_tokens=1000, temperature=0.0)
    prompt = "What is the capital of France?"
    response = llm.generate(prompt, sampling_params=sampling_params)    
    print(response)

    # ===============================================

    problem = """
    You're a contestant on a game show. The host, Monty Hall, presents you with three doors. Behind one door is a valuable prize (like a car), and behind the other two doors are less desirable prizes (like goats).
    The Setup:

    You choose one door (say, Door #1)
    Monty, who knows what's behind each door, opens one of the remaining doors that contains a goat (say, Door #3)
    Monty then asks: "Do you want to stick with your original choice (Door #1) or switch to the remaining unopened door (Door #2)?"

    The Question: What should you do to maximize your chances of winning the car?"""

    task = f"Solve the following problem with two different methods:\n\n {problem}"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": task,
        },
    ]

    fork_manager = ForkManager(llm, TOOLS)
    response = fork_manager.run_entry(messages)
    print(response)
