from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Any, cast
import string
import random
import re
import concurrent.futures
from openai.types.chat import ChatCompletionMessageToolCall

from agents.generate_gantt_chart import generate_gantt_chart

from agents.vllm_like import LLM, SamplingParams
from agents.utils import suppress_tqdm


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
                "Example of an MID prefix in message content: [A1B2] Blah blah. The MID itself is A1B2 here. "
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
            "description": ("Finishes the agent or the forked agent operation when the result is ready. "
                            "If the parent LLM thread (or the user) asked you to return the results in "
                            "some specific format, remember to do so."),
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
                        "description": "The message to the caller LLM thread or the user",
                    },
                },
                "required": [],
            },
        },
    },
]


SYSTEM_MESSAGE_TEMPLATE = """
You are a helpful agent that can solve problems. As an agent you can make multiple turns to solve the problem.
You must parallelize your operation as much as possible to minimize the time to the final answer.
Use the tools at your disposal. Do not try using any other tools that are not listed.
If the set of fork, join and status tools are available, you are highly encouraged to use them.
The way fork-join works is it launches a separate LLm conversation (a thread) that will be running
in parallel with your main conversation. Thus, if you want to try multiple ideas or approaches,
or you want to run errands, you can fork yourself, shape the message history as you want with
the message_ids and message parameters, and then join the threads to get the results of the forked conversations.
Before joining the threads that you forked, you check their status with the status tool.
Remember that when you call a fork tool, your message history is not automatically inherited by the forked agent.
You must explicitly provide the forked agent with the list of Message IDs (MIDs) without square brackets that you think are essential to the task you give to it via the message_ids parameter. The task itself can be given in the message parameter.
You can also use the status tool to get the status of all threads launched by this thread. You can thus check if one of the LLM threads you launched is already finished, you can join it and leave the others running.
When you are done with the task, you MUST use the finish tool to return the results of your operation.
Remember that if you do not use the finish tool, nothing will be returned to the caller agent or the user.
ATTENTION: You MUST use the finish tool to return the results of your operation before you run out of turns.
The user will be very disappointed if the answer/response is not returned to them. Make the user happy in this respect.
You MUST use at least one tool in your response.
Do not use thinking too much, rely more on using the tools to solve the problem and delegating subploblems.
You as the main thread or any other thread will have {max_turns} turns to solve the problem.
Every your response must be less than {max_words} words including tool call arguments.
"""


@dataclass
class ThreadResult:
    result: str | None
    start_time: float
    end_time: float
    message_history: list[dict[str, Any]]


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
    if "role" not in message:
        return message
    if "content" not in message:
        return message
    if message["content"] is None:
        return message
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
                 extra_tools: list[dict[str, Any]] | None = None,
                 artifact_dir: str | None = None,
                 ):
        self.llm = llm
        self.artifact_dir = artifact_dir

        self.tools = TOOLS + (extra_tools if extra_tools is not None else [])

        self.max_tokens = 1000

        temperature = 0.5 # 0.0 does not recover from repeats

        self.sampling_params = SamplingParams(max_tokens=self.max_tokens, temperature=temperature)

        self.tools_functions: set[str] = {"fork", "join", "status", "finish"}

        self.thread_pool = concurrent.futures.ThreadPoolExecutor()

        self.thread_records: dict[str, dict[str, Any]] = {}
        self.thread_records_lock = threading.Lock()

        self.tid_length = 3 # Thread ID, chars
        self.tcid_length = 2 # Tool Call ID, chars
        self.mid_length = 4 # Message ID, chars

        self.max_turns = 10

        self.max_forking_level = 2

    def fork(
        self,
        my_tid: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        message: str | None = None,
        message_ids: list[str] | None = None,
    ) -> str:
        # Always add the system message
        if len(messages) > 0:
            first_message = messages[0]
            if first_message["role"] == "system":
                mid = extract_mid(first_message)
                if mid is not None:
                    message_ids = [mid] if message_ids is None else [mid] + message_ids

        selected_messages = []
        if message_ids is not None and len(message_ids) > 0:
            for m in messages:
                mid = extract_mid(m)
                if mid is not None and mid in message_ids:
                    selected_messages.append(m)

        if message is not None:
            user_message = {"role": "user", "content": message}
            user_message = tag_message(user_message, self.mid_length)
            selected_messages.append(user_message)

        with self.thread_records_lock:
            my_level = self.thread_records[my_tid]['level']
        if my_level >= self.max_forking_level-1:
            tools = remove_forking_tools(tools)

        future_tid = self.submit_task(my_tid, selected_messages, tools)

        return f"Forked TID: {future_tid}"

    def join(self, my_tid: str, tids: list[str]) -> str:
        results: dict[str, str] = {}
        if my_tid not in self.thread_records:
            raise ValueError(f"Thread {my_tid} does not exist.")
        my_record = self.thread_records[my_tid]
        for tid in tids:
            if tid in my_record['child_tids']:
                future: concurrent.futures.Future
                future = self.thread_records[tid]['future']
                if future is not None:
                    result: ThreadResult = future.result()
                    with self.thread_records_lock:
                        self.thread_records[tid]['start_time'] = result.start_time
                        self.thread_records[tid]['end_time'] = result.end_time
                        self.thread_records[tid]['join_time'] = time.time()
                    results[tid] = f"Thread {tid} result:\n\n {result.result}"
                else:
                    raise ValueError(f"Thread {tid} has no future associated with it.")
            else:
                results[tid] = f"Thread {tid} does not exist or is not owned by this thread. Can't join it."
        results_str = "Successfully joined threads.\n\n"
        for tid, message in results.items():
            results_str += f"Thread {tid}\n"
            results_str += f"{message}\n\n"
        return results_str

    def status(self, my_tid: str) -> str:
        status_lines = []
        with self.thread_records_lock:
            for tid, record in self.thread_records.items():
                parent_tid = record['parent_tid']
                if parent_tid != my_tid:
                    continue  # Only show threads that are children of the current thread
                future: concurrent.futures.Future = record['future']
                if future is not None:
                    if future.done():
                        status = "Finished"
                        future_result = future.result()
                        result_str = future_result.result
                        if result_str is not None:
                            status = "Finished"
                            if isinstance(result_str, str):
                                result = f"Text of length {len(result_str)} characters."
                            else:
                                raise ValueError(f"Unexpected result type: {type(future_result)}")
                        else:
                            result = "The result is None."
                    else:
                        status = "Running"
                        result = "Not yet completed"
                    status = "Running" if not future.done() else "Finished"
                    result = f"Text of length {len(future.result())} characters." if future.done() else "Not yet completed"
                else:
                    status = "Main thread"
                    result = "No result available"
                status_lines.append(
                    f"TID: {tid}, Status: {status}, Result: {result}"
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
            for m in messages:
                mid = extract_mid(m)
                if mid is not None and mid in message_ids:
                    selected_messages.append(m)

        result_str = ""
        for m in selected_messages:
            if m["role"] in ["system", "user", "assistant"]:
                result_str += f"{m['role'].upper()}: {m['content']}\n"
            elif m["role"] == "tool":
                tool_call_id = m.get("tool_call_id", "unknown")
                result_str += f"Tool Call [{tool_call_id}]: {m['content']}\n"
        
        if message is not None:
            result_str += f"Message: {message}\n"
        
        return result_str

    def submit_task(self,
                    parent_tid: str,
                    messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]],
                    ) -> str:
        new_tid = generate_random_id(self.tid_length)
        with self.thread_records_lock:
            future = self.thread_pool.submit(
                self.run_agent, new_tid, messages, tools
            )
            self.thread_records[parent_tid]['child_tids'].append(new_tid)
            self.thread_records[new_tid] = {
                "future": future,
                "parent_tid": parent_tid,
                "child_tids": [],
                "level": self.thread_records[parent_tid]['level'] + 1,
                "fork_time": time.time(),
                "start_time": None,  # Will be set when the future is started
                "end_time": None,  # Will be set when the future is done
                "join_time": None,  # Will be set when the future is joined
            }
        return new_tid

    def call_tool(self,
                  my_tid: str,
                  tool_call: dict[str, Any],
                  messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]],
                  ) -> str:

        func = tool_call["function"]
        tool_name = func["name"]
        tool_args = func["arguments"]
        if tool_name in self.tools_functions:
            enabled_tools = {tool["function"]["name"] for tool in tools}
            if tool_name in enabled_tools:
                print(f"MyTID={my_tid}, Tool call: {tool_name}, args: {tool_args}")
                if tool_name == 'fork':
                    messages_forking = messages.copy()
                    messages_forking.append(tool_call)
                    unexpected_args = set(tool_args.keys()) - {"message_ids", "message"}
                    no_unexpected_args = len(unexpected_args) == 0
                    is_message_good = (("message" not in tool_args) or
                                    isinstance(tool_args["message"], str))
                    is_message_ids_good = (("message_ids" not in tool_args) or
                                        (isinstance(tool_args["message_ids"], list) and
                                            all(isinstance(x, str) for x in tool_args["message_ids"])))
                    if no_unexpected_args and is_message_good and is_message_ids_good:
                        tool_answer = self.fork(
                            my_tid=my_tid,
                            messages=messages_forking,
                            tools=tools,
                            **tool_args)
                    else:
                        tool_answer = "Invalid arguments for fork."
                elif tool_name == 'join':
                    unexpected_args = set(tool_args.keys()) - {"tids"}
                    no_unexpected_args = len(unexpected_args) == 0
                    is_tids_good = "tids" in tool_args and (
                        isinstance(tool_args["tids"], list) and
                        all(isinstance(x, str) for x in tool_args["tids"])
                        )
                    if no_unexpected_args and is_tids_good:
                        tool_answer = self.join(my_tid, **tool_args)
                    else:
                        tool_answer = "Invalid arguments for join."
                elif tool_name == 'status':
                    tool_answer = self.status(my_tid)
                elif tool_name == 'finish':
                    tool_answer = self.finish(messages=messages, **tool_args)
                else:
                    raise ValueError(f"Unknown tool name: {tool_name}")
            else:
                tool_answer = f"Tool {tool_name} is disabled."
        else:
            tool_answer = f"Tool {tool_name} does not exist."
        return tool_answer

    @staticmethod
    def tool_call_to_dict(tool_call: ChatCompletionMessageToolCall) -> dict[str, Any]:
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            },
        }

    @staticmethod
    def encode_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
        # Encode the tool call to a format suitable for the LLM
        return {
            "id": tool_call["id"],
            "type": tool_call["type"],
            "function": {
                "name": tool_call["function"]["name"],
                "arguments": json.dumps(tool_call["function"]["arguments"]),
            },
        }
        
    def run_agent(self,
                  my_tid: str,
                  messages: list[dict[str, Any]],
                  tools: list[dict[str, Any]],
                  ) -> ThreadResult:

        start_time = time.time()

        chat_template_kwargs = {
            "enable_thinking": False,
        }

        result_str: str | None = None
        for i in range(self.max_turns):
            print(f"MyTID={my_tid}, Turn {i+1}/{self.max_turns}")

            with suppress_tqdm():
                outputs = self.llm.chat(
                    messages,
                    sampling_params=self.sampling_params,
                    tools=tools,
                    )
            usage = outputs.usage
            message = outputs.choices[0].message
            tool_calls_api = message.tool_calls
            output = message.content if message.content is not None else ""
            content = remove_tool_calls(output)
            tool_calls_extracted = extract_tool_calls(output)
            tool_calls = [self.tool_call_to_dict(tc) for tc in tool_calls_api] \
                if tool_calls_api is not None else tool_calls_extracted
            print(f"MyTID={my_tid}, Assistant: {content}, Tool calls: {tool_calls}")
            tool_answers = []
            is_finish = False
            new_message = {"role": "assistant",
                           "content": content,
                           "tool_calls": [self.encode_tool_call(tc) for tc in tool_calls]}
            messages.append(tag_message(new_message, mid_length=self.mid_length))
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
                tool_answer = self.call_tool(
                    my_tid=my_tid,
                    tool_call=tool_call,
                    messages=messages,
                    tools=tools,
                )
                tool_answers.append(tool_answer)
            for tool_call, tool_answer in zip(tool_calls, tool_answers):
                tool_answer_message = {
                    "role": "tool",
                    "content": tool_answer,
                    "tool_call_id": tool_call['id'],
                }
                messages.append(tool_answer_message)

        end_time = time.time()

        result = ThreadResult(
            result=result_str,
            start_time=start_time,
            end_time=end_time,
            message_history=messages
        )
        return result

    def join_all_threads(self) -> None:
        with self.thread_records_lock:
            unfinished_tids = set(self.thread_records.keys())
        while len(unfinished_tids) > 0:
            print(f"Waiting for {unfinished_tids} unfinished threads...")
            with self.thread_records_lock:
                for tid in list(unfinished_tids):
                    record = self.thread_records[tid]
                    if record['future'] is not None:
                        future: concurrent.futures.Future = record['future']
                        try:
                            result = future.result(timeout=0.001)
                            record['start_time'] = result.start_time
                            record['end_time'] = result.end_time
                            print(f"Joining thread {tid} with parent {record['parent_tid']}, level {record['level']}")
                            unfinished_tids.remove(tid)
                        except concurrent.futures.TimeoutError:
                            continue
                    else:
                        print(f"Thread {tid} has no future associated with it. Removing it from unfinished threads.")
                        unfinished_tids.remove(tid)
            time.sleep(1.0)

    def get_thread_records(self) -> dict[str, dict[str, Any]]:
        with self.thread_records_lock:
            records_json = {
                tid: {
                    "parent_tid": record["parent_tid"],
                    "child_tids": record["child_tids"],
                    "level": record["level"],
                    "fork_time": record["fork_time"],
                    "start_time": record["start_time"],
                    "end_time": record["end_time"],
                    "join_time": record["join_time"],
                }
                for tid, record in self.thread_records.items()
            }
        return records_json

    def save_thread_records(self, filename_wo_ext: str) -> None:
        if self.artifact_dir is not None:
            os.makedirs(self.artifact_dir, exist_ok=True)
            records_json = self.get_thread_records()
            json_path = f"{self.artifact_dir}/{filename_wo_ext}.json"
            print(f"Saving thread records to {json_path}")
            with open(json_path, 'w') as f:
                json.dump(records_json, f, indent=4)

            gantt = generate_gantt_chart(records_json)

            svg_path = f"{self.artifact_dir}/{filename_wo_ext}.svg"
            gantt.save_svg(svg_path)
            print(f"Gantt chart saved as {svg_path}")

    def run_entry(self, task: str) -> str | None:
        safety_gap = 2.0
        max_words = self.max_tokens // 2 // safety_gap # Rough estimate of words based on tokens

        system_message = SYSTEM_MESSAGE_TEMPLATE.format(
            max_turns=self.max_turns,
            max_words=max_words)

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": task,
            },
        ]

        messages = tag_messages(messages, self.mid_length)
        new_tid = generate_random_id(self.tid_length)
        self.thread_records[new_tid] = {
            "future": None,
            "parent_tid": None,
            "child_tids": [],
            "level": 0,
            "fork_time": time.time(),
            "start_time": None,  # Will be set when the future is started
            "end_time": None,  # Will be set when the future is done
            "join_time": None,  # Will be set when the future is joined
        }
        result = self.run_agent(new_tid, messages, self.tools) # start with all tools

        with self.thread_records_lock:
            self.thread_records[new_tid]['start_time'] = result.start_time
            self.thread_records[new_tid]['end_time'] = result.end_time
            self.thread_records[new_tid]['join_time'] = time.time()

        self.join_all_threads()

        name_with_datetime = "graph_" + time.strftime("%Y%m%d_%H%M%S")
        self.save_thread_records(name_with_datetime)

        if self.artifact_dir is not None:
            # save message history to a file
            messages_path = f"{self.artifact_dir}/messages.json"
            with open(messages_path, 'w') as f:
                json.dump(messages, f, indent=4)

        result_str = result.result
        return result_str


def main():
    llm = LLM()
    
    problem = """
    You're a contestant on a game show. The host, Monty Hall, presents you with three doors. Behind one door is a valuable prize (like a car), and behind the other two doors are less desirable prizes (like goats).
    The Setup:

    You choose one door (say, Door #1)
    Monty, who knows what's behind each door, opens one of the remaining doors that contains a goat (say, Door #3)
    Monty then asks: "Do you want to stick with your original choice (Door #1) or switch to the remaining unopened door (Door #2)?"

    The Question: What should you do to maximize your chances of winning the car?"""

    sampling_params = SamplingParams(max_tokens=1000, temperature=0.0)
    response = llm.chat([dict(role='user', content=problem)], sampling_params=sampling_params)
    print(response.choices[0].message.content)

    # ===============================================

    task = f"Solve the following problem with 5 different methods:\n\n {problem}"
    fork_manager = ForkManager(llm, artifact_dir=".")
    response = fork_manager.run_entry(task)
    print(response)


if __name__ == "__main__":
    main()
