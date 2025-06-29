import random
import re
from typing import Callable, Literal

from agents.agent_base import AgentBase
from evals.evaluators.equality_checker_base import EqualityCheckerBase

import evals.evaluators.common as common
from evals.evaluators.types import EvalResult, SingleEvalResult
from evals.evaluators.eval_base import EvalBase

from datasets import load_dataset


class Gsm8kEqualityChecker(EqualityCheckerBase):
    def get_format_instruction(self) -> str:
        return "Put your final answer within \\boxed{}."

    def extract_answer(self, submitted_answer: str) -> str | None:
        match = re.search(common.ANSWER_PATTERN_BOXED, submitted_answer)
        if match:
            return match.group(1).strip()
        return None

    def compare(self, ground_truth: str, submitted_answer: str) -> bool:
        extracted_answer = self.extract_answer(submitted_answer)
        if extracted_answer is not None:
            equal = ground_truth.strip() == extracted_answer.strip()
        else:
            equal = False
        return equal


class Gsm8kEvaluator(EvalBase):
    def __init__(
        self,
        num_examples: int | None = None,
        n_repeats: int = 1,
        split: Literal["test"] = "test",
        max_concurrent_tasks: int = 25,
    ):
        dataset = load_dataset("openai/gsm8k", 'main', split=split)
        examples = [row for row in dataset]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.equality_checker = Gsm8kEqualityChecker()
        self.max_concurrent_tasks = max_concurrent_tasks

    def __call__(self, agent_factory: Callable[[], AgentBase]) -> EvalResult:
        def fn(row: dict):
            question = row["question"]
            content = question + "\n\n" + self.equality_checker.get_format_instruction()
            prompt_messages = [
                dict(role="user", content=content)
            ]
            agent = agent_factory()
            agent_result = agent(prompt_messages)
            response_text = agent_result.response
            ground_truth_answer = row["answer"].split('#### ')[-1]
            score = float(self.equality_checker.compare(ground_truth_answer, response_text))
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=ground_truth_answer,
                extracted_answer=self.equality_checker.extract_answer(response_text) or "None",
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(fn, self.examples, num_threads=self.max_concurrent_tasks)
        return common.aggregate_results(results)
