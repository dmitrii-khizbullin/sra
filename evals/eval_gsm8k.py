import os

from datetime import datetime
from typing import Type, cast
from agents.agent_base import AgentBase
from evals.evaluators.gsm8k import Gsm8kEvaluator
from agents.baseline_agent import BaselineAgent
from agents.self_replicating_agent import SelfReplicatingAgent


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate agents on the GSM8K dataset.")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to evaluate on. If None, use all examples.")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="Number of times to repeat each example.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Dataset split to use for evaluation.")
    parser.add_argument("--agent_names", nargs='*', default=None,
                        help="Names of agents to evaluate. Provide a space-separated list, "
                        "e.g., --agent_names BaselineAgent SelfReplicatingAgent. "
                        "If None, evaluate all available agents.")
    parser.add_argument("--max_concurrent_tasks", type=int, default=None,
                        help="Maximum concurrency for dataset tasks. If None, use default concurrency.")
    args = parser.parse_args()

    available_agent_classes: list[Type[AgentBase]] = [
        cast(Type[AgentBase], a)
        for a in (BaselineAgent, SelfReplicatingAgent)]

    agent_class_dict: dict[str, Type[AgentBase]] = {
        ac.__name__: ac for ac in available_agent_classes
    }
    agent_names = args.agent_names
    agent_classes = [agent_class_dict[name] for name in agent_names] if agent_names else available_agent_classes

    extra_kwargs = dict(max_concurrent_tasks=args.max_concurrent_tasks) if args.max_concurrent_tasks is not None else {}

    evaluator = Gsm8kEvaluator(num_examples=args.num_examples, n_repeats=args.n_repeats, split=args.split, **extra_kwargs)

    runs_folder = "runs"
    exp_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_name = "gsm8k"
    exp_folder = f"{runs_folder}/{exp_datetime}/{dataset_name}"

    for agent_class in agent_classes:
        agent_name = agent_class.__name__
        result_folder = f"{exp_folder}/{agent_name}"
        os.makedirs(result_folder, exist_ok=True)

        print(f"Evaluating {agent_class.__name__} on GSM8K dataset...")
        # no extra tools for GSM8K
        result = evaluator(agent_factory=lambda sample_id: agent_class(
            extra_tools=None, artifact_dir=os.path.join(result_folder, sample_id)))
        print(f"Results for {agent_class.__name__}:")
        print(result)     

        with open(f"{result_folder}/result.txt", "w") as f:
            f.write(f"Score: {result.score}\n")
            f.write(f"Metrics: {result.metrics}\n")
            f.write(f"Number of conversations: {len(result.convos)}\n")
            for i, convo in enumerate(result.convos):
                f.write(f"\nConversation {i}:\n")
                for message in convo:
                    f.write(f"{message['role']}: {message['content']}\n")
        with open(f"{result_folder}/result.html", "w") as f:
            f.write("<html><body>\n")
            for html in result.htmls:
                f.write(html + "<br>\n")
            f.write("</body></html>\n")

    print("Done")

    
if __name__ == "__main__":
    main()
