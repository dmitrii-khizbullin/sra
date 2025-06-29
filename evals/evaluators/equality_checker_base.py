from abc import ABC


class EqualityCheckerBase(ABC):
    """
    Base class for defining an equality checker.
    """
    
    def get_format_instruction(self) -> str:
        """
        Returns the instruction on how to format the answer.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def compare(self, ground_truth: str, submitted_answer: str) -> bool:
        """
        Compares the ground truth answer with the submitted answer.
        """
        raise NotImplementedError("Subclasses should implement this method.")
