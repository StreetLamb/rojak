from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Retriever(ABC):
    type: str
    """The prefix of the activity name."""


class RetrieverActivities(ABC):
    """
    Abstract base class for Retriever implementations.
    This class provides a common structure for different types of retrievers.
    """

    def __init__(self, options: any):
        self._options = options
        pass

    @abstractmethod
    async def retrieve(self, text: str) -> any:
        """Retrieve information based on the input text.

        This abstract method must be implemented by all concrete subclasses. It handles the retrieval of
        information relevant to the provided text.

        Args:
            text (str): The input text to base retrieval on.

        Returns:
            Any: The retrieved information corresponding to the input text.
        """
        pass

    @abstractmethod
    async def retrieve_and_combine_results(self, text: str) -> str:
        """Retrieve information and combine the results.

        This abstract method must be implemented by all concrete subclasses. It is responsible for performing
        data retrieval based on the input text and combining or processing the results into a final output.

        Args:
            text (str): The input text text to guide the retrieval process.

        Returns:
            str: The processed and combined results of the retrieval operation.
        """
        pass
