from abc import ABC, abstractmethod
from typing import Any


class CatToVadMethod(ABC):
    """
    Defines the contract for categorical-to-VAD conversion methods.
    """

    @abstractmethod
    def convert(
        self,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Converts CAT observations to VAD.

        Observations already represented in VAD must remain unchanged.
        """
        raise NotImplementedError