from abc import ABC, abstractmethod


class EmpathicBehaviorMappingMethod(ABC):
    """
    Abstract interface for AEB empathic behavior mapping methods.
    """

    @abstractmethod
    def map(
        self,
        robot_affective_state: dict,
        parameters: dict | None = None
    ) -> dict:
        """
        Maps the current robot affective state to a structured
        set of empathic behavior parameters.

        Parameters
        ----------
        robot_affective_state : dict
            Robot affective state represented in VAD:
            {
                "valence": float,
                "arousal": float,
                "dominance": float
            }

        parameters : dict | None
            Optional implementation-specific parameters.

        Returns
        -------
        dict
            Structured key/value representation of empathic
            behavior parameters.
        """
        pass