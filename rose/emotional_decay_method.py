from abc import ABC, abstractmethod
import numpy as np


class EmotionalDecayMethod(ABC):

    @abstractmethod
    def update(
        self,
        robot_state: np.ndarray,
        base_mood: np.ndarray,
        parameters: dict
    ) -> np.ndarray:
        """
        Computes the next robot affective state toward the base mood.

        s_r(t + dt) = f_ED(s_r(t), b, theta_ED)

        Parameters
        ----------
        robot_state : np.ndarray
            Current robot affective state (VAD).

        base_mood : np.ndarray
            Robot base mood (VAD).

        parameters : dict
            Parameters used by the decay method.

        Returns
        -------
        np.ndarray
            Updated robot affective state.
        """
        pass