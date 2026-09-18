from abc import ABC, abstractmethod
import numpy as np


class EmpathicTransformationMethod(ABC):

    @abstractmethod
    def transform(
        self,
        user_state: np.ndarray,
        robot_state: np.ndarray,
        parameters: dict
    ) -> np.ndarray:
        """
        Computes the transformed robot affective state.

        s'_r = f_ET(s_u, s_r, theta_ET)

        Parameters
        ----------
        user_state : np.ndarray
            User affective state (VAD).

        robot_state : np.ndarray
            Current robot affective state (VAD).

        parameters : dict
            Parameters used by the transformation method.

        Returns
        -------
        np.ndarray
            Transformed robot affective state.
        """
        pass