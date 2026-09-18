from typing import Any

from cat_to_vad_method import CatToVadMethod


class ReferenceBasedCatToVadMethod(CatToVadMethod):
    """
    Converts categorical affective observations to VAD through
    probability-weighted reference vectors.
    """

    VAD_REFERENCE: dict[str, tuple[float, float, float]] = {
        "happiness": (0.76, 0.48, 0.35),
        "sadness": (-0.63, -0.27, -0.33),
        "anger": (-0.43, 0.67, 0.34),
        "fear": (-0.64, 0.60, -0.43),
        "surprise": (0.40, 0.67, -0.13),
        "disgust": (-0.60, 0.35, 0.11),
        "neutral": (0.00, 0.00, 0.00),
    }

    def convert(
        self,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(observations, list):
            raise TypeError("observations must be a list.")

        for observation in observations:
            if observation.get("representation") != "CAT":
                continue

            probabilities = observation.get("probabilities")

            if not isinstance(probabilities, dict):
                raise TypeError(
                    "A CAT observation must contain a probability dictionary."
                )

            if not probabilities:
                raise ValueError(
                    "A CAT observation cannot contain an empty distribution."
                )

            valence = 0.0
            arousal = 0.0
            dominance = 0.0

            for emotion, probability in probabilities.items():
                if emotion not in self.VAD_REFERENCE:
                    raise ValueError(
                        f"Unknown categorical emotion: {emotion}"
                    )

                probability_value = float(probability)

                if not 0.0 <= probability_value <= 1.0:
                    raise ValueError(
                        f"Probability for '{emotion}' must be in [0, 1]."
                    )

                reference_valence, reference_arousal, reference_dominance = (
                    self.VAD_REFERENCE[emotion]
                )

                valence += probability_value * reference_valence
                arousal += probability_value * reference_arousal
                dominance += probability_value * reference_dominance

            observation["representation"] = "VAD"

            observation["values"] = {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
            }

            observation.pop("probabilities", None)

        return observations