import math

from empathic_behavior_mapping_method import EmpathicBehaviorMappingMethod


class ReferencePointBehaviorMapping(EmpathicBehaviorMappingMethod):
    """
    Empathic behavior mapping based on reference points
    in the VAD space.
    """

    def map(
        self,
        robot_affective_state: dict,
        parameters: dict | None = None
    ) -> dict:

        if parameters is None:
            parameters = {}

        reference_points = parameters.get("reference_points")

        if reference_points is None:
            raise ValueError(
                "ReferencePointBehaviorMapping requires "
                "'reference_points' in parameters."
            )

        nearest_point = self._find_nearest_reference_point(
            robot_affective_state,
            reference_points
        )

        return self._create_affective_behavior(
            nearest_point["point"]
        )

    # ---------------------------------------------------------
    # INTERNAL METHODS
    # ---------------------------------------------------------

    def _euclidean_distance(
        self,
        point_a: dict,
        point_b: dict
    ) -> float:

        return math.sqrt(
            sum(
                (
                    float(point_a[dimension])
                    - float(point_b[dimension])
                ) ** 2
                for dimension in (
                    "valence",
                    "arousal",
                    "dominance"
                )
            )
        )

    def _find_nearest_reference_point(
        self,
        vad_point: dict,
        reference_points: dict
    ) -> dict:

        distances = {
            name: self._euclidean_distance(
                vad_point,
                reference_point
            )
            for name, reference_point
            in reference_points.items()
        }

        nearest_name = min(
            distances,
            key=distances.get
        )

        return {
            "point": nearest_name,
            "distance": distances[nearest_name],
            "all_distances": distances
        }

    def _create_affective_behavior(
        self,
        affective_state: str
    ) -> dict:

        valid_emotions = {
            "happiness",
            "anger",
            "sadness",
            "neutral"
        }

        valid_levels = {
            "l1",
            "l2",
            "l3",
            "l4"
        }

        if not isinstance(affective_state, str):
            raise TypeError(
                "The affective state must be a string."
            )

        affective_state = affective_state.strip().lower()

        # Neutral state
        if affective_state == "neutral":
            return {
                "affective_state": "neutral",
                "facial_expression": "neutral",
                "leds": "neutral",
                "pose": "neutral"
            }

        try:
            emotion, level = affective_state.rsplit("_", 1)

        except ValueError as error:
            raise ValueError(
                "Invalid affective state format."
            ) from error

        if emotion not in valid_emotions:
            raise ValueError(
                f"Invalid emotion: '{emotion}'."
            )

        if level not in valid_levels:
            raise ValueError(
                f"Invalid level: '{level}'."
            )

        return {
            "affective_state": affective_state,
            "facial_expression": level,
            "leds": level,
            "pose": level
        }