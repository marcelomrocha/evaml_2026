from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from paho.mqtt import client as mqtt_client


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

ROBOT_PACKAGE_DIR = PARENT_DIR / "robot_package"

if str(ROBOT_PACKAGE_DIR) not in sys.path:
    sys.path.append(str(ROBOT_PACKAGE_DIR))


import config
import robot_profile

from confidence_method import ConfidenceMethod
from entropy_confidence_method import EntropyConfidenceMethod

from cat_to_vad_method import CatToVadMethod
from reference_based_cat_to_vad_method import (
    ReferenceBasedCatToVadMethod,
)

from fusion_method import FusionMethod
from weighted_fusion_method import WeightedFusionMethod


AffectiveObservation = dict[str, Any]
VADState = dict[str, float]


# ============================================================
# PULSE
# ============================================================

class Pulse:
    """
    Coordinates perception windows and estimates the user's
    affective state.

    Confidence estimation, CAT-to-VAD conversion, and multimodal
    fusion are supplied through constructor-based dependency
    injection.
    """

    def __init__(
        self,
        confidence_method: ConfidenceMethod,
        cat_to_vad_method: CatToVadMethod,
        fusion_method: FusionMethod,
    ) -> None:
        if not isinstance(confidence_method, ConfidenceMethod):
            raise TypeError(
                "confidence_method must implement ConfidenceMethod."
            )

        if not isinstance(cat_to_vad_method, CatToVadMethod):
            raise TypeError(
                "cat_to_vad_method must implement CatToVadMethod."
            )

        if not isinstance(fusion_method, FusionMethod):
            raise TypeError(
                "fusion_method must implement FusionMethod."
            )

        self._confidence_method = confidence_method
        self._cat_to_vad_method = cat_to_vad_method
        self._fusion_method = fusion_method

        self._sources_to_receive: list[str] = []
        self._affective_observations: list[AffectiveObservation] = []

    @property
    def pending_sources(self) -> tuple[str, ...]:
        return tuple(self._sources_to_receive)

    @property
    def observations(self) -> tuple[AffectiveObservation, ...]:
        return tuple(self._affective_observations)

    def start_perception_window(
        self,
        sources: list[str],
    ) -> None:
        """
        Starts a new perception window and clears data from
        the preceding window.
        """
        if not isinstance(sources, list):
            raise TypeError("sources must be a list.")

        normalized_sources = [
            str(source).strip()
            for source in sources
            if str(source).strip()
        ]

        if not normalized_sources:
            raise ValueError(
                "The perception window must contain at least one source."
            )

        if len(normalized_sources) != len(set(normalized_sources)):
            raise ValueError(
                "The perception source list contains duplicated values."
            )

        self._sources_to_receive = normalized_sources.copy()
        self._affective_observations.clear()

        print(
            "Sources expected in the perception window:",
            self._sources_to_receive,
        )

    def receive_observation(
        self,
        observation: AffectiveObservation,
    ) -> VADState | None:
        """
        Stores one affective observation.

        Processing starts only after all expected sources have
        provided their observations.
        """
        self._validate_observation(observation)

        source = observation["source"]

        if source not in self._sources_to_receive:
            print(
                f"Observation from '{source}' ignored. "
                "The source is not pending in the current window."
            )
            return None

        self._sources_to_receive.remove(source)

        # A shallow copy prevents direct modification of the
        # dictionary received by the caller.
        self._affective_observations.append(
            observation.copy()
        )

        print(f"Source '{source}' received.")

        if self._sources_to_receive:
            print(
                "Waiting for sources:",
                self._sources_to_receive,
            )
            return None

        return self._process_observations()

    def _process_observations(self) -> VADState:
        """
        Executes the PULSE affective-state estimation pipeline.
        """
        if not self._affective_observations:
            raise ValueError(
                "No affective observations were provided."
            )

        print("1) Estimating confidence...")

        observations = self._confidence_method.estimate(
            self._affective_observations
        )

        if observations is None:
            observations = self._affective_observations

        print("2) Converting CAT observations to VAD...")

        observations = self._cat_to_vad_method.convert(
            observations
        )

        if observations is None:
            raise ValueError(
                "The CAT-to-VAD method returned no observations."
            )

        print("3) Fusing observations...")

        user_affective_state = self._fusion_method.fuse(
            observations
        )

        return self._normalize_vad_state(
            user_affective_state
        )

    @staticmethod
    def _normalize_vad_state(
        state: VADState,
    ) -> VADState:
        """
        Validates and rounds the final fused VAD state.
        """
        if not isinstance(state, dict):
            raise TypeError(
                "The fusion method must return a dictionary."
            )

        required_dimensions = {
            "valence",
            "arousal",
            "dominance",
        }

        missing_dimensions = (
            required_dimensions - state.keys()
        )

        if missing_dimensions:
            raise ValueError(
                "The fusion result is missing dimensions: "
                f"{sorted(missing_dimensions)}"
            )

        return {
            "valence": round(
                float(state["valence"]),
                2,
            ),
            "arousal": round(
                float(state["arousal"]),
                2,
            ),
            "dominance": round(
                float(state["dominance"]),
                2,
            ),
        }

    @staticmethod
    def _validate_observation(
        observation: AffectiveObservation,
    ) -> None:
        """
        Validates the common input contract accepted by PULSE.
        """
        if not isinstance(observation, dict):
            raise TypeError(
                "The observation must be a dictionary."
            )

        if not observation:
            raise ValueError(
                "An empty observation was received."
            )

        required_fields = {
            "source",
            "representation",
        }

        missing_fields = (
            required_fields - observation.keys()
        )

        if missing_fields:
            raise ValueError(
                "The observation is missing fields: "
                f"{sorted(missing_fields)}"
            )

        source = observation["source"]

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "The observation source must be a non-empty string."
            )

        representation = observation["representation"]

        if representation not in {"CAT", "VAD"}:
            raise ValueError(
                "representation must be either 'CAT' or 'VAD'."
            )

        if representation == "CAT":
            Pulse._validate_cat_observation(
                observation
            )
        else:
            Pulse._validate_vad_observation(
                observation
            )

    @staticmethod
    def _validate_cat_observation(
        observation: AffectiveObservation,
    ) -> None:
        probabilities = observation.get(
            "probabilities"
        )

        if not isinstance(probabilities, dict):
            raise TypeError(
                "A CAT observation must contain "
                "a probability dictionary."
            )

        if not probabilities:
            raise ValueError(
                "The probability distribution cannot be empty."
            )

        for emotion, probability in probabilities.items():
            if not isinstance(emotion, str):
                raise TypeError(
                    "Categorical emotion names must be strings."
                )

            probability_value = float(probability)

            if not 0.0 <= probability_value <= 1.0:
                raise ValueError(
                    f"Probability for '{emotion}' "
                    "must be in [0, 1]."
                )

    @staticmethod
    def _validate_vad_observation(
        observation: AffectiveObservation,
    ) -> None:
        values = observation.get("values")

        if not isinstance(values, dict):
            raise TypeError(
                "A VAD observation must contain "
                "a values dictionary."
            )

        for dimension in (
            "valence",
            "arousal",
            "dominance",
        ):
            if dimension not in values:
                raise ValueError(
                    "The VAD observation is missing "
                    f"the '{dimension}' dimension."
                )

            float(values[dimension])

        if "confidence" not in observation:
            raise ValueError(
                "A VAD observation must provide confidence."
            )

        confidence = float(
            observation["confidence"]
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be in [0, 1]."
            )


# ============================================================
# MQTT APPLICATION
# ============================================================

class PulseMqttApplication:
    """
    Connects the PULSE processing component to MQTT.
    """

    def __init__(
        self,
        pulse: Pulse,
        broker: str,
        port: int,
        robot_base_topic: str,
    ) -> None:
        self._pulse = pulse
        self._broker = broker
        self._port = port

        self._robot_base_topic = (
            robot_base_topic.rstrip("/")
        )

        self._perception_topic = (
            f"{self._robot_base_topic}/PERCEPTION"
        )

        self._input_topic = (
            f"{self._robot_base_topic}/PULSE/INPUT"
        )

        self._output_topic = (
            f"{self._robot_base_topic}"
            "/USER_AFFECTIVE_STATE"
        )

        self._client = mqtt_client.Client()

        self._client.on_connect = (
            self._on_connect
        )

        self._client.on_message = (
            self._on_message
        )

    def _on_connect(
        self,
        client: mqtt_client.Client,
        userdata: Any,
        flags: dict[str, Any],
        rc: int,
    ) -> None:
        if rc != 0:
            print(
                "PULSE connected to MQTT with "
                f"error code {rc}."
            )
            return

        client.subscribe(
            [
                (self._input_topic, 1),
                (self._perception_topic, 1),
            ]
        )

        print(
            "PULSE - Perception and Unification "
            "Layer for State Estimation - Connected."
        )

    def _on_message(
        self,
        client: mqtt_client.Client,
        userdata: Any,
        msg: Any,
    ) -> None:
        try:
            message = json.loads(
                msg.payload.decode("utf-8")
            )

            if msg.topic == self._perception_topic:
                self._handle_perception_message(
                    message
                )
                return

            if msg.topic == self._input_topic:
                self._handle_input_message(
                    message
                )
                return

            print(
                f"Unexpected MQTT topic: {msg.topic}"
            )

        except json.JSONDecodeError as error:
            print(
                "Invalid JSON received on topic "
                f"{msg.topic}: {error}"
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            print(
                "Invalid PULSE message on topic "
                f"{msg.topic}: {error}"
            )

        except Exception as error:
            print(
                "Unexpected error while processing "
                f"MQTT message: {error}"
            )

    def _handle_perception_message(
        self,
        message: dict[str, Any],
    ) -> None:
        action = str(
            message.get("action", "")
        ).upper()

        if action == "START":
            sources = message.get("sources")

            if sources is None:
                raise ValueError(
                    "A PERCEPTION START message "
                    "must provide sources."
                )

            self._pulse.start_perception_window(
                sources
            )
            return

        if action == "END":
            print(
                "Perception window END received."
            )
            return

        raise ValueError(
            f"Unknown PERCEPTION action: {action}"
        )

    def _handle_input_message(
        self,
        message: AffectiveObservation,
    ) -> None:
        user_affective_state = (
            self._pulse.receive_observation(
                message
            )
        )

        if user_affective_state is None:
            return

        self._publish_user_affective_state(
            user_affective_state
        )

    def _publish_user_affective_state(
        self,
        user_affective_state: VADState,
    ) -> None:
        payload = json.dumps(
            user_affective_state,
            ensure_ascii=False,
        )

        result = self._client.publish(
            self._output_topic,
            payload,
            qos=1,
        )

        if result.rc != mqtt_client.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                "Unable to publish "
                "USER_AFFECTIVE_STATE. "
                f"MQTT error code: {result.rc}"
            )

        print(
            "User affective state published on "
            f"{self._output_topic}: "
            f"{user_affective_state}"
        )

    def run(self) -> None:
        try:
            self._client.connect(
                self._broker,
                self._port,
            )

        except Exception as error:
            print(
                "Unable to connect to the "
                f"MQTT broker: {error}"
            )

            raise SystemExit(1) from error

        try:
            self._client.loop_forever()

        except KeyboardInterrupt:
            print("\nStopping PULSE...")

        finally:
            self._client.disconnect()


# ============================================================
# COMPONENT CONFIGURATION
# ============================================================

def create_pulse() -> Pulse:
    """
    Configures one concrete PULSE instantiation.

    These implementations are selected for the current
    demonstration and are not imposed by the architecture.
    """
    return Pulse(
        confidence_method=EntropyConfidenceMethod(),
        cat_to_vad_method=ReferenceBasedCatToVadMethod(),
        fusion_method=WeightedFusionMethod(),
    )


def main() -> None:
    pulse = create_pulse()

    application = PulseMqttApplication(
        pulse=pulse,
        broker=config.MQTT_BROKER_ADRESS,
        port=config.MQTT_PORT,
        robot_base_topic=(
            robot_profile.ROBOT_BASE_TOPIC
        ),
    )

    application.run()


if __name__ == "__main__":
    main()