from paho.mqtt import client as mqtt_client

import json
import sys
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

parent_dir = os.path.abspath(
    os.path.join(BASE_DIR, "../")
)

sys.path.append(parent_dir)

import config

sys.path.append(
    os.getcwd() + "/" + "robot_package/"
)

import robot_profile

from empathic_behavior_mapping_method import (
    EmpathicBehaviorMappingMethod
)

from reference_point_behavior_mapping import (
    ReferencePointBehaviorMapping
)


# ============================================================
# MQTT CONFIGURATION
# ============================================================

broker = config.MQTT_BROKER_ADRESS
port = config.MQTT_PORT

robot_base_topic = robot_profile.ROBOT_BASE_TOPIC


# ============================================================
# AEB
# ============================================================

class AEB:

    def __init__(
        self,
        behavior_mapping_method: EmpathicBehaviorMappingMethod,
        mapping_parameters: dict | None = None
    ):

        self.behavior_mapping_method = behavior_mapping_method

        self.mapping_parameters = (
            mapping_parameters
            if mapping_parameters is not None
            else {}
        )

        self.robot_current_behavior = None

        self.client = mqtt_client.Client()

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    # --------------------------------------------------------
    # MQTT CALLBACKS
    # --------------------------------------------------------

    def on_connect(
        self,
        client,
        userdata,
        flags,
        rc
    ):

        client.subscribe(
            robot_base_topic
            + "/ROBOT_AFFECTIVE_STATE",
            qos=1
        )

        client.subscribe(
            robot_base_topic
            + "/ROBOT_AFFECTIVE_PROFILE",
            qos=1
        )

        print(
            "AEB - Automatic Empathic Behavior - Connected."
        )

    def on_message(
        self,
        client,
        userdata,
        msg
    ):

        # ----------------------------------------------------
        # ROBOT AFFECTIVE PROFILE
        # ----------------------------------------------------

        if (
            msg.topic
            == robot_base_topic
            + "/ROBOT_AFFECTIVE_PROFILE"
        ):

            message = json.loads(
                msg.payload.decode()
            )

            # Uses the base mood as the initial
            # robot affective state.
            robot_affective_state = message["mood"]

            self.process_robot_affective_state(
                robot_affective_state
            )

        # ----------------------------------------------------
        # ROBOT AFFECTIVE STATE
        # ----------------------------------------------------

        elif (
            msg.topic
            == robot_base_topic
            + "/ROBOT_AFFECTIVE_STATE"
        ):

            robot_affective_state = json.loads(
                msg.payload.decode()
            )

            self.process_robot_affective_state(
                robot_affective_state
            )

    # --------------------------------------------------------
    # AEB CORE
    # --------------------------------------------------------

    def process_robot_affective_state(
        self,
        robot_affective_state: dict
    ):

        behavior_parameters = (
            self.behavior_mapping_method.map(
                robot_affective_state,
                self.mapping_parameters
            )
        )

        # Avoids publishing the same behavior repeatedly.
        if (
            behavior_parameters
            != self.robot_current_behavior
        ):

            self.robot_current_behavior = (
                behavior_parameters
            )

            self.publish_behavior_parameters(
                behavior_parameters
            )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    def publish_behavior_parameters(
        self,
        behavior_parameters: dict
    ):

        payload = json.dumps(
            behavior_parameters
        )

        topic = (
            robot_base_topic
            + "/ROBOT_BEHAVIOR_STATE"
        )

        self.client.publish(
            topic,
            payload
        )

        print(
            topic,
            payload
        )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    def run(self):

        try:

            self.client.connect(
                broker,
                port
            )

        except Exception as error:

            print(
                "Unable to connect to Broker."
            )

            print(error)

            return

        self.client.loop_forever()


# ============================================================
# CURRENT AEB INSTANTIATION
# ============================================================

def create_reference_points(
    emotion_l4_points: dict
) -> dict:

    points = {
        "neutral": {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0
        }
    }

    for emotion, final_point in emotion_l4_points.items():

        for level in range(1, 5):

            proportion = level / 4

            points[
                f"{emotion}_l{level}"
            ] = {

                dimension: round(
                    float(value * proportion),
                    2
                )

                for dimension, value
                in final_point.items()
            }

    return points


EMOTION_L4_POINTS = {

    "happiness": {
        "valence": 0.76,
        "arousal": 0.48,
        "dominance": 0.35
    },

    "anger": {
        "valence": -0.43,
        "arousal": 0.67,
        "dominance": 0.34
    },

    "sadness": {
        "valence": -0.63,
        "arousal": -0.27,
        "dominance": -0.33
    }
}


reference_points = create_reference_points(
    EMOTION_L4_POINTS
)


mapping_parameters = {
    "reference_points": reference_points
}


mapping_method = ReferencePointBehaviorMapping()


aeb = AEB(
    behavior_mapping_method=mapping_method,
    mapping_parameters=mapping_parameters
)


aeb.run()