from paho.mqtt import client as mqtt_client

import numpy as np

import math

import json

import sys
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
parent_dir = os.path.abspath(os.path.join(BASE_DIR, "../"))
sys.path.append(parent_dir)


import config  # Module with network device configurations.

sys.path.append(os.getcwd() + "/" + "robot_package/")

import robot_profile


broker = config.MQTT_BROKER_ADRESS # Broker address.
port = config.MQTT_PORT # Broker Port.
robot_base_topic = robot_profile.ROBOT_BASE_TOPIC


# Initializing
robot_current_affective_state_point = ""


# Definição dos pontos base de cada emoção
EMOTION_L4_POINTS = {
    "happiness": {
        "valence": 0.76,
        "arousal": 0.48,
        "dominance": 0.35
    },
    "anger": {
        "valence": -0.43, # -0.51
        "arousal": 0.67, # 0.59
        "dominance": 0.34 # 0.25
    },
    "sadness": {
        "valence": -0.63,
        "arousal": -0.27,
        "dominance": -0.33
    }
}

def create_reference_points(emotion_l4_points):
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

            points[f"{emotion}_l{level}"] = {
                dimension: round(float(value * proportion), 2) # Duas casas decimais como padrão dos outros módulos
                for dimension, value in final_point.items()
            }

    return points


reference_points = create_reference_points(EMOTION_L4_POINTS)


# MQTT
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    # Subscribing in on_connect() means that if we lose the connection and
    # Reconnect then subscriptions will be renewed.
    client.subscribe(topic=[(robot_base_topic + '/ROBOT_AFFECTIVE_STATE', 1), ]) # Robot topic
    client.subscribe(topic=[(robot_base_topic + '/ROBOT_AFFECTIVE_PROFILE', 1), ]) # Robot topic
    print("AEB - Automatic Empathic Behavior - Connected.")
            


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global robot_current_affective_state_point

    # Recebe os parâmetros (JSON) com o perfil afetivo do robô vindo do script EvaML
    if msg.topic == robot_base_topic + '/ROBOT_AFFECTIVE_PROFILE':
        # Estrutura do JSON:
        # message = {
        #     "profile": String",
        #     "empathy": 0.00, # Intervalo entre [0, 1], inclusive. 
        #     "decay": 0.00, # Intervalo entre [0, 1], inclusive.
        #     "delay": 0.00, # Intervalo de [0, 10] segundos.
        #     "mood": {
        #         "valence": 0.00, # Intervalo entre [-1, 1], inclusive.
        #         "arousal": 0.00,  # Intervalo entre [-1, 1], inclusive.
        #         "dominance": 0.00  # Intervalo entre [-1, 1], inclusive.
        #     }
        # }
        message = json.loads(msg.payload.decode())

        mood = message['mood']
        result = find_nearest_reference_point(mood, reference_points)
        robot_current_affective_state_point = create_affective_behavior(result['point'])["affective_state"]
        print(f"Ponto mais próximo: {result['point']}")
        print(f"Distância: {result['distance']:.2f}")
        print(f"Json de saída para o SENSEI {create_affective_behavior(result['point'])}")
        print(robot_base_topic + '/ROBOT_BEHAVIOR_STATE', json.dumps(create_affective_behavior(result['point'])))
        client.publish(robot_base_topic + '/ROBOT_BEHAVIOR_STATE', json.dumps(create_affective_behavior(result['point'])))

        
    elif msg.topic == robot_base_topic + '/ROBOT_AFFECTIVE_STATE':
        # Estrutura da mansagem JSON:
        # user_emotion_vad_vector = {
        #     "valence": 0.00, # Arredonda para duas casas decimais. Intervalo entre [-1, 1], inclusive.
        #     "arousal": 0.00, # Arredonda para duas casas decimais. Intervalo entre [-1, 1], inclusive.
        #     "dominance": 0.00 # Arredonda para duas casas decimais. Intervalo entre [-1, 1], inclusive.
        # }

        mood = json.loads(msg.payload.decode())
        # # Extraindo os valores da estrutura e armazenando nas variáveis do AEB
        # robot_v_mood = float(mood['valence'])
        # robot_a_mood = float(mood['arousal'])
        # robot_d_mood = float(mood['dominance'])
        # # Transforma os valores v, a e d em uma vetor numpy;
        # robot_affective_state_array = np.array([robot_v_mood, robot_a_mood, robot_d_mood], dtype=np.float16)

        result = find_nearest_reference_point(mood, reference_points)
        aux_robot_current_affective_state_point = create_affective_behavior(result['point'])["affective_state"]
        if aux_robot_current_affective_state_point != robot_current_affective_state_point:
            robot_current_affective_state_point = aux_robot_current_affective_state_point
            print(robot_base_topic + '/ROBOT_BEHAVIOR_STATE', json.dumps(create_affective_behavior(result['point'])["affective_state"]))
            client.publish(robot_base_topic + '/ROBOT_BEHAVIOR_STATE', json.dumps(create_affective_behavior(result['point'])))



def array_to_json(vad_array): # Retorna uma estado VAD em JSON pronto para publicação.

    vad_state = {
        "valence": round(float(vad_array[0]), 2), # round(float(vad_array[0]), 2)
        "arousal": round(float(vad_array[1]), 2),
        "dominance": round(float(vad_array[2]), 2)
    }

    return json.dumps(vad_state)


def euclidean_distance(point_a, point_b):
    return math.sqrt(
        sum(
            (
                float(point_a[dimension]) # até aqui os valores eram strings
                - point_b[dimension]
            ) ** 2
            for dimension in (
                "valence",
                "arousal",
                "dominance"
            )
        )
    )
          

# Calcula o estado afetivo do robô com base na distância euclidiana.
# Também, retorna o JSON com o estado (em seu nível) e alguns comportamentos definidos a partir da posição no espaço VAD.
def find_nearest_reference_point(vad_point, reference_points):
    distances = {
        name: euclidean_distance(
            vad_point,
            reference_point
        )
        for name, reference_point in reference_points.items()
    }

    nearest_name = min(distances, key=distances.get)

    return {
        "point": nearest_name,
        "distance": distances[nearest_name],
        "all_distances": distances
    }



def create_affective_behavior(affective_state):
    VALID_EMOTIONS = {
        "happiness",
        "anger",
        "sadness",
        "neutral"
    }

    VALID_LEVELS = {
        "l1",
        "l2",
        "l3",
        "l4"
    }

    if not isinstance(affective_state, str):
        raise TypeError(
            "O estado afetivo deve ser uma string."
        )

    affective_state = affective_state.strip().lower()

    # Caso especial: neutral
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
            "Formato inválido. Use 'neutral' ou, por exemplo, 'happiness_l1'."
        ) from error

    if emotion not in VALID_EMOTIONS:
        raise ValueError(
            f"Emoção inválida: '{emotion}'."
        )

    if level not in VALID_LEVELS:
        raise ValueError(
            f"Nível inválido: '{level}'."
        )

    return {
        "affective_state": affective_state,
        "facial_expression": level,
        "leds": level,
        "pose": level
    }


# Run the MQTT client thread.
client = mqtt_client.Client()
client.on_connect = on_connect
client.on_message = on_message
try:
    client.connect(broker, port)
except:
    print ("Unable to connect to Broker.")
    exit(1)

# You cannot use the "forever" method (as in other modules) because it blocks not allowing
# for the graphical interface thread loop to execute.
# client.loop_forever()
# client.loop_start()


client.loop_forever()

