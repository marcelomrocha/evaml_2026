from paho.mqtt import client as mqtt_client

import json

import math

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

# Armazena a lista de fontes (FER, TER etc) que serão enviadas pelos módulos multimodais
# Esta informação é obtida na mensagem do tópico PERCEPTION
list_sources_to_receive = [] 

# Armazena as observaçoes enviadas dos inputs (sources)
# Essa lista será passada para a função que faz a fusão e envia a emoção do usuário para o ROSE
# A fusão só é feita após a chegada de todos os itens contidos em list_sources_to_receive
affective_observations = [] 

user_affective_state_vad_vector = {}



# MQTT
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    # Subscribing in on_connect() means that if we lose the connection and
    # Reconnect then subscriptions will be renewed.
    # client.subscribe(topic=[(sim_base_topic + '/PULSE/INPUT', 1), ]) # Simulator topic
    client.subscribe(topic=[(robot_base_topic + '/PULSE/INPUT', 1), ]) # Robot topic
    # client.subscribe(topic=[(sim_base_topic + '/PERCEPTION', 1), ]) # Simulator topic
    client.subscribe(topic=[(robot_base_topic + '/PERCEPTION', 1), ]) # Robot topic
    print("PULSE - Processing Unit for Affective Latent Estimation - Connected.")
            

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global list_sources_to_receive , affective_observations    

    # Recebe a definição das fontes usadas na janela de percepção
    if msg.topic == robot_base_topic + "/" + 'PERCEPTION':
        
        message = json.loads(msg.payload.decode()) # Transforma a mensagem para JSON

        # Padrão das duas mensagens (START e END) do tópico PERCEPTION
        # {"action": "START", "sources": ["FER", "TER"]}
        # {"action": "END"}

        if message["action"] == "START": # A mensagem de END não interessa ao PULSE porque ele finaliza após a recepção de todos os sources 
            list_sources_to_receive = message["sources"] # Inicializa a lista de fontes a receber
            affective_observations = [] # Zera as observações anteriores
            print(f"Source to be received in Perception Window: {list_sources_to_receive}.")
        
    
    if msg.topic == robot_base_topic + "/" + 'PULSE/INPUT':

        message = json.loads(msg.payload.decode()) # Transforma a mensagem para JSON
        if message == {}:
            print("Mensagem vazia...")
            exit(1)
        # Padrão da observação vinda dos inputs do tipo CAT
            # {
            #     "source": "FER",
            #     "representation": "CAT",
            #     "probabilities": {
            #         "neutral": 0.98,
            #         "surprise": 0.01,
            #         "happiness": 0.01,
            #         "fear": 0.0,
            #         "anger": 0.0,
            #         "sadness": 0.0,
            #         "disgust": 0.0
            #      }
            # }

        # Padrão da observação vinda dos inputs do tipo VAD
            # {
            #     "source": "SER",
            #     "representation": "VAD",
            #     "values": {
            #         "valence": -0.42,
            #         "arousal": 0.51,
            #         "dominance": 0.28
            #     },
            #     "confidence": 0.87
            # }

        if message["source"] in list_sources_to_receive:
            print(f"Source {message["source"]} received...")
            list_sources_to_receive.remove(message["source"])
            affective_observations.append(message)
            if len(list_sources_to_receive) == 0: # Não há mais sources a serem recebidas. Inicia fusão multimodal.
                print("1) Setting confidence...")
                set_confidence()
                print("2) Type conversion...")
                cat_to_vad_conversion()
                print("3) Fusing observations...")
                fuse_observations()
                print("4) Sending user...")
                send_user_affective_state()


def set_confidence():
    # Primeiro: Calcula-se a entropia
    # Segundo: Normalização
    # Confiança: 1 - Normalização
    for obs in affective_observations:
        if obs['representation'] == "CAT":
            entropy = -sum(p * math.log2(p) for p in obs['probabilities'].values() if p > 0)
            Hmax = math.log2(7)
            confidence = 1 - entropy/Hmax # entropy/Hmax equivale à normalização
            obs['confidence'] = confidence # A confiança é calculada é adicionada à estrutura.
        else:
            # Observações do tipo VAD já devem vir com a confiança calculada no intervalo de [0, 1].
            pass


def cat_to_vad_conversion():
    # Conferir com os valores do artigo.
    # Apesar de aparecer no cálculo, a dominância não será utilizada.
    VAD_REFERENCE = {
        "happiness": (0.76, 0.48, 0.35),
        "sadness": (-0.63, -0.27, -0.33),
        "anger": (-0.51, 0.59, 0.25),
        "fear": (-0.64, 0.60, -0.43),
        "surprise": (0.40, 0.67, -0.13),
        "disgust": (-0.60, 0.35, 0.11),
        "neutral": (0.00, 0.00, 0.00)
    }

    for obs in affective_observations:

        if obs['representation'] == "CAT":
            valence = 0.0
            arousal = 0.0
            dominance = 0.0

            for emotion, probability in obs["probabilities"].items():

                v, a, d = VAD_REFERENCE[emotion]

                valence += probability * v
                arousal += probability * a
                dominance += probability * d

            obs["representation"] = "VAD"
            obs.pop("probabilities")
            obs['values'] = {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance
                }
            print("VAD conversion:", obs)


def fuse_observations():
    global user_emotion_vad_vector

    if not affective_observations:
        raise ValueError("No observations were provided.")

    total_confidence = sum(
        obs["confidence"]
        for obs in affective_observations
    )

    if total_confidence <= 0:
        raise ValueError("The total confidence must be greater than zero.")

    valence = sum(
        obs["confidence"] * obs["values"]["valence"]
        for obs in affective_observations
    ) / total_confidence

    arousal = sum(
        obs["confidence"] * obs["values"]["arousal"]
        for obs in affective_observations
    ) / total_confidence

    dominance = sum(
        obs["confidence"] * obs["values"]["dominance"]
        for obs in affective_observations
    ) / total_confidence

    user_emotion_vad_vector = {
        "valence": round(valence, 2), # Arredonda para duas casas decimais.
        "arousal": round(arousal, 2),
        "dominance": round(dominance, 2)
    }


def send_user_affective_state():
    global user_emotion_vad_vector
    print(f"Sending user affective state VAD vector {user_emotion_vad_vector} to BASE_TOPIC/USER_AFFECTIVE_STATE topic.")
    user_emotion_vad_vector = json.dumps(user_emotion_vad_vector) # Transforma para JSON
    # Estrutura da saída em JSON:
    # user_emotion_vad_vector = {
    #     "valence": 0.00, # Arredonda para duas casas decimais. Intervalo entre [0, 1], inclusive.
    #     "arousal": 0.00, # Arredonda para duas casas decimais. Intervalo entre [0, 1], inclusive.
    #     "dominance": 0.00 # Arredonda para duas casas decimais. Intervalo entre [0, 1], inclusive.
    # }

    # client.publish(robot_profile.SIMULATOR_BASE_TOPIC + "/USER_AFFECTIVE_STATE", user_emotion_vad_vector)
    client.publish(robot_base_topic + "/USER_AFFECTIVE_STATE", user_emotion_vad_vector)



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

client.loop_forever()
