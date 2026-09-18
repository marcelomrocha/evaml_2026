from paho.mqtt import client as mqtt_client

import numpy as np

import time

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
sim_base_topic = robot_profile.SIMULATOR_BASE_TOPIC
robot_base_topic = robot_profile.ROBOT_BASE_TOPIC


# Initializing
empathy = 0.0
decay = 0.0
user_affective_state_array = np.array([0.0, 0.0, 0.0], dtype=np.float16)
robot_affective_state_array = np.array([0.0, 0.0, 0.0], dtype=np.float16)
robot_empathy_state_array = np.array([0.0, 0.0, 0.0], dtype=np.float16)
robot_vad_mood_array = np.array([0.0, 0.0, 0.0], dtype=np.float16)
target = "NONE"
t_zero = 0.0
decay_delay = 0.0

# MQTT
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    # Subscribing in on_connect() means that if we lose the connection and
    # Reconnect then subscriptions will be renewed.
    # client.subscribe(topic=[(sim_base_topic + '/USER_AFFECTIVE_STATE', 1), ]) # Simulator topic
    # client.subscribe(topic=[(sim_base_topic + '/ROBOT_AFFECTIVE_PROFILE', 1), ]) # Simulator topic
    client.subscribe(topic=[(robot_base_topic + '/USER_AFFECTIVE_STATE', 1), ]) # Robot topic
    client.subscribe(topic=[(robot_base_topic + '/ROBOT_AFFECTIVE_PROFILE', 1), ]) # Robot topic
    print("ROSE - Robot Sentiment Engine - Connected.")
            

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global empathy, decay, robot_affective_state_array, user_affective_state_array, robot_vad_mood_array, target, decay_delay

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
        # Extraindo os valores da estrutura e armazenando nas variáveis do ROSE
        empathy = float(message['empathy'])
        decay = float(message['decay'])
        decay_delay = float(message['delay']) # Valor em segundos
        robot_v_mood = float(message['mood']['valence'])
        robot_a_mood = float(message['mood']['arousal'])
        robot_d_mood = float(message['mood']['dominance'])
        # Transforma os valores v, a e d em uma vetor numpy;
        robot_vad_mood_array = np.array([robot_v_mood, robot_a_mood, robot_d_mood], dtype=np.float16)
        
        mood = array_to_json(robot_vad_mood_array)
        client.publish(robot_base_topic + '/BASE_MOOD', mood)
        # robot_affective_state_array = robot_vad_mood_array
        target = "MOOD"
        print(f"Mensagem de profile. Empathy:{empathy}, Decay:{decay}")
        # robot_affective_state_update() # Inicializa o estado afetivo do robô com base no base mood.

        
    elif msg.topic == robot_base_topic + '/USER_AFFECTIVE_STATE':
        # A informação sobre o estado afetivo do usuário é enviada pelo PULSE
        # Estrutura da mansagem JSON:
        # user_emotion_vad_vector = {
        #     "valence": 0.00, # Arredonda para duas casas decimais. Intervalo entre [0, 1], inclusive.
        #     "arousal": 0.00, # Arredonda para duas casas decimais. Intervalo entre [0, 1], inclusive.
        #     "dominance": 0.00 # Arredonda para duas casas decimais. Intervalo entre [0, 1], inclusive.
        # }
        message = json.loads(msg.payload.decode())
        user_v = float(message['valence'])
        user_a = float(message['arousal'])
        user_d = float(message['dominance'])
        user_affective_state_array = np.array([user_v, user_a, user_d], dtype=np.float16)
        empathic_transformation()
        target = "EMPATHY"
        
        
        # Sempre que chega uma mensagem com o estado afetivo do usuário acontece a transformação empática.
        
        # robot_affective_target_state = user_affective_state
        # robot_affective_target_state = user_affective_state * empathy # A emoção do robo
        
        # user_affective_state = user_affective_state * empathy + robot_affective_state * (1 - empathy)
        # robot_affective_target_state = user_affective_state
        print(f"User aff state: {user_affective_state_array}, type: {type(user_affective_state_array)}")
        # time.sleep(2)
        # empathic_transformation()

    
def empathic_transformation():
    global robot_affective_state_array, robot_empathy_state_array
    robot_empathy_state_array = (empathy * user_affective_state_array) + ((1 - empathy) * robot_affective_state_array)
    print(f"Empathy Transformation: {empathy}.")
    print(f"User: {user_affective_state_array}, Robot_Empathy: {robot_empathy_state_array}.")

    # robot_affective_state_update()


def array_to_json(vad_array): # Retorna uma estado VAD em JSON pronto para publicação.

    vad_state = {
        "valence": round(float(vad_array[0]), 2), # round(float(vad_array[0]), 2)
        "arousal": round(float(vad_array[1]), 2),
        "dominance": round(float(vad_array[2]), 2)
    }

    # print(vad_state, type(vad_state))

    return json.dumps(vad_state)


def robot_affective_state_update():
    # Envia os valores VAD do base mood como o estado afetivo inicial do robô.
    # Os módulos VAD Monitor (mostra o vetor no espaço 2D) e SharedStaeSync (armazena o vetor VAD na memória) assinam este tópico.
    # Assim, atualiza a memória do robô e o VAD monitor.
    client.publish(robot_base_topic + '/ROBOT_AFFECTIVE_STATE', array_to_json(robot_empathy_state_array))
    print(robot_base_topic + '/ROBOT_AFFECTIVE_STATE', array_to_json(robot_empathy_state_array))
  
            
def main_loop():
    global target, robot_affective_state_array, t_zero

    if target == "EMPATHY":
        if np.allclose(robot_affective_state_array, robot_empathy_state_array, atol=0.005, rtol=0) != True:
            robot_affective_state_array = robot_affective_state_array + (0.3 * (robot_empathy_state_array - robot_affective_state_array))
            client.publish(robot_base_topic + '/ROBOT_AFFECTIVE_STATE', array_to_json(robot_affective_state_array))
            print("indo para ", target, robot_affective_state_array, robot_empathy_state_array)
        else:
            robot_affective_state_array = robot_empathy_state_array.copy()
            print("Cheguei no estado empático...")
            print(f"Emotional latency pediod: {decay_delay} s.")
            t_zero = time.time() # Agora
            target = "MOOD"

    elif target == "MOOD":
        t = time.time()
        if t - t_zero > decay_delay:
            if np.allclose(robot_affective_state_array, robot_vad_mood_array, atol=0.005, rtol=0) != True:
                robot_affective_state_array = robot_affective_state_array + (0.2 * decay * (robot_vad_mood_array - robot_affective_state_array))
                client.publish(robot_base_topic + '/ROBOT_AFFECTIVE_STATE', array_to_json(robot_affective_state_array))
                print("indo para", target, robot_affective_state_array, robot_vad_mood_array)
            else:
                robot_affective_state_array = robot_vad_mood_array.copy()
                print("Cheguei no mood...")
                target = "NONE"

    time.sleep(0.1)
          


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

client.loop_start()

time.sleep(1)
while(1):
    main_loop()
