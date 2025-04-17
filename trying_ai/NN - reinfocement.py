import tensorflow as tf
import gymnasium as gym
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
import numpy as np
from random import randint, shuffle, uniform
from Calcul import G1, G2, J
from keras.layers.advanced_activations import PReLU
from time import time
print(G2(1))
print(G1(1))
temps1 = time()

#https://towardsdatascience.com/an-ai-agent-learns-to-play-tic-tac-toe-part-3-training-a-q-learning-rl-agent-2871cef2faf0
# Voir les liens en dessous
#https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc
#https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
#https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/
#https://www.youtube.com/watch?v=cO5g5qLrLSo
#https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial?hl=fr
#https://www.youtube.com/watch?v=2nKD6zFQ8xI
def dense_layer(num_units):
    return tf.keras.layers.Dense(num_units, activation=tf.keras.activation.relu,kernel_initializers.VarianceScaling(scale=2.0, mode='fan_in', distibution='truncated_normal'))

dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])



def play_game(joueur1, joueur2):
    action_log_j1 = []
    action_log_j2 = []
    victoire_j1 = False
    pos1 = [0, 25]
    pos2 = [0, 0]
    for i in range(40):
        n = randint(0, 4)
        sortie1 = joueur1.predict(np.array([[n, pos1[0], pos1[1], pos2[0], pos2[1]]]))
        sortie2 = joueur2.predict(np.array([[n, pos1[0], pos1[1], pos2[0], pos2[1]]]))
        d1x, d1y, d2x, d2y = sortie1[0][0], sortie1[0][1], sortie2[0][0], sortie2[0][1]
        d1x = int(d1x)
        d1y = int(d1y)
        d2x = int(d2x)
        d2y = int(d2y)
        if abs(d1x) + abs(d1y) > (n+1):
            d1x, d1y = 0, 0
        if abs(d2x) + abs(d2y) > n:
            d2x, d2y = 0, 0
        pos1[0] += d1x
        pos1[1] += d1y
        pos2[0] += d2x
        pos2[1] += d2y
        action_log_j1.append([d1x, d1y])
        action_log_j2.append([d2x, d2y])
        if pos1 == pos2:
            victoire_j1 = True
            break
    distance_finale = J(pos1, pos2)  # J calcul, la distance entre deux points
    return action_log_j1, action_log_j2, victoire_j1, distance_finale


def agent_generator():
    cte = 0.75
    agent = tf.keras.Sequential([
     tf.keras.layers.Dense(units=8, input_shape=(5, )),
     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),
     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),
     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),
     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),
     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),
       # Reseau a 5 entrees coord j1x coord j1y coord j2x coord j2y et nombre de deplacement
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),  # 10 hidden layer de 8 neuronnes chacune
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),  # sortie de deux neuronnes pour avoir les deux valeurs qui
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),  # correspondent au deplacement
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),

     tf.keras.layers.Dense(units=2, activation=PReLU(alpha_initializer=tf.initializers.constant(cte)))
    ])
    poids = np.array(agent.get_weights())
    agent.set_weights(poids)
    return agent


def clone_generator_decalage(clone):
    mon_clone = tf.keras.models.clone_model(clone)
    poids = mon_clone.get_weights()
    for a in range(len(poids)):
        for b in range(len(poids[a])): # peut être-1
            poids[a][b] += uniform(-0.02, 0.02)
    """
    print(poids)
    nouveau_poids = [[synapse + uniform(-0.005, 0.005) for synapse in ensemble_synapse]for ensemble_synapse in poids]
    print(nouveau_poids)
    """
    mon_clone.set_weights(np.array(poids)) # essayer de changer tous les poids aléatoirement de façon différente ou partir sur le reward
    return tf.keras.models.clone_model(clone)

"""
nombre_agent = 200
J1_Agent = []
J2_Agent = []
valeurs = []
for i in range(nombre_agent):
    Agent1 = agent_generator()
    Agent2 = agent_generator()
    #  Agent1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #  Agent2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    J1_Agent.append(Agent1)
    J2_Agent.append(Agent2)

# Testmodel1 = agent_generator()
# Testmodel2 = agent_generator()

# Testmodel2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Testmodel1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def sauver_quart(Agent_list):
    id = []
    for i in range(int(len(Agent_list)/4)):
        minimum = 0
        maximum = 0
        for j in range(len(Agent_list)):
            if Agent_list[j] > Agent_list[maximum]:
                maximum = j
            if Agent_list[j] < Agent_list[minimum]:
                minimum = j
        id.append((maximum,minimum))
        Agent_list[minimum] = 0
        Agent_list[maximum] = 0
        Agent_list.remove(0)
        Agent_list.remove(0)
    return id


generation = 500

for j in range(generation):
    print(len(J1_Agent))
    for i in range(nombre_agent):
        a, b, c, valeur = play_game(J1_Agent[i], J2_Agent[i])
        valeurs.append(valeur)
        print(valeur)
    les_indices = sauver_quart(valeurs)
    Bestj1 = [J1_Agent[i[1]] for i in les_indices]
    Bestj2 = [J2_Agent[i[0]] for i in les_indices]
    J1_Agent = Bestj1
    J2_Agent = Bestj2
    for i in range(nombre_agent-int(nombre_agent/4)):
        J1_Agent.append(clone_generator_decalage(Bestj1[randint(1, int(nombre_agent/4))-1]))
        J2_Agent.append(clone_generator_decalage(Bestj2[randint(1, int(nombre_agent/4))-1]))
    shuffle(J1_Agent)
    valeurs = []



for i in range(1000):
    n = randint(0, 4)
    rewards_j1 = G1(n)
    rewards_j2 = G2(n)
    x_j1 = np.array([])
    y_j1 = np.array([])
    x_j2 = np.array([])
    y_j2 = np.array([])
    Testmodel1.fit(x_j1, y_j1, initial_epoch=i, epochs=i + 1, sample_weight=rewards_j1)
    Testmodel2.fit(x_j2, y_j2, initial_epoch=i, epochs=i + 1, sample_weight=rewards_j2)

# Testmodel1.summary()
# print(Testmodel1.get_weights())


def sauvegarder(list, mot):
    for i in range(len(list)):
        list[i].save("Agent"+ mot + str(i))


sauvegarder(Bestj1," j1 ")
sauvegarder(Bestj2," j2 ")
b = time()
print((temps1 - b)/60)

Bestj1.save("best1")
Bestj2.save("best2")
"""
# tf.keras.models.load_model()
# structure du NN entrer 2 pour pos j1 2 pour pos j2 1 pour deplacement sortie 2 pour coord du deplacement
# https://blog.tensorflow.org/2021/10/building-board-game-app-with-tensorflow.html
# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
