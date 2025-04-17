import tensorflow as tf
import numpy as np
from random import randint, shuffle, uniform
from Calcul import J
#  from keras.layers.advanced_activations import PReLU
from time import time
temps1 = time()


def rectification(x: int, y: int, n: int) -> (int, int):
    """

    :param x: coordones x
    :param y: coordones y
    :param n: max de la somme en valeur abolue
    :return:
    """
    while abs(x)+abs(y) > n:
        choix = randint(1, 2)
        if choix == 1:
            if x > 0:
                x -= 1
            elif x < 0:
                x += 1
            else:
                if y > 0:
                    y -= 1
                else:
                    y += 1
        else:
            if y > 0:
                y -= 1
            elif y < 0:
                y += 1
            else:
                if x > 0:
                    x -= 1
                else:
                    x += 1
    return x, y


def play_game(joueur1, joueur2):
    """

    :param joueur1: Reseau de neuronnes
    :param joueur2: Reseau de neuronnes
    :return: les actions faites par le joueur 1,
    les actions faites par le joueurs 2,
    si le joueur 1 a gagné sous form de booléen,
    distance entre les deux joueurs,
    les gain du joueur 1,
    les gain du joueur 2
    """
    # action_log_j1 = []
    # action_log_j2 = []
    gain_du_joueur_1 = 0
    gain_du_joueur_2 = 0
    victoire_j1 = False
    pos1 = [0, 15]
    pos2 = [0, 0]
    d_avant = J(pos1, pos2)
    for i in range(200):
        n = 4
        #  table_gain1 = G1(n, pos1, pos2)
        #  table_gain2 = G2(n, pos1, pos2)
        sortie1 = joueur1.predict(np.array([[pos1[0], pos1[1], pos2[0], pos2[1]]]))
        sortie2 = joueur2.predict(np.array([[pos1[0], pos1[1], pos2[0], pos2[1]]]))
        d1x, d1y, d2x, d2y = sortie1[0][0]-0.5, sortie1[0][1]-0.5, sortie2[0][0]-0.5, sortie2[0][1]-0.5
        d1x = int(d1x*10)
        d1y = int(d1y*10)
        d2x = int(d2x*8)
        d2y = int(d2y*8)
        # print(d1x, d1y)
        # print(d2x, d2y)
        d1x, d1y = rectification(d1x, d1y, n+1)
        d2x, d2y = rectification(d2x, d2y, n)
        pos1[0] += d1x
        pos1[1] += d1y
        pos2[0] += d2x
        pos2[1] += d2y
        d_apres = J(pos1, pos2)
        gain_du_joueur_1 += d_avant - d_apres + 2*n + 1  # table_gain1[liste_table[n][(d2x, d2y)]][liste_table[n+1][(d1x, d1y)]]
        gain_du_joueur_2 += d_apres - d_avant + 2*n + 1  # table_gain2[liste_table[n][(d2x, d2y)]][liste_table[n+1][(d1x, d1y)]]
        d_avant = d_apres
        # action_log_j1.append([d1x, d1y])
        # action_log_j2.append([d2x, d2y])
        if pos1 == pos2:
            victoire_j1 = True
            gain_du_joueur_1 = 50000  # sur de passer
            break
    distance_finale = J(pos1, pos2)  # J calcul, la distance entre deux points
    return "action_log_j1", "action_log_j2", victoire_j1, distance_finale, gain_du_joueur_1, gain_du_joueur_2


def agent_generator():
    """

    :return: reseau de neuronne generer aleatoirement
    """
    # cte = uniform(0.70,0.80)
    agent = tf.keras.Sequential([
     tf.keras.layers.Dense(units=8, input_shape=(4, ), activation="sigmoid"),
     tf.keras.layers.Dense(units=8, activation="sigmoid"),
     tf.keras.layers.Dense(units=8, activation="sigmoid"),
     # tf.keras.layers.Dense(units=8, activation="sigmoid"),
     # tf.keras.layers.Dense(units=8, activation="sigmoid"),
     # tf.keras.layers.Dense(units=8, activation="sigmoid"),
     # Reseau a 5 entrees coord j1x coord j1y coord j2x coord j2y et nombre de deplacement
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),  # 10 hidden layer de 8 neuronnes chacune
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),  # sortie de deux neuronnes pour avoir les deux valeurs qui
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),  # correspondent au deplacement
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),
#     tf.keras.layers.Dense(units=8, activation=PReLU(alpha_initializer=tf.initializers.constant(cte))),

     tf.keras.layers.Dense(units=2, activation="sigmoid")
    ])
    # poids = np.array(agent.get_weights())
    # agent.set_weights(poids)
    return agent


def clone_generator_decalage(clone, mutation_rate):
    """

    :param clone: reseau de neuronne que l'on va faire "evoluer"
    :return: reseau de neuronne qui a "muter"
    """
    mon_clone = tf.keras.models.clone_model(clone)
    poids = mon_clone.get_weights()
    for a in range(len(poids)):
        for b in range(len(poids[a])):
            if type(poids[a][b]) is np.ndarray:
                for c in range(len(poids[a][b])):
                    poids[a][b][c] += uniform(-mutation_rate, mutation_rate)
            else:
                poids[a][b] += uniform(-mutation_rate, mutation_rate)
    """
    print(poids)
    nouveau_poids = [[synapse + uniform(-0.005, 0.005) for synapse in ensemble_synapse]for ensemble_synapse in poids]
    print(nouveau_poids)
    """
    mon_clone.set_weights(poids)  # essayer de changer tous les poids aléatoirement de façon différente
    return tf.keras.models.clone_model(clone)


def sauver_quart(Agent_list):
    """

    :param Agent_list: est une liste de tuple (gainj1,gainj2)
    :return: deux liste d'indice des resaux avec le plus de gain
    """
    id_agent_1 = []
    id_agent_2 = []
    for i in range(int(len(Agent_list)/4)):
        maximum_j1 = 0
        maximum_j2 = 0
        for j in range(len(Agent_list)):
            if Agent_list[j][0] > Agent_list[maximum_j1][0]:
                maximum_j1 = j
            if Agent_list[j][1] > Agent_list[maximum_j2][1]:
                maximum_j2 = j
        id_agent_1.append(maximum_j1)
        id_agent_2.append(maximum_j2)
        Agent_list[maximum_j1][0] = -1
        Agent_list[maximum_j2][1] = -1
    return id_agent_1, id_agent_2


def sauvegarder(list, mot):
    """

    :param list:
    :param mot:
    :return:
    """
    for i in range(len(list)):
        list[i].save("Agent" + mot + str(i))


nombre_agent = 40
div = 4
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


generation = 90

mutation_list = [0.01, 0.05, 0.2]

for j in range(generation):
    print(j, len(J1_Agent))
    for i in range(nombre_agent):
        a, b, c, d, gain_j1, gain_j2 = play_game(J1_Agent[i], J2_Agent[i])
        valeurs.append([gain_j1, gain_j2])
        print(gain_j1, gain_j2)
    les_indices1, les_indices2 = sauver_quart(valeurs)
    Bestj1 = [J1_Agent[i] for i in les_indices1]
    Bestj2 = [J2_Agent[i] for i in les_indices2]
    J1_Agent = Bestj1
    J2_Agent = Bestj2
    for i in range(int(nombre_agent/div)):
        for t in range(div-1):
            J1_Agent.append(clone_generator_decalage(Bestj1[i], mutation_list[t]))
            J2_Agent.append(clone_generator_decalage(Bestj2[i], mutation_list[t]))
    shuffle(J1_Agent)
    valeurs = []

"""
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
"""
# Testmodel1.summary()
# print(Testmodel1.get_weights())


sauvegarder(Bestj1, " j1 ")
sauvegarder(Bestj2, " j2 ")
b = time()
print((b-temps1)/60)
"""
Bestj1.save("best1")
Bestj2.save("best2")
"""
# tf.keras.models.load_model()
# structure du NN entrer 2 pour pos j1 2 pour pos j2 1 pour deplacement sortie 2 pour coord du deplacement
# https://blog.tensorflow.org/2021/10/building-board-game-app-with-tensorflow.html
# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
