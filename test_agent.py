# Créé par emili, le 03/05/2023 en Python 3.7

import tensorflow as tf
import numpy as np
from random import randint, shuffle, uniform
from Calcul import G1, G2, J
from keras.layers.advanced_activations import PReLU

model = tf.keras.models.load_model("Agent j1 11")
model2 = tf.keras.models.load_model("Agent j2 10")


def rectification(x, y, n):
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
    action_log_j1 = []
    action_log_j2 = []
    victoire_j1 = False
    pos1 = [0, 12]
    pos2 = [0, 0]
    for i in range(40):
        n = randint(0, 4)
        sortie1 = joueur1.predict(np.array([[n, pos1[0], pos1[1], pos2[0], pos2[1]]]))
        sortie2 = joueur2.predict(np.array([[n, pos1[0], pos1[1], pos2[0], pos2[1]]]))
        d1x, d1y, d2x, d2y = sortie1[0][0], sortie1[0][1], sortie2[0][0], sortie2[0][1]
        d1x = int(d1x*5)
        d1y = int(d1y*5)
        d2x = int(d2x*5)
        d2y = int(d2y*5)
        d1x, d1y = rectification(d1x, d1y, n + 1)
        d2x, d2y = rectification(d2x, d2y, n)
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


print(play_game(model, model2))
