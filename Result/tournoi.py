import tensorflow as tf
import numpy as np
from random import randint, shuffle, uniform
from Calcul import G1, G2, J, liste_table

from time import time

J1 = []
J2 = []

gain1 = [0 for i in range(40)]
gain2 = [0 for i in range(40)]

for i in range(40):
    J1.append(tf.keras.models.load_model("Agent J1 " + str(i)))
    J2.append(tf.keras.models.load_model("Agent J2 " + str(i)))


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
    for i in range(50):
        n = 4
        table_gain1 = G1(n, pos1, pos2)
        table_gain2 = G2(n, pos1, pos2)
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
        gain_du_joueur_1 += table_gain1[liste_table[n][(d2x, d2y)]][liste_table[n+1][(d1x, d1y)]]
        gain_du_joueur_2 += table_gain2[liste_table[n][(d2x, d2y)]][liste_table[n+1][(d1x, d1y)]]
        # action_log_j1.append([d1x, d1y])
        # action_log_j2.append([d2x, d2y])
        if pos1 == pos2:
            victoire_j1 = True
            gain_du_joueur_1 = 901  # sur de passer
            break
    distance_finale = J(pos1, pos2)  # J calcul, la distance entre deux points
    return "action_log_j1", "action_log_j2", victoire_j1, distance_finale, gain_du_joueur_1, gain_du_joueur_2


for i in range(40):
    print(i)
    j1 = J1[i]
    for j in range(40):
        j2 = J2[j]
        _, _, _, _, g1, g2 = play_game(j1, j2)
        gain1[i] += g1
        gain2[j] += g2

print(gain1)
print(gain2)
print(max(gain1), max(gain2))
m=0
n=0
for i in range(40):
    if gain1[i]>gain1[m]:
        m=i
    if gain2[i]>gain2[n]:
        n=i
print(m, n)
