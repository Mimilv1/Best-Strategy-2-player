"""
On implémente ici la strategie optimale determiner par algoglouton2
"""
from Calcul import *
from random import random
from algoglouton2 import glouton

n = 4

"""
On va calculer tous les cas ou la distance est inferieur ou egale a n
"""
lien_deplacement_j1 = {}
lien_deplacement_j2 = {}

les_deplacemetns = Dp2(n)

Liste_deplacement_proba_j1 = []
Liste_deplacement_proba_j2 = []

max_pour_j2 = P(n)
for i in range(P(n)):
    predict = glouton(les_deplacemetns[i], [0, 0], n=4)
    lien_deplacement_j1[tuple(les_deplacemetns[i])] = i
    Liste_deplacement_proba_j1.append(predict[0])

    lien_deplacement_j2[tuple(les_deplacemetns[i])] = i
    Liste_deplacement_proba_j2.append(predict[1])


def choix(loi):
    """
    Prend une liste de nombre positif dont la somme vaut 1 et
    renvoie un indice de la liste selon la distribution de probabilite
    """
    s = 0
    val = random()
    for i in range(len(loi)):
        proba = loi[i]
        s += proba
        if s > val:
            return i
        if i == len(loi) - 1:
            print(loi)
            print(s)
            print(val)
            print("error la somme des proba ne vaut pas 1")


def joueur1_deplacement_grande_distance(pos_j1, pos_j2, nb_dp):
    """

    :param pos_j1:
    :param pos_j2:
    :param nb_dp:
    :return:
    """
    x1, y1 = pos_j1
    x2, y2 = pos_j2
    for i in range(nb_dp + 1):
        if x1 - x2 > 0:
            x1 -= 1
        elif x1 - x2 < 0:
            x1 += 1
        elif y1 - y2 > 0:
            y1 -= 1
        else:
            y1 += 1
    return [x1 - pos_j1[0], y1 - pos_j1[1]]


def joueur2_deplacement_grande_distance(pos_j1, pos_j2, nb_dp):
    """

    :param pos_j1:
    :param pos_j2:
    :param nb_dp:
    :return:
    """
    x1, y1 = pos_j1
    x2, y2 = pos_j2
    deplacement = [0, 0]
    if x1 > x2:
        deplacement[0] -= nb_dp
    elif x1 < x2:
        deplacement[0] += nb_dp
    elif y1 > y2:
        deplacement[1] -= nb_dp
    else:
        deplacement[1] += nb_dp
    return deplacement


def prochain_deplacement_j1(position_joueur_1, position_joueur_2, nb_dp):
    """
    :param position_joueur_1: [x1, y1] position du joueur 1
    :param position_joueur_2:
    :param nb_dp:
    :return: [x, y] avec x et y entier le deplacement optimale calculer
    """
    distance = J(position_joueur_1, position_joueur_2)
    x1, y1 = position_joueur_1
    x2, y2 = position_joueur_2
    if distance > nb_dp:
        return joueur1_deplacement_grande_distance(position_joueur_1, position_joueur_2, nb_dp)
    else:
        nouvelle_coord_j1 = [x1-x2, y1-y2]
        proba_j1 = Liste_deplacement_proba_j1[lien_deplacement_j1[tuple(nouvelle_coord_j1)]]
        indice = choix([proba_j1[i][1] for i in range(len(proba_j1))])
        return proba_j1[indice][0]


def prochain_deplacement_j2(position_joueur_1, position_joueur_2, nb_dp):
    """
    :param position_joueur_1: [x1, y1] position du joueur 1
    :param position_joueur_2:
    :param nb_dp:
    :return: [x, y] avec x et y entier le deplacement optimale calculer
    """
    distance = J(position_joueur_1, position_joueur_2)
    x1, y1 = position_joueur_1
    x2, y2 = position_joueur_2
    if distance > nb_dp:
        return joueur2_deplacement_grande_distance(position_joueur_1, position_joueur_2, nb_dp)
    else:
        nouvelle_coord_j1 = [x1-x2, y1-y2]
        proba_j2 = Liste_deplacement_proba_j2[lien_deplacement_j2[tuple(nouvelle_coord_j1)]]
        indice = choix([proba_j2[i][1] for i in range(len(proba_j2))])
        return proba_j2[indice][0]


# print(prochain_deplacement_j1([-4, 0], [0, 0], 4))
# print(prochain_deplacement_j2([-4, 0], [0, 0], 4))
