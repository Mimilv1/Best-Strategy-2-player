# Algorithme glouton pour les stratégies des joueurs 1 et 2
from Calcul import *
from random import randint, random


def main():
    coord_j2 = [0, 0]
    coord_j1 = [0, 3]

    n = 4

    strategie_proba_j2 = [1 / P(n) for i in range(P(n))]
    print(sum(strategie_proba_j2))

    lien_dp_proba_j2 = {}
    les_deplacemetns = Dp(n)
    for i in range(P(n)):
        lien_dp_proba_j2[i] = tuple(les_deplacemetns[i])
    print(lien_dp_proba_j2[0])

    lien_dp_proba_j1 = {}

    les_deplacemetns = Dp1(n)
    strategie_proba_j1 = [1 / P(n + 1) for i in range(P(n + 1))]

    print(sum(strategie_proba_j1))

    for i in range(P(n + 1)):
        lien_dp_proba_j1[i] = tuple(les_deplacemetns[i])
    print(lien_dp_proba_j1[0])

    def somme_de_liste(l1, l2):
        """
        Prend deux liste d'entier de meme taille
        Renvoie l'addition des indices identiques
        """
        l = []
        for i in range(len(l1)):
            l.append(l1[i] + l2[i])
        return l

    def choix(loi):
        """
        Prend une liste de nombre positif dont la somme vaut 1 et
        renvoie un indice de la liste selon la distribution de probabilite
        """
        rectifieur_float(loi)
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
                print("eror")

    def somme_terme(l):
        s = 0
        for i in l:
            s += i
        return s

    def rectifieur_float(liste):
        val = somme_terme(liste)
        diff = 1 - val
        ajout = diff / len(liste)
        for i in range(len(liste)):
            liste[i] += ajout

    def diminuer(liste, reste):
        """
        liste de proba un reste
        on veut retirer reste de la liste de maniere uniforme
        """
        nb_sans_zero = 0
        for i in range(len(liste)):
            if liste[i] != 0:
                nb_sans_zero += 1

        val = reste / nb_sans_zero
        reste_de_reste = 0
        for i in range(len(liste)):
            if liste[i] != 0:
                if liste[i] >= val:
                    if liste[i] >= val + reste_de_reste:
                        liste[i] -= val + reste_de_reste
                    else:
                        liste[i] -= val
                else:
                    reste_de_reste += val - liste[i]
                    liste[i] = 0

    def reduire(liste, indice, rate):
        reste = 0
        longeur = len(liste)
        val = rate / (longeur - 1)
        for i in range(longeur):
            if i != indice and liste[i] >= val:
                liste[i] -= val
            elif i != indice and liste[i] < val:
                res = val - liste[i]
                liste[i] = 0
                reste += res

        diminuer(liste, reste)
        # il faut retirer de liste les restes

    def match(strat_j1, strat_j2):
        rate = 0.00001
        distance_initiale = J(coord_j1, coord_j2)
        choix_1 = choix(strat_j1)
        choix_2 = choix(strat_j2)
        deplacement_1 = lien_dp_proba_j1[choix_1]
        deplacement_2 = lien_dp_proba_j2[choix_2]
        if J(somme_de_liste(coord_j1, deplacement_1), somme_de_liste(coord_j2, deplacement_2)) > distance_initiale:
            if strat_j2[choix_2] + rate <= 1:
                strat_j2[choix_2] += rate
                reduire(strat_j2, choix_2, rate)
                if strat_j1[choix_1] - rate >= 0:
                    strat_j1[choix_1] -= rate
                    reduire(strat_j1, choix_1, -rate)
                else:
                    reste = rate - strat_j1[choix_1]
                    strat_j1[choix_1] = 0
                    diminuer(strat_j1, reste)

        elif J(somme_de_liste(coord_j1, deplacement_1), somme_de_liste(coord_j2, deplacement_2)) < distance_initiale:
            if strat_j1[choix_1] + rate <= 1:
                strat_j1[choix_1] += rate
                reduire(strat_j1, choix_1, rate)
                if strat_j2[choix_2] - rate >= 0:
                    strat_j2[choix_2] -= rate
                    reduire(strat_j2, choix_2, -rate)
                else:
                    reste = rate - strat_j2[choix_2]
                    strat_j2[choix_2] = 0
                    diminuer(strat_j2, reste)

    for i in range(10 ** 5):
        match(strategie_proba_j1, strategie_proba_j2)

    print(strategie_proba_j1)
    print(somme_terme(strategie_proba_j1))
    for i in range(len(strategie_proba_j1)):
        if strategie_proba_j1[i] > 0.05:
            print(lien_dp_proba_j1[i], end=" , ")
    print()
    print()
    print(strategie_proba_j2)
    print(somme_terme(strategie_proba_j2))
    for i in range(len(strategie_proba_j2)):
        if strategie_proba_j2[i] > 0.05:
            print(lien_dp_proba_j2[i], end=" , ")


main()
