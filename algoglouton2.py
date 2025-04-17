from Calcul import *
import matplotlib.pyplot as plt
import numpy as np


def glouton(coord_j1, coord_j2, n=4, affichage=False):
    """
    :param coord_j2: emplacement du joueur 2 sous la forme [x2 ,y2]
    :param coord_j1: emplacement du joueur 1 sous la forme [x1, y1]
    :param n: nb de deplacement
    :param affichage: quand vaut true affiche les strategies sur des graphiques
    :return:
    """

    puissance_j1 = 1
    puissance_j2 = 1

    lien_dp_proba_j2 = {}
    les_deplacements_j2 = Dp(n)
    for i in range(P(n)):
        lien_dp_proba_j2[i] = tuple(les_deplacements_j2[i])
    # print(lien_dp_proba_j2[0])

    lien_dp_proba_j1 = {}

    les_deplacemetns = Dp1(n)

    for i in range(P(n+1)):
        lien_dp_proba_j1[i] = tuple(les_deplacemetns[i])
    # print(lien_dp_proba_j1[0])

    # Le joueur 1 ne va se deplacer que dans les cases accesibles par le joueur 2

    def distance_moyenne_j1(deplacement):
        """

        :param deplacement:
        :return:
        """
        somme = 0
        emplacement = Some_pos(coord_j1, deplacement)
        for i in range(P(n)):
            somme += J(emplacement, Some_pos(coord_j2, list(lien_dp_proba_j2[i])))
        return [emplacement, deplacement, somme / P(n)]

    def distance_moyenne_j2(deplacement):
        """

        :param deplacement:
        :return:
        """
        somme = 0
        emplacement = Some_pos(coord_j2, deplacement)
        compteur = 0
        for i in range(P(n+1)):
            if Some_pos(coord_j1, list(lien_dp_proba_j1[i])) in case_acessible_j2:
                somme += J(emplacement, Some_pos(coord_j1, list(lien_dp_proba_j1[i])))
                compteur += 1
        return [emplacement, deplacement, somme / compteur]

    def graphique(x, y, z):
        """

        :param x:
        :param y:
        :param z:
        :return:
        """
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(x, y, z,
                        cmap='inferno', edgecolor='black')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probabilité')
        # CMRmap  gist_rainbow, viridis , inferno

        plt.show()

    joueur1_distance_moyenne = []

    case_acessible_j2 = []
    for i in range(P(n)):
        case_acessible_j2.append(Some_pos(coord_j2, lien_dp_proba_j2[i]))
    # print(case_acessible_j2)

    for j in range(P(n+1)):
        if Some_pos(coord_j1, lien_dp_proba_j1[j]) in case_acessible_j2:
            joueur1_distance_moyenne.append(distance_moyenne_j1(list(lien_dp_proba_j1[j])))
    # print(Joueur1_distance_moyenne)

    # print(totale) # maintenant plus une valeur est basse plus elle doit avoir une grande probabilite grace a l'inverse

    joueur1_distance_moyenne_inverse = [(x[0], x[1], 1/((x[2])**puissance_j1)) for x in joueur1_distance_moyenne]

    somme_inverse = sum(x[2] for x in joueur1_distance_moyenne_inverse)
    # print(Joueur1_distance_moyenne_inverse)
    # print(somme_inverse)

    probabilite_deplacement_j1 = [[x[1], x[2]/somme_inverse] for x in joueur1_distance_moyenne_inverse]
    if affichage:
        print(probabilite_deplacement_j1)
    somme_proba = sum(x[1] for x in probabilite_deplacement_j1)
    # on reajuste en raison des erreurs de calcul en addition de float
    probabilite_deplacement_j1[1][1] += 1 - somme_proba

    somme_proba = sum(x[1] for x in probabilite_deplacement_j1)
    if affichage:
        print(somme_proba)

        x = np.array([x[0][0] for x in probabilite_deplacement_j1])
        y = np.array([x[0][1] for x in probabilite_deplacement_j1])

        z = np.array([x[1] for x in probabilite_deplacement_j1])

        graphique(x, y, z)

    """ 
    On a donc la strategie du joueur 1 dans probabilite_deplacement_j1
    on fait de meme pour j2 seule difference ici on veut la plus grande distance possible 
    """
    if affichage:
        print(joueur1_distance_moyenne)
    # deplacement
    joueur2_distance_moyenne = []
    for j in range(P(n)):
        joueur2_distance_moyenne.append(distance_moyenne_j2(list(lien_dp_proba_j2[j])))

    distance_moyenne_j2_puissance = [[x[0], x[2]**puissance_j2] for x in joueur2_distance_moyenne]

    totale = sum([x[1] for x in distance_moyenne_j2_puissance])

    probabilite_deplacement_j2 = [[x[0], x[1]/totale] for x in distance_moyenne_j2_puissance]
    if affichage:
        print(probabilite_deplacement_j2)

        print(sum([x[1] for x in probabilite_deplacement_j2]))

        probabilite_deplacement_j2[0][1] += 1 - sum([x[1] for x in probabilite_deplacement_j2])

        x = np.array([x[0][0] for x in probabilite_deplacement_j2])
        y = np.array([x[0][1] for x in probabilite_deplacement_j2])

        z = np.array([x[1] for x in probabilite_deplacement_j2])

        graphique(x, y, z)

    return [probabilite_deplacement_j1, probabilite_deplacement_j2]


# glouton([0, 0], [0, 1], affichage=True)
