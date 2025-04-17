from Calcul import *
from Strategie_opti import joueur1_deplacement_grande_distance, joueur2_deplacement_grande_distance, choix
import numpy as np
import matplotlib.pyplot as plt
# from spicy.spatial import Delaunay


def prochain_deplacement_j1_naive1(position_joueur_1: List[int], position_joueur_2: List[int], nb_dp: int, affichage: bool = False) -> List[int]:
    """
    strategie se deplacer a l'ancien emplacement du joueur 2

    :param position_joueur_1: [x1, y1] avec x1 y1  qui sont la position du joueur 1 en int
    :param position_joueur_2: [x2, y2] avec x2 y2  qui sont la position du joueur 2 en int
    :param nb_dp: nombre de deplacement authorisee
    :param affichage: False de base si mis a True alors representation graphique de la strategie
    :return: le prochain deplacement
    """
    distance = J(position_joueur_1, position_joueur_2)
    if distance > nb_dp:
        return joueur1_deplacement_grande_distance(position_joueur_1, position_joueur_2, nb_dp)
    else:
        if affichage:
            affichage_strategie(prochain_deplacement_j1_naive1, position_joueur_1, position_joueur_2, nb_dp, numero=1)
        return [position_joueur_2[0] - position_joueur_1[0], position_joueur_2[1] - position_joueur_1[1]]


def prochain_deplacement_j2_naive1(position_joueur_1: List[int], position_joueur_2: List[int], nb_dp: int, affichage: bool = False) -> List[int]:
    """
    strategie s'eloigner le plus possible du joueur 1 sur x

    :param position_joueur_1: [x1, y1] avec x1 y1  qui sont la position du joueur 1 en int
    :param position_joueur_2: [x2, y2] avec x2 y2  qui sont la position du joueur 2 en int
    :param nb_dp: nombre de deplacement authorisee
    :param affichage: False de base si mis a True alors representation graphique de la strategie
    :return: le prochain deplacement
    """
    distance = J(position_joueur_1, position_joueur_2)
    if distance > nb_dp:
        return joueur2_deplacement_grande_distance(position_joueur_1, position_joueur_2, nb_dp)
    else:
        if affichage:
            affichage_strategie(prochain_deplacement_j2_naive1, position_joueur_1, position_joueur_2, nb_dp, numero=2)
        if position_joueur_2[0]-position_joueur_1[0] > 0:
            return [nb_dp, 0]
        else:
            return [-nb_dp, 0]


def prochain_deplacement_j2_naive2(position_joueur_1: List[int], position_joueur_2: List[int], nb_dp: int, affichage: bool = False) -> List[int]:
    """
    strategie se deplacer aleatoirement quand on est a proximite de j1
    :param affichage: False de base si mis a True alors representation graphique de la strategie
    :param position_joueur_1: [x1, y1] avec x1 y1  qui sont la position du joueur 1 en int
    :param position_joueur_2: [x2, y2] avec x2 y2  qui sont la position du joueur 2 en int
    :param nb_dp: nombre de deplacement authorisee
    :return: le prochain deplacement
    """
    distance = J(position_joueur_1, position_joueur_2)
    if distance > nb_dp:
        return joueur2_deplacement_grande_distance(position_joueur_1, position_joueur_2, nb_dp)
    else:
        taille = len(Dp2(nb_dp))
        inverse = 1/taille
        if affichage:
            tab = [[i, 1/taille] for i in Dp2(nb_dp)]
            x = np.array([i[0][0] for i in tab])
            y = np.array([i[0][1] for i in tab])
            z = np.array([i[1] for i in tab])
            graphique(x, y, z)
        return Dp2(nb_dp)[choix([inverse for i in range(taille)])]


def prochain_deplacement_j1_naive2(position_joueur_1: List[int], position_joueur_2: List[int], nb_dp: int, affichage: bool = False) -> List[int]:
    """
    strategie se deplacer aleatoirement sur une case accessible par le joueur 2

    :param position_joueur_1: [x1, y1] avec x1 y1  qui sont la position du joueur 1 en int
    :param position_joueur_2: [x2, y2] avec x2 y2  qui sont la position du joueur 2 en int
    :param nb_dp: nombre de deplacement authorisee
    :param affichage: False de base si mis a True alors representation graphique de la strategie
    :return: le prochain deplacement
    """
    distance = J(position_joueur_1, position_joueur_2)
    if distance > nb_dp:
        return joueur1_deplacement_grande_distance(position_joueur_1, position_joueur_2, nb_dp)
    else:
        deplacement_possible_j2 = Dp2(nb_dp)
        deplacement_possible_j1 = Dp1(nb_dp)
        position_possble_j2 = [Some_pos(position_joueur_2, i) for i in deplacement_possible_j2]
        deplacement_j1 = []
        for i in deplacement_possible_j1:
            if Some_pos(position_joueur_1, i) in position_possble_j2:
                deplacement_j1.append(i)
        taille = len(deplacement_j1)
        inverse = 1/taille
        if affichage:
            tab = [[i, inverse if i in deplacement_j1 else 0] for i in Dp1(nb_dp)]
            x = np.array([i[0][0] for i in tab])
            y = np.array([i[0][1] for i in tab])
            z = np.array([i[1] for i in tab])
            graphique(x, y, z)
        return deplacement_j1[choix([inverse for i in range(taille)])]


# Fonctions d'affichages :


def graphique(x: np.ndarray, y: np.ndarray, z: np.ndarray, valeur_min: float = 0.1 + 0.2):
    """
    affiche graphiquement la distribution de probabilite selon x,y

    :param x: deplacement en x
    :param y: deplacement en y
    :param z:tableau qui contient les proba en fonction des deplacement
    :param valeur_min : permet de changer l'affichage utile si toutes les valeurs sont proches
    """
    ax = plt.axes(projection='3d')
    if min(z) == max(z):
        ax.plot_trisurf(x, y, z,
                         cmap='inferno', edgecolor='black', vmin=min(z)-0.01, vmax=max(z)+0.01)
    else:
        if valeur_min != 0.1+0.2:
            ax.plot_trisurf(x, y, z,
                        cmap='inferno', edgecolor='black', vmin=valeur_min)  # sipmlices=Delaunay(points2D)
        else:
            ax.plot_trisurf(x, y, z,
                            cmap='inferno', edgecolor='black')
    # CMRmap  gist_rainbow, viridis , inferno
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probabilité')
    plt.show()


def affichage_strategie(strategie: Callable[[List[int], List[int], int], List[int]],
                        pos_j1: List[int], pos_j2: List[int], nb_dp: int, numero: int = 2):
    """
    affiche une strategie en faisant un nombre consequent d'appel sur la meme entree
    :param strategie: une strategie qui est une fonction qui renvoie un deplacement en prenant 3 entrees
    :param pos_j1: position du joueur 1 sous la forme [x1, y1]
    :param pos_j2: position du joueur 2 sous la forme [x2, y2]
    :param nb_dp: nombre de deplacement
    :param numero: indique si c'est une strategie du joueur 1 ou du joueur 2 de base sur 2
    """

    nb_simulation = 50000
    data = [strategie(pos_j1, pos_j2, nb_dp) for i in range(nb_simulation)]

    # Permet d'ajouter les evenements impossible sur le schema :
    ajout = 0
    if numero == 1:
        ajout = 1
    for i in Dp(nb_dp+ajout):
        data.append(i)

    deplacement = []
    for i in data:
        if not(i in deplacement):
            deplacement.append(i)

    proba = [[i, data.count(i)/nb_simulation] for i in deplacement]
    x = np.array([x[0][0] for x in proba])

    y = np.array([x[0][1] for x in proba])

    z = np.array([x[1] for x in proba])

    graphique(x, y, z)


"""
affichage_strategie(prochain_deplacement_j1_naive1, [0, 0], [0, 2], 4, numero=1)
affichage_strategie(prochain_deplacement_j1_naive2, [0, 0], [0, 2], 4, numero=1)
affichage_strategie(prochain_deplacement_j2_naive1, [0, 0], [0, 2], 4, numero=2)
affichage_strategie(prochain_deplacement_j2_naive2, [0, 0], [0, 2], 4, numero=2)
"""
"""
prochain_deplacement_j2_naive2([0, 0], [0, 2], 4, affichage=True)
prochain_deplacement_j2_naive1([0, 0], [0, 2], 4, affichage=True)
prochain_deplacement_j1_naive1([0, 0], [0, 2], 4, affichage=True)
prochain_deplacement_j1_naive2([0, 0], [0, 2], 4, affichage=True)
"""
