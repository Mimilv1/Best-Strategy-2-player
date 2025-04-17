# Script pout faire du calcul
from typing import List, Tuple, Callable, Union, Any


def J(l1: List[int], l2: List[int]) -> int:
    """
    Calcule la distance entre deux points.

    Parameters:
    - l1 (List[int]): Coordonnées du premier point [x1, y1].
    - l2 (List[int]): Coordonnées du deuxième point [x2, y2].

    Returns:
    - int: Distance entre les deux points.
    """
    return abs(l1[0] - l2[0]) + abs(l1[1] - l2[1])


def P(n: int) -> int:
    """
    Calcule le nombre de cases accessibles avec un nombre de déplacements n.

    Parameters:
    - n (int): Nombre de déplacements.

    Returns:
    - int: Nombre de cases accessibles.
    """
    return 2 * n * (n + 1) + 1


def Some_pos(l1: List[int], l2: List[int]) -> List[int]:
    """
    Effectue la somme des coordonnées de deux points.

    Parameters:
    - l1 (List[int]): Coordonnées du premier point [x1, y1].
    - l2 (List[int]): Coordonnées du deuxième point [x2, y2].

    Returns:
    - List[int]: Somme des coordonnées [x1 + x2, y1 + y2].
    """
    return [l1[0] + l2[0], l1[1] + l2[1]]


def Dp(n: int) -> List[List[int]]:
    """
    Calcule tous les déplacements possibles pour un joueur avec n déplacements.

    Parameters:
    - n (int): Nombre de déplacements.

    Returns:
    - List[List[int]]: Liste des déplacements possibles.
    """
    return [[i, j] for i in range(-n, n+1) for j in range(-n + abs(i), n - abs(i) + 1)]


def Dp2(n: int) -> List[List[int]]:
    """
    Déplacements possibles du joueur 2 avec n déplacements.

    Parameters:
    - n (int): Nombre de déplacements.

    Returns:
    - List[List[int]]: Liste des déplacements possibles pour le joueur 2.
    """
    return Dp(n)


def Dp1(n: int) -> List[List[int]]:
    """
    Déplacements possibles du joueur 1 avec n déplacements.

    Parameters:
    - n (int): Nombre de déplacements.

    Returns:
    - List[List[int]]: Liste des déplacements possibles pour le joueur 1.
    """
    return Dp(n + 1)


def H(M1: List[List[int]], M2: List[List[int]], l1: List[int], l2: List[int]) -> List[List[int]]:
    """
    Calcule la distance entre les deux joueurs possible après un déplacement limité à n.

    Parameters:
    - M1 (List[List[int]]): Liste des déplacements possibles pour le joueur 1.
    - M2 (List[List[int]]): Liste des déplacements possibles pour le joueur 2.
    - l1 (List[int]): Coordonnées du joueur 1 [x1, y1].
    - l2 (List[int]): Coordonnées du joueur 2 [x2, y2].

    Returns:
    - List[List[int]]: Matrice des distances entre les déplacements possibles des joueurs.
    """
    return [[J(Some_pos(M1[j], l1), Some_pos(M2[i], l2)) for j in range(len(M1))] for i in range(len(M2))]


def Possibilite_futur(n: int, Position1: List[int], Position2: List[int]) -> List[List[int]]:
    """
    Appelle la fonction H avec les entrées d'une situation de jeu.

    Parameters:
    - n (int): Nombre de déplacements limité.
    - Position1 (List[int]): Coordonnées du joueur 1 [x1, y1].
    - Position2 (List[int]): Coordonnées du joueur 2 [x2, y2].

    Returns:
    - List[List[int]]: Résultat de la fonction H.
    """
    return H(Dp1(n), Dp2(n), Position1, Position2)


def relu(n: int) -> int:
    """
    Fonction Rectified Linear Unit (ReLU).

    Parameters:
    - n (int): Valeur à laquelle appliquer la fonction ReLU.

    Returns:
    - int: Résultat de la fonction ReLU.
    """
    return n if n >= 0 else 0


def gain2(M: List[List[int]], n: int, Position1: List[int], Position2: List[int]) -> List[List[int]]:
    """
    Calcule la matrice des gains pour le joueur 2.

    Parameters:
    - M (List[List[int]]): Matrice des distances entre les déplacements possibles des joueurs.
    - n (int): Nombre de déplacements limité.
    - Position1 (List[int]): Coordonnées du joueur 1 [x1, y1].
    - Position2 (List[int]): Coordonnées du joueur 2 [x2, y2].

    Returns:
    - List[List[int]]: Matrice des gains pour le joueur 2.
    """
    return [[j - relu(J(Position1, Position2) - 2 * n - 1) for j in i] for i in M]


def G2(n: int, Position1: List[int], Position2: List[int]) -> List[List[int]]:
    """
    Calcule la matrice des gains pour le joueur 2 en utilisant la fonction gain2.

    Parameters:
    - n (int): Nombre de déplacements limité.
    - Position1 (List[int]): Coordonnées du joueur 1 [x1, y1].
    - Position2 (List[int]): Coordonnées du joueur 2 [x2, y2].

    Returns:
    - List[List[int]]: Matrice des gains pour le joueur 2.
    """
    return gain2(Possibilite_futur(n, Position1, Position2), n, Position1, Position2)


def mat_min_max(M):
    """

    :param M: Matrice
    :return: Minimum et maximum dans la matrice
    """
    if len(M) > 0 and len(M[0]) > 1:
        mini = M[0][0]
        maxi = M[0][0]
        for i in M:
            for j in i:
                if mini > j:
                    mini = j
                if j > maxi:
                    maxi = j
    else:
        mini = 0
        maxi = 0
    return M, mini, maxi


def gain1(mat_tuple: Tuple[List[List[int]], int, int]) -> List[List[int]]:
    """
    Calcule la matrice des gains pour le joueur 1.

    Parameters:
    - mat_tuple (Tuple[List[List[int]], int, int]): Tuple contenant la matrice des gains du joueur 2 (mat_tuple[0]),
      la valeur minimale de cette matrice (mat_tuple[1]) et la valeur maximale (mat_tuple[2]).

    Returns:
    - List[List[int]]: Matrice des gains pour le joueur 1.
    """
    return [[mat_tuple[2] - mat_tuple[1] - j for j in i] for i in mat_tuple[0]]


def G1(n: int, Position1: List[int], Position2: List[int]) -> List[List[int]]:
    """
    Calcule la matrice des gains pour le joueur 1 en utilisant la fonction gain1.

    Parameters:
    - n (int): Nombre de déplacements limité.
    - Position1 (List[int]): Coordonnées du joueur 1 [x1, y1].
    - Position2 (List[int]): Coordonnées du joueur 2 [x2, y2].

    Returns:
    - List[List[int]]: Matrice des gains pour le joueur 1.
    """
    return gain1(mat_min_max(G2(n, Position1, Position2)))


def symetricG1(n: int, Position1: List[int], Position2: List[int]) -> List[List[int]]:
    """
    Calcule la matrice des gains symétrique pour le joueur 1.

    Parameters:
    - n (int): Nombre de déplacements limité.
    - Position1 (List[int]): Coordonnées du joueur 1 [x1, y1].
    - Position2 (List[int]): Coordonnées du joueur 2 [x2, y2].

    Returns:
    - List[List[int]]: Matrice des gains symétrique pour le joueur 1.
    """
    return [[-j for j in i] for i in G2(n, Position1, Position2)]


#print(G1(1, posj1, posj2))


def association(n):
    """
    Le but de cette fonction est de relier les déplacement des joueurs a la bonne ligne dans la matrice de gain

    :param n: valeur du deplacement
    :return: dictionaire entre deplacement effectuer et indice de la ligne/colonne dans la matrice de gain
    """
    table = {}
    compteur = 0
    for x in range(-n, n+1):
        for y in range(-abs(n-abs(x)), abs(n-abs(x))+1):
            table[(x, y)] = compteur
            compteur += 1
    return table


liste_table = []

for i in range(6):
    liste_table.append(association(i)) # table d'association entre les deplacements et la colone ou la ligne dans le matrice de gain avec n fixe
# print(liste_table)
