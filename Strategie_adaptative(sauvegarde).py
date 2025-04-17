"""
Le but est de creer une strategie qui s'adapte aux deplacements de l'autre joueur
"""
import numpy as np
from Calcul import *
from Strategie_opti import joueur1_deplacement_grande_distance, joueur2_deplacement_grande_distance, choix
from Strategie_naive import graphique
import matplotlib.pyplot as plt


class CaseMemoire(object):
    """
    Permet de compter le nombre de fois que l'adversaire choisi ce deplacement
    """
    def __init__(self, valeur):
        """

        :param valeur: de la forme [x, y]
        """
        self.deplacement = valeur
        self.compteur = 1

    def augmenter_compteur(self):
        self.compteur += 1

    def get_compteur(self):
        return self.compteur

    def get_x(self):
        return self.deplacement[0]

    def get_y(self):
        return self.deplacement[1]

    def get_deplacement(self):
        return self.deplacement


class StrategieAdaptative(object):
    def __init__(self, tab, joueur, n):
        """
        :param tab: liste des deplacements possible du joueur adverse
        :param joueur: numero du joueur
        :param n: nombre de deplacement
        """
        assert joueur == 1 or joueur == 2, "Mauvais numero de joueur"

        self.n = n
        self.joueur = joueur
        self.data = []
        taille = len(tab)
        self.association = {tuple(tab[i]): i for i in range(taille)}
        self.nb_deplacement_memoire = taille
        for deplacement in tab:
            self.data.append(CaseMemoire(deplacement))

    def deplacement_a_tab(self, deplacement):
        return self.data[self.association[tuple(deplacement)]]

    def nouveau_deplacement(self, deplacement):
        """
        Procedure pour augmenter la valeur d'un deplacement dans la case memoire
        :param deplacement:
        """
        self.deplacement_a_tab(deplacement).augmenter_compteur()
        self.nb_deplacement_memoire += 1

    def afficher_stockage(self):
        """
        affiche la strategie de l'adversaire jouer pour le moment
        """
        x = np.array([x.get_x() for x in self.data])
        y = np.array([x.get_y() for x in self.data])

        z = np.array([x.get_compteur() for x in self.data])
        valeur_min = 0
        ax = plt.axes(projection='3d')
        if min(z) == max(z):
            ax.plot_trisurf(x, y, z,
                            cmap='inferno', edgecolor='black', vmin=min(z) - 0.01, vmax=max(z) + 0.01)
        else:
            if valeur_min != 0.1 + 0.2:
                ax.plot_trisurf(x, y, z,
                                cmap='inferno', edgecolor='black', vmin=valeur_min)  # sipmlices=Delaunay(points2D)
            else:
                ax.plot_trisurf(x, y, z,
                                cmap='inferno', edgecolor='black')
        # CMRmap  gist_rainbow, viridis , inferno
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('strategie en memoire ' + str(self.joueur))
        plt.show()

    def calcul_strategie(self, position_moi: list, position_adversaire: list, affichage=False) -> list:
        """

        :param position_moi: position du joueur
        :param position_adversaire: position du joueur adverse
        :param affichage: affichage ou non de la distribution de probabilite
        :return: deplacement à jouer
        """
        if self.joueur == 1:
            case_acessible = [Some_pos(position_adversaire, i) for i in Dp2(self.n)]
            deplacement_interessant = []

            for i in Dp1(self.n):
                nouvelle_coordone = Some_pos(position_moi, i)
                if nouvelle_coordone in case_acessible:
                    proba = self.deplacement_a_tab(Some_pos(nouvelle_coordone, [-a for a in position_adversaire])).get_compteur()/self.nb_deplacement_memoire
                    deplacement_interessant.append([i, proba])
            deplacement_possible = [i[0] for i in deplacement_interessant]

            for i in Dp2(self.n):  # il faut prendre en compte les cases non acessible par le joueur 1
                nouvelle_coordone = Some_pos(position_adversaire, i)
                if not(Some_pos(nouvelle_coordone, [-a for a in position_moi]) in deplacement_possible):
                    proba = self.deplacement_a_tab(i).get_compteur()/self.nb_deplacement_memoire
                    stop = False
                    for j in range(1, self.n):
                        for k in Dp2(j):
                            case_possible = Some_pos(Some_pos(k, Some_pos(i, position_adversaire)), [-a for a in position_moi])
                            if case_possible in deplacement_possible:
                                for l in deplacement_interessant:
                                    if l[0] == case_possible:
                                        l[1] += proba
                                        stop = True
                                        break
                            if stop:
                                break
                        if stop:
                            break

            indice = choix([prob[1] for prob in deplacement_interessant])
            if affichage:
                self.afficher_stockage()
                x = np.array([x[0][0] for x in deplacement_interessant])
                y = np.array([x[0][1] for x in deplacement_interessant])

                z = np.array([x[1] for x in deplacement_interessant])

                graphique(x, y, z, valeur_min=0)
            return deplacement_interessant[indice][0]
        else:
            deplacement_possible = Dp2(self.n)
            case_accessible = [Some_pos(i, position_moi) for i in deplacement_possible]
            deplacement_interessant = []
            for i in deplacement_possible:
                nouvelle_coordone = Some_pos(position_moi, i)
                distance_moyenne = 0
                for k in Dp1(self.n):
                    nouvelle_position_adversaire = Some_pos(k, position_adversaire)
                    if nouvelle_position_adversaire in case_accessible:
                        distance_moyenne += J(nouvelle_coordone, nouvelle_position_adversaire) / (self.deplacement_a_tab(k).get_compteur())
                    else:
                        distance_moyenne += J(nouvelle_coordone, nouvelle_position_adversaire)
                deplacement_interessant.append([i, distance_moyenne])

            somme_distance = np.sum([i[1] for i in deplacement_interessant])
            for i in deplacement_interessant:
                i[1] = i[1]/somme_distance

            indice = choix([prob[1] for prob in deplacement_interessant])
            if affichage:
                self.afficher_stockage()
                x = np.array([x[0][0] for x in deplacement_interessant])
                y = np.array([x[0][1] for x in deplacement_interessant])
                z = np.array([x[1] for x in deplacement_interessant])

                graphique(x, y, z, valeur_min=0)

            return deplacement_interessant[indice][0]


def deplacement_adaptatif_j1(joueur_adaptatif, position_joueur_1, position_joueur_2, nb_dp):
    """
    Détermine le déplacement du joueur 1 de manière adaptative en fonction de la stratégie adaptative.

    :param joueur_adaptatif: Instance de la classe StrategieAdaptative
    :param position_joueur_1: Position actuelle du joueur 1 sous la forme [x1, y1]
    :param position_joueur_2: Position actuelle du joueur 2 sous la forme [x2, y2]
    :param nb_dp: Nombre de déplacements
    :return: Déplacement du joueur 1 sous la forme [x1, y1], et un booléen indiquant si le joueur a effectué un nouveau déplacement
    """
    if J(position_joueur_1, position_joueur_2) > nb_dp:
        return joueur1_deplacement_grande_distance(position_joueur_1, position_joueur_2, nb_dp), False
    else:
        return joueur_adaptatif.calcul_strategie(position_joueur_1, position_joueur_2, affichage=True), True


def deplacement_adaptatif_j2(joueur_adaptatif, position_joueur_1, position_joueur_2, nb_dp):
    """
    Détermine le déplacement du joueur 2 de manière adaptative en fonction de la stratégie adaptative.

    :param joueur_adaptatif: Instance de la classe StrategieAdaptative
    :param position_joueur_1: Position actuelle du joueur 1 sous la forme [x1, y1]
    :param position_joueur_2: Position actuelle du joueur 2 sous la forme [x2, y2]
    :param nb_dp: Nombre de déplacements
    :return: Déplacement du joueur 2 sous la forme [x2, y2]
    """
    if J(position_joueur_1, position_joueur_2) > nb_dp:  # on fait le choix de ce deplacer sur les cases
        return joueur2_deplacement_grande_distance(position_joueur_1, position_joueur_2, nb_dp), False
    else:
        return joueur_adaptatif.calcul_strategie(position_joueur_2, position_joueur_1, affichage=True), True


def init_adaptatif_j1(n):
    """
    Initialise une instance de la classe StrategieAdaptative pour le joueur 1.

    :param n: Nombre de déplacements
    :return: Instance de la classe StrategieAdaptative pour le joueur 1
    """
    return StrategieAdaptative(Dp2(n), 1, n)


def init_adaptatif_j2(n):
    """
    Initialise une instance de la classe StrategieAdaptative pour le joueur 2.

    :param n: Nombre de déplacements
    :return: Instance de la classe StrategieAdaptative pour le joueur 2
    """
    return StrategieAdaptative(Dp1(n), 2, n)
