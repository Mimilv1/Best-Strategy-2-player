from Strategie_opti import prochain_deplacement_j1, prochain_deplacement_j2
from Strategie_naive import prochain_deplacement_j2_naive1, prochain_deplacement_j2_naive2, prochain_deplacement_j1_naive1, prochain_deplacement_j1_naive2
from Strategie_adaptative import *
from math import sqrt
import matplotlib.pyplot as plt
tours = []
nb_partie = 0
recherche_alpha = False


def jouer_une_partie(pos_j1: List[int], pos_j2: List[int],
                     strategie1: Union[Callable[[List[int], List[int], int], Any], Any],
                     strategie2: Union[Callable[[List[int], List[int], int], Any], Any],
                     n: int, un_adaptatif: bool = False, deux_adaptatif: bool = False, alpha=10**4) -> int:
    """
    Simule le déroulement d'une partie entre deux joueurs avec des stratégies données.

    :param pos_j1: Position initiale du joueur 1 [x1, y1].
    :param pos_j2: Position initiale du joueur 2 [x2, y2].
    :param strategie1: Stratégie du joueur 1, soit sous forme de fonction ou d'objet.
    :param strategie2: Stratégie du joueur 2, soit sous forme de fonction ou d'objet.
    :param n: Nombre de déplacements.
    :param un_adaptatif: True si la stratégie1 est adaptative, False sinon.
    :param deux_adaptatif: True si la stratégie2 est adaptative, False sinon.
                          Si True, strategie2 doit être un objet de type StrategieAdaptative.
                          Sinon, strategie2 doit être une fonction à 3 paramètres pos_j1, pos_j2, et n.
    :param alpha valeur multiplicative pour la probabilité de se deplacer sur une case non accesssible par le joueur 1
    :return: Le nombre de tours jusqu'à la fin de la partie, limité à 800.
    """
    assert pos_j1 != pos_j2, "Position initiale confondue"
    tour = 0
    continuer = True
    position1 = pos_j1
    position2 = pos_j2

    while continuer and tour < 800:
        tour += 1

        if un_adaptatif:
            deplacement_j1 = deplacement_adaptatif_j1(strategie1, position1, position2, n)
            if deux_adaptatif:
                deplacement_j2 = deplacement_adaptatif_j2(strategie2, position1, position2, n, alpha=alpha)
                if deplacement_j2[1]:
                    strategie2.nouveau_deplacement(deplacement_j1[0])
                if deplacement_j1[1]:
                    strategie1.nouveau_deplacement(deplacement_j2[0])
                position2 = Some_pos(position2, deplacement_j2[0])
            else:
                deplacement_j2 = strategie2(position1, position2, n)
                if deplacement_j1[1]:
                    strategie1.nouveau_deplacement(deplacement_j2)
                position2 = Some_pos(position2, deplacement_j2)

            position1 = Some_pos(position1, deplacement_j1[0])
        else:
            deplacement_j1 = strategie1(position1, position2, n)

            if deux_adaptatif:
                deplacement_j2 = deplacement_adaptatif_j2(strategie2, position1, position2, n, alpha=alpha)
                if deplacement_j2[1]:
                    strategie2.nouveau_deplacement(deplacement_j1)
                position2 = Some_pos(position2, deplacement_j2[0])
            else:
                deplacement_j2 = strategie2(position1, position2, n)
                position2 = Some_pos(position2, deplacement_j2)
            position1 = Some_pos(position1, deplacement_j1)
        if position1 == position2:
            continuer = False

    return tour


def lancer_partie(pos_j1: List[int], pos_j2: List[int], strategie1, strategie2,
                     n: int, un_adaptatif: bool = False, deux_adaptatif: bool = False) -> int:
    """
    Sert a utlisier les init adaptatif pour avoir des nouveaux objets dans la memoire
    :param pos_j1:
    :param pos_j2:
    :param strategie1:
    :param strategie2:
    :param n:
    :param un_adaptatif:
    :param deux_adaptatif:
    :return:
    """
    if un_adaptatif:
        j1 = strategie1(n)
    else:
        j1 = strategie1
    if deux_adaptatif:
        j2 = strategie2(n)
    else:
        j2 = strategie2
    return jouer_une_partie(pos_j1, pos_j2, j1, j2, n, un_adaptatif=un_adaptatif, deux_adaptatif=deux_adaptatif)


def ecartype(tableau):
    moy = sum(tableau)/len(tableau)
    somme = 0
    for i in tableau:
        somme += (i-moy)**2
    return sqrt(somme/len(tableau))


if recherche_alpha:
    valeur_prise_par_alpha = [10**i/3*j for i in range(10) for j in range(1, 4)]
    for c in valeur_prise_par_alpha:
        print(c)
        result = []
        for i in range(nb_partie):
            n = 4
            posj1 = [0, 0]
            posj2 = [2, 0]
            nb_tour = jouer_une_partie(posj1, posj2, StrategieAdaptative(Dp2(4), 1, 4), StrategieAdaptative(Dp1(4), 2, 4), 4, un_adaptatif=True, deux_adaptatif=True, alpha=c) # [135.88, 347.24, 502.55, 542.56, 641.15, 570.54, 552.66, 530.09, 541.0, 607.71]
            result.append(nb_tour)
        tours.append(sum(result)/len(result))

    plt.plot(np.log10(np.array(valeur_prise_par_alpha)), tours)
    plt.show()

for i in range(nb_partie):
    n = 4
    posj1 = [0, 1]
    posj2 = [0, 0]
    nb_tour = jouer_une_partie(posj1, posj2, StrategieAdaptative(Dp2(4), 1, 4), prochain_deplacement_j2, 4, un_adaptatif=True, deux_adaptatif=False)
    tours.append(nb_tour)


if nb_partie != 0:
    print("Donnees :", tours)
    print("max et min :", max(tours), min(tours))
    print("Moyenne :", sum(tours)/len(tours))
    tours.sort()
    print("Mediane :", tours[nb_partie//2])

    print("Ecart type :", ecartype(tours))


# RESULTAT CONTRE ADAPTATIFJ1 prochain_deplacement_j2_naive2 69
# RESULTAT CONTRE ADAPTATIFJ1 ADAPTATIF 69
# RESULTAT CONTRE ADAPTATIFJ1 prochain_deplacement_j2_naive1 75
# RESULTAT CONTRE ADAPTATIFJ1 prochain_deplacement_j2 81


Strategies1 = [(prochain_deplacement_j1_naive1, 0), (prochain_deplacement_j1_naive2, 1),
               (prochain_deplacement_j1, 2), (init_adaptatif_j1, 3)]


Strategies2 = [(prochain_deplacement_j2_naive1,  4), (prochain_deplacement_j2_naive2, 5),
               (prochain_deplacement_j2, 6), (init_adaptatif_j2, 7)]


Joueurs1 = Strategies1
Joueurs2 = Strategies2

rounds = 0
nb_afrontements = 400
if rounds!=0:
    couleur = ["blue", "purple", "orange", "red"]
    fig, ax = plt.subplots()
    labels = ['Naïve 1', 'Naïve 2', 'Glouton', 'Adaptative']
    JT = [[] for i in range(len(Joueurs1))]
    x = np.arange(len(labels))  # the label locations
    width = 0.45  # the width of the bars
    for i in range(rounds):
        resultat = []
        compteur = 0
        for j in Joueurs1:
            for t in Joueurs2:
                somme = 0
                for r in range(nb_afrontements):
                    somme += lancer_partie([0, 0], [2, 0], j[0], t[0], 4, un_adaptatif=j[1] == 3, deux_adaptatif=t[1] == 7)
                resultat.append((j, t, somme / nb_afrontements))
                JT[compteur].append(somme/nb_afrontements)
            compteur += 1
        for k in resultat:
            print(k[0][1], k[1][1], k[2])


    print(JT[0])
    print(JT[1])
    print(JT[2])
    print(JT[3])
    Result_J1 = ax.bar(x - width*3/4, JT[0], width/2, label='Naïve 1')
    Result_J2 = ax.bar(x - width/4, JT[1], width/2, label='Naïve 2')
    Result_J3 = ax.bar(x + width/4, JT[2], width/2, label='Glouton')
    Result_J4 = ax.bar(x + width*3/4, JT[3], width/2, label='Adaptative')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Nombre moyen de deplacement', fontsize=25)
    ax.set_title('Resultats', fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=25)
    plt.grid(True)
    ax.legend(fontsize=25)


    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(Result_J1)
    autolabel(Result_J2)
    autolabel(Result_J3)
    autolabel(Result_J4)

    fig.tight_layout()

    plt.show()
