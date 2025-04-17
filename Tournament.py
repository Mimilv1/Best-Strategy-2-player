from Strategie_naive import prochain_deplacement_j2_naive1, prochain_deplacement_j2_naive2,\
    prochain_deplacement_j1_naive2, prochain_deplacement_j1_naive1
from Strategie_opti import prochain_deplacement_j1, prochain_deplacement_j2
from Strategie_adaptative import *
from Test_strategies_optimale import lancer_partie


def affichage_joueurs(Joueurs):
    color_start = ["\033[93m", "\033[92m", "\033[94m", "\033[95m", "\033[96m", "\033[91m", "\033[97m", "\033[98m"]
    color_end = "\033[00m"
    for i in Joueurs:
        print(color_start[i[1]], i[1], color_end, end="  ")
    print()


Strategies1 = [(prochain_deplacement_j1_naive1, 0), (prochain_deplacement_j1_naive2, 1),
               (prochain_deplacement_j1, 4), (init_adaptatif_j1, 6)]


Strategies2 = [(prochain_deplacement_j2_naive1,  2), (prochain_deplacement_j2_naive2, 3),
               (prochain_deplacement_j2, 5), (init_adaptatif_j2, 7)]


Joueurs1 = [i for i in Strategies1 for j in range(1)]
Joueurs2 = [i for i in Strategies2 for a in range(1)]
affichage_joueurs(Joueurs1)
affichage_joueurs(Joueurs2)
rounds = 15
nb_afrontements = 200

for i in range(rounds):
    resultat = []
    compteur = 0
    for j in Joueurs1:
        for t in Joueurs2:
            compteur += 1
            somme = 0
            for r in range(nb_afrontements):
                somme += lancer_partie([0, 0], [1, 0], j[0], t[0], 4, un_adaptatif=j[1] > 5, deux_adaptatif=t[1] > 5)
            resultat.append((j, t, somme / nb_afrontements))
            print(compteur, 16)
    for k in range(1):
        minimum = min(resultat, key=lambda tab: tab[2])
        maximum = max(resultat, key=lambda tab: tab[2])
        Joueurs2.remove(minimum[1])
        Joueurs2.append(maximum[1])
        Joueurs1.remove(maximum[0])
        Joueurs1.append(minimum[0])
        if Joueurs2.count(minimum[1]) == 0:
            boucle = len(resultat)
            accumulateur = 0
            for h in range(boucle):
                if resultat[h-accumulateur][1] == minimum[1]:
                    resultat.remove(resultat[h-accumulateur])
                    accumulateur += 1
        if Joueurs1.count(maximum[0]) == 0:
            boucle = len(resultat)
            accumulateur = 0
            for h in range(boucle):
                if resultat[h-accumulateur][0] == maximum[0]:
                    resultat.remove(resultat[h-accumulateur])
                    accumulateur += 1
    affichage_joueurs(Joueurs1)
    affichage_joueurs(Joueurs2)
    print(i)

print(Joueurs1)
print(Joueurs2)
