from Strategie_opti import prochain_deplacement_j1, prochain_deplacement_j2, Some_pos
from math import sqrt
import tkinter as tk
from time import sleep
role = int(input("1 pour attraper 2, 2 pour fuire 1"))
tour = []
nb_partie = 1
attente_deplacement = [True]
n = [4]
deplacementj1 = [0, 0]
deplacementj2 = [0, 0]
largeur = 1000
hauteur = 1000

nbl = 13  # impair

mon_app = tk.Tk()
surface_dessin = tk.Canvas(mon_app, width=largeur, height=hauteur, bg='ivory')
for i in range(1, nbl):
    surface_dessin.create_line(i * largeur / nbl, 0, i * largeur / nbl, hauteur)
    surface_dessin.create_line(0, i * hauteur / nbl, largeur, i * hauteur / nbl)


def clic(evenement):
    x, y = evenement.x, evenement.y
    deplacementj1[0] = round((x - largeur/2)*nbl/largeur)
    deplacementj1[1] = round((hauteur/2 - y)*nbl/hauteur)
    print(deplacementj1)
    surface_dessin.update()
    if abs(deplacementj1[0])+abs(deplacementj1[1]) <= n[0] + 2 - role:
        attente_deplacement[0] = False


def clic2(evenement):
    x, y = evenement.x, evenement.y
    deplacementj2[0] = round((x - largeur / 2) * nbl / largeur)
    deplacementj2[1] = round((hauteur / 2 - y) * nbl / hauteur)
    print(deplacementj2)
    surface_dessin.update()
    if abs(deplacementj2[0]) + abs(deplacementj2[1]) <= n[0] + 2 - role:
        attente_deplacement[0] = False


def ecartype(tableau):
    moy = sum(tableau) / len(tableau)
    somme = 0
    for i in tableau:
        somme += (i - moy) ** 2
    return sqrt(somme / len(tableau))


posj1 = [1, 0]
posj2 = [0, 0]

difference = [posj1[0]-posj2[0], posj1[1]-posj2[1]]
surface_dessin.pack(side=tk.LEFT)
mon_app.attributes('-topmost', True)
surface_dessin.update()
if role == 1:
    joueur1 = surface_dessin.create_oval(largeur / 2 + largeur / (4 * nbl), hauteur / 2 + hauteur / (4 * nbl),
                                         largeur / 2 - largeur / (4 * nbl), hauteur / 2 - hauteur / (4 * nbl),
                                         fill='red')
    joueur2 = surface_dessin.create_oval(largeur / 2 + largeur / (4 * nbl) - difference[0] * largeur / nbl,
                                         hauteur / 2 + hauteur / (4 * nbl) + difference[1] * hauteur / nbl,
                                         largeur / 2 - largeur / (4 * nbl) - difference[0] * largeur / nbl,
                                         hauteur / 2 - hauteur / (4 * nbl) + difference[1] * hauteur / nbl, fill='blue')

    surface_dessin.bind('<Button-1>', clic)
    for i in range(nb_partie):
        nb_tour = 0

        while posj1 != posj2:
            nb_tour += 1
            while attente_deplacement[0]:
                surface_dessin.update()
                sleep(0.01)
            attente_deplacement[0] = True

            deplacement_j1 = deplacementj1
            deplacement_j2 = prochain_deplacement_j2(posj1, posj2, n[0])
            print(deplacement_j2)
            posj1 = Some_pos(posj1, deplacement_j1)
            posj2 = Some_pos(posj2, deplacement_j2)
            surface_dessin.move(joueur2, largeur/nbl * deplacement_j2[0] - largeur/nbl * deplacement_j1[0], -hauteur/nbl * deplacement_j2[1] + hauteur/nbl * deplacement_j1[1])
        tour.append(nb_tour)
    print("Donnees :", tour)
    print("max et min :", max(tour), min(tour))
    print("Moyenne :", sum(tour)/len(tour))
    tour.sort()
    print("Mediane :", tour[nb_partie//2])
    print("Ecart type :", ecartype(tour))
elif role == 2:
    joueur2 = surface_dessin.create_oval(largeur / 2 + largeur / (4 * nbl), hauteur / 2 + hauteur / (4 * nbl),
                                         largeur / 2 - largeur / (4 * nbl), hauteur / 2 - hauteur / (4 * nbl),
                                         fill='blue')
    joueur1 = surface_dessin.create_oval(largeur / 2 + largeur / (4 * nbl) + difference[0] * largeur / nbl,
                                         hauteur / 2 + hauteur / (4 * nbl) - difference[1] * hauteur / nbl,
                                         largeur / 2 - largeur / (4 * nbl) + difference[0] * largeur / nbl,
                                         hauteur / 2 - hauteur / (4 * nbl) - difference[1] * hauteur / nbl, fill='red')

    surface_dessin.bind('<Button-1>', clic2)
    for i in range(nb_partie):
        nb_tour = 0

        while posj1 != posj2:
            nb_tour += 1
            while attente_deplacement[0]:
                surface_dessin.update()
                sleep(0.01)
            attente_deplacement[0] = True

            deplacement_j2 = deplacementj2

            deplacement_j1 = prochain_deplacement_j1(posj1, posj2, n[0])
            print(deplacement_j1)
            posj1 = Some_pos(posj1, deplacement_j1)
            posj2 = Some_pos(posj2, deplacement_j2)
            surface_dessin.move(joueur1, largeur/nbl * deplacement_j1[0] - largeur/nbl * deplacement_j2[0], -hauteur/nbl * deplacement_j1[1] + hauteur/nbl * deplacement_j2[1])
        tour.append(nb_tour)
    print("Donnees :", tour)
    print("max et min :", max(tour), min(tour))
    print("Moyenne :", sum(tour) / len(tour))
    tour.sort()
    print("Mediane :", tour[nb_partie // 2])
    print("Ecart type :", ecartype(tour))
else:
    print("Erreur role indefini")
mon_app.destroy()
