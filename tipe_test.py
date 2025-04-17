from random import randint
from tkinter import *
from time import sleep
import tensorflow as tf
import numpy as np


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


for j in range(2,40):
    Vecteur1 = [0, 0]
    Vecteur2 = [0, 10]
    difference = [-4, 4]
    compteur = 0
    largeur = 1000
    hauteur = 1000
    nbl = 31
    model = tf.keras.models.load_model("Agent J1 " + str(30))
    model2 = tf.keras.models.load_model("Agent J2 " + str(4))
    for c in range(12):
        fenetre = Tk()
        monCanvas = Canvas(fenetre, width=largeur, height=hauteur, bg='ivory')

        for i in range(1, nbl):
            monCanvas.create_line(i * largeur / nbl, 0, i * largeur / nbl, hauteur)
            monCanvas.create_line(0, i * hauteur / nbl, largeur, i * hauteur / nbl)
        monCanvas.create_oval(largeur / 2 + largeur / (4 * nbl), hauteur / 2 + hauteur / (4 * nbl), largeur / 2 - largeur / (4 * nbl), hauteur / 2 - hauteur / (4 * nbl), fill='red')
        difference = [Vecteur1[0] - Vecteur2[0], Vecteur1[1] - Vecteur2[1]]
        monCanvas.create_oval(largeur / 2 + largeur / (4 * nbl) - difference[0] * largeur / nbl, hauteur / 2 + hauteur / (4 * nbl) + difference[1] * hauteur / nbl, largeur / 2 - largeur / (4 * nbl) - difference[0] * largeur / nbl, hauteur / 2 - hauteur / (4 * nbl) + difference[1] * hauteur / nbl, fill='blue')
        monCanvas.pack()
        fenetre.mainloop()

        compteur += 1
        deplacement = 4
        print(deplacement)
        #x1, y1 = list(map(int, input().split()))
        sortie_1 = model.predict(np.array([[Vecteur1[0], Vecteur1[1], Vecteur2[0], Vecteur2[1]]]))
        x1, y1 = int((sortie_1[0][0]-0.5)*10), int((sortie_1[0][1]-0.5)*10)
        x1, y1 = rectification(x1, y1, deplacement + 1)
        Vecteur1[0] += x1
        Vecteur1[1] += y1
        #x2, y2 = list(map(int, input().split()))
        sortie_2 = model2.predict(np.array([[Vecteur1[0], Vecteur1[1], Vecteur2[0], Vecteur2[1]]]))
        x2, y2 = int((sortie_2[0][0] - 0.5) * 10), int((sortie_2[0][1] - 0.5) * 10)
        x2, y2 = rectification(x2, y2, deplacement)
        Vecteur2[0] += x2
        Vecteur2[1] += y2
        print("tour :", compteur, Vecteur1, Vecteur2)
        if Vecteur1 ==Vecteur2 :
            break

print("Victoire en", compteur, "tours")
