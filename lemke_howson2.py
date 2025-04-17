import numpy as np
import nashpy as nash
from Calcul import G1, G2, symetricG1


A = np.array(G1(1,[0,0],[0,4]))
B = np.array(G2(1,[0,0],[0,4]))

game = nash.Game(A, B)
list(game.lemke_howson_enumeration())