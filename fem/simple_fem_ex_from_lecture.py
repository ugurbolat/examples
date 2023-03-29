# from fem online course bilkent?

import numpy as np

NL = np.array([[0,0],
[1,0],
[0.5,1]])

EL = np.array([[1,2],
[2,3]
[3,1]])

DorN = np.array([[-1,-1],
[1,-1],
[1,1]])

Fu = np.array([[0,0],
[0,0],
[0,-20]])

U_u = np.array([[0,0],
[0,0],
[0,0]])

# material prop.
E=10**6
A=0.01