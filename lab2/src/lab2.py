import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation
import math

Steps = 1001
t = np.linspace(0, 10, Steps)

thetta = np.cos(10*t)
phi = np.cos(16*t)

""" omega_phi = sp.diff(phi, t)
omega_thetta = sp.diff(thetta, t) """


# Ground 
alpha = np.linspace(-math.pi, 0, Steps)
R_Ground = 6

X_Ground = R_Ground + R_Ground * np.cos(alpha)
Y_Ground = R_Ground + R_Ground * np.sin(alpha)

# Point O
X_O = R_Ground
Y_O = R_Ground

# circle 
beta = np.linspace(0, 2*math.pi, Steps)
R_Circle = 1

X_Circle = R_Circle * np.cos(beta)
Y_Circle = R_Circle * np.sin(beta)

# Point O1
X_O1 = -(R_Ground - R_Circle) * np.sin(thetta) + R_Ground
Y_O1 = -(R_Ground - R_Circle) * np.cos(thetta) + R_Ground

# point A
l = 3 # length of the palka between O1 and A
X_A = X_O1 + l*np.sin(phi) 
Y_A = Y_O1 - l*np.cos(phi)


# some settings
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.axis("equal")
ax.set(xlim=(0, 12), ylim=(0, 12))


# plot first zero state
Ground = ax.plot(X_Ground, Y_Ground, color='black', linewidth=2)
Point_O = ax.plot(X_O, Y_O, color='red', linewidth=4)
Draw_palka = ax.plot([X_O, X_O1[0]], [Y_O, Y_O1[0]], 'r--')[0]
Draw_Circle = ax.plot(X_Circle + X_O1[0], Y_Circle + Y_O1[0], color='blue', linewidth=1)[0]
Draw_point_O1 = ax.plot(X_O1[0], Y_O1[0], color='blue', linewidth=3, marker='o')[0]
Draw_point_A = ax.plot(X_A[0], Y_A[0], 'r', marker='o', markersize=15)[0]
Draw_palka_O1_A = ax.plot([X_O1[0], X_A[0]], [Y_O1[0], Y_A[0]], 'b')[0]


# function for updating state of the system
def kinoteatr_five_zvezd_na_novokuzneckoy(i):
    Draw_point_O1.set_data(X_O1[i], Y_O1[i])
    Draw_Circle.set_data(X_Circle + X_O1[i], Y_Circle + Y_O1[i])
    Draw_palka.set_data([X_O, X_O1[i]], [Y_O, Y_O1[i]])
    Draw_point_A.set_data(X_A[i], Y_A[i])
    Draw_palka_O1_A.set_data([X_O1[i], X_A[i]], [Y_O1[i], Y_A[i]])
    return [Draw_point_O1, Draw_Circle, Draw_palka, Draw_point_A]

anime = FuncAnimation(fig, kinoteatr_five_zvezd_na_novokuzneckoy,
                      frames=Steps, interval=1)

# show figure
plt.show()

# anime.save("cringe.gif")
