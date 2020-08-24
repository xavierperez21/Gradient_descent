#!/usr/bin/env python
# coding: utf-8

# **_Luis Xavier Pérez Miramontes_ _Mechatronics Engineer_**

import numpy as np
import scipy as sc

import matplotlib.pyplot as plt

# Declaring the cost function as a lambda (anonymous) function
func = lambda th: np.sin(1/2 * th[0]**2 - 1/4 * th[1]**2 + 3) * np.cos(2*th[0] + 1 - np.e**th[1])

resolution = 100

_X = np.linspace(-2, 2, resolution)
_Y = np.linspace(-2, 2, resolution)

# We first create the matrix where all the values of X and Y will be evaluated
_Z = np.zeros((resolution, resolution))

# Evaluating the cost function in points of X and Y
# 'enumerate' returns the index and the value of each element in the array (vector)
for ix, x in enumerate(_X):
    for iy, y in enumerate(_Y):
        _Z[iy, ix] = func([x, y]) # We fill the matrix by columns


# Plotting our cost function visualized from top.
plt.contourf(_X, _Y, _Z, 100)
plt.colorbar()

# Generating a random point ([?, ?]) that goes from -2 to 2
# This random point represents the values of our parameters (W0, W1)
# This values will change when we apply the gradient descent until we find
# the best values for our parameters that locates the point to the most deep zone of the cost function
Theta = np.random.rand(2) * 4 - 2


# --- Applying the gradient descent algorithm ---
# We have to find the gradient vector that indicates the slope of the cost function in the actual location
# So once we found it, we have to move to the opposite direction because we want to go to the deeper zones.
# To calculate this, we have to use de partial derivatives.
_T = np.copy(Theta)

h = 0.001
learning_rate = 0.001  # Hyperparameter "Learning Rate"

plt.plot(Theta[0], Theta[1], "o", c="white")  # Start point

grad = np.zeros(2)

for _ in range(10000):
    for it, th in enumerate(Theta):

        _T = np.copy(Theta)

        # Calculating the partial derivative through a numeric method (Numerical differentiation)
        _T[it] = _T[it] + h
        deriv = (func(_T) - func(Theta)) / h

        grad[it] = deriv

    # Once we have calculated our gradient vector, we update the values of our parameters
    # We substract the gradient values to our parameters (Theta vector) to update them in order to...
    # ...get the values that makes the cost function to decrease
    Theta = Theta - learning_rate * grad


    if (_ % 100 == 0):
        plt.plot(Theta[0], Theta[1], ".", c="red")  # Road of every iteration

plt.plot(Theta[0], Theta[1], "o", c="blue")  # Final point
plt.show()

