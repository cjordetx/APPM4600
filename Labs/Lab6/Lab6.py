import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm



def driver():
    f = lambda x: math.cos(x)
    x = math.pi / 2
    h = 1 / (0.01 * 2 ** (np.arange(0,10)))

    fprime1 = forward(f,x,h)
    print(f"The approximate derivative of f at x for each h step is {fprime1}")

    print("\n")

    fprime2 = backward(f,x,h)
    print(f"The approximate derivative of f at x for each h step is {fprime2}")

    print("\n")

    x0 = np.array([1, 0])

    Nmax = 100
    tol = 1e-10

    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  SlackerNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Slacker Newton: the error message reads:',ier)
    print('Slacker Newton: took this many seconds:',elapsed/20)
    print('Slacker Newton: number of iterations is:',its)


def evalF(x):
    F = np.zeros(2)

    F[0] = 4 * (x[0]) ** 2 + x[1] ** 2 - 4
    F[1] = x[0] + x[1] - math.sin(x[0] - x[1])
    #F[2] = np.exp(-x[0] * x[1]) + 20 * x[2] + (10 * math.pi - 3) / 3
    return F


def evalJ(x):
    Japprox = np.array([[8*x[0], 2 * x[1]], [1-math.cos(x[0]-x[1]), 1+math.cos(x[0]-x[1])]])

    return Japprox


def evalJapprox(x):
    J = np.array([(evalF(x[0] + h[0]) - evalF(x[0]))/h[0], (evalF(x[0]) - evalF(x[0] + h[1]))/h[1]],
                [(evalF(x[1] + h[0]) - evalF(x[1]))/h[0], (evalF(x[1]) - evalF(x[1] + h[1]))/h[1]],)

    return J


def SlackerNewton(x0, tol, Nmax):
    ''' SlackerNewton = use only the inverse of the Jacobian for initial guess then
    if it takes too long to converge or doesn't, we recalculate the jacobian at each step'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)

        if (norm(np.dot(Jinv[0], F[0])) >= 0.00001):
            J = evalJ(x0)
            Jinv = inv(J)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def Newton(x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)

        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def forward(f,x,h):
    fprime1 = np.zeros(len(h))
    for i in range(len(h)):
        print(h[i])
        fprime1[i] = (f(x + h[i]) - f(x))/(h[i])
    return fprime1


def backward(f,x,h):
    fprime2 = np.zeros(len(h))
    for i in range(len(h)):
        fprime2[i] = (f(x + h[i]) - f(x - h[i]))/(2*h[i])
    return fprime2





driver()