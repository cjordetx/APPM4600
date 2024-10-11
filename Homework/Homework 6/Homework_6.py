import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm



def driver():
    f1 = lambda x,y: x**2 + y ** 2 -4
    f2 = lambda x,y: np.exp(x) + y - 1

    x0 = np.array([1, 1])

    print(np.exp(x0[0]))

    Nmax = 100
    tol = 1e-10

    [xstar,ier,its] =  SlackerNewton(x0,tol,Nmax)
    print(xstar)
    print('Slacker Newton: the error message reads:',ier)
    print('Slacker Newton: number of iterations is:',its)

    print('\n')

    [xstar, ier, its] = Newton(f1, f2, x0, tol, Nmax)
    print(xstar)
    print('Newton: the error message reads:', ier)
    print('Netwon: number of iterations is:', its)


def evalF(x):
    F = np.zeros(2)

    F[0] = (x[0]) ** 2 + (x[1]) ** 2 - 4
    F[1] = np.exp(x[0]) + x[1] - 1
    #F[2] = np.exp(-x[0] * x[1]) + 20 * x[2] + (10 * math.pi - 3) / 3
    return F


def evalJ(x):
    J = np.array([[2*x[0], 2 * x[1]], [np.exp(x[0]), 1]])

    return J


def evalJapprox(f1, f2, x0, h):
    Japprox = np.array([[forward(f1, x0, h)[0], forward(f1, x0, h)[1]],
                [forward(f2, x0, h)[0], forward(f2, x0, h)[1]]])

    return Japprox


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


def Newton(f1, f2, x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = evalJapprox(f1, f2, x0, (0.01 * 2 ** (np.arange(0,10))))
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
    fprime2 = np.zeros(len(h))
    for i in range(len(h)):
        fprime1[i] = (f(x[0] - h[i],x[1]) - f(x[0],x[1]))/(h[i])
        fprime2[i] = (f(x[0],x[1] + h[i]) - f(x[0],x[1]))/(h[i])
    return [fprime1[len(fprime1) - 1], fprime2[len(fprime2) - 1]]




driver()