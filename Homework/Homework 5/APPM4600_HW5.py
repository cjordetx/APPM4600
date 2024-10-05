import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm



def driver():
    f = lambda x,y: 3*x**2 - y**2
    g = lambda x,y: 3 * x * (y **2) - x ** 3 - 1
    x0 = 1
    y0 = 1
    tol = 10**(-10)
    Nmax = 100


    [xstar, ystar, ier, its] = System(x0, y0, f, g, tol, Nmax)
    print(f'The fixed points for the system are:\n{np.array([[xstar],[ystar]])}')
    print('System: the error message reads:', ier)
    print('System: number of iterations is:', its)

    print('\n')

    x0newt = np.array([1,1])
    [xstar, ier, its] = Newton(x0newt, tol, Nmax)
    print(xstar)
    print('Newton: the error message reads:', ier)
    print('Netwon: number of iterations is:', its)


def evalF(x):
    F = np.zeros(2)

    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3 * x[0] * (x[1] **2) - x[0] ** 3 - 1
    return F


def evalJ(x):
    J = np.array([[6*x[0], -2*x[1]],
                  [3*(x[1] ** 2)-3*(x[0]**2), 6*x[0]*x[1]]])
    return J


def Newton(x0newt, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = evalJ(x0newt)
        Jinv = inv(J)
        F = evalF(x0newt)

        x1 = x0newt - Jinv.dot(F)

        if (norm(x1 - x0newt) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0newt = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def System(x0, y0, f, g, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        #print(np.array([[1/6, 1/18], [0, 1/6]]))
        F = np.array([[x0],[y0]]) - np.matmul(np.array([[1/6, 1/18], [0, 1/6]]), np.array([[f(x0,y0)],[g(x0,y0)]]))
        #print(F[0][0])

        if (norm(F - np.array([[x0],[y0]])) < tol):
            xstar = F[0][0]
            ystar = F[1][0]
            ier = 0
            return [xstar, ystar, ier, its]

        x0 = F[0][0]
        y0 = F[1][0]

    xstar = F[0][0]
    ystar = F[1][0]
    ier = 1
    return [xstar, ystar, ier, its]

driver()