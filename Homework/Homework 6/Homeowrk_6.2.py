import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm

def driver():

    initial_guess = np.array([0.5, 0.5, 0.5])
    tol=1e-6
    Nmax=100

    [x_newton, its] = newton_method(initial_guess, tol, Nmax)
    print(f'The result from newton is: {x_newton}')
    print(f'The iteration number is {its}')

    print('\n')

    # First, perform steepest descent with a larger tolerance
    [x_sd, its] = steepest_descent_with_initial_guess(initial_guess, tol, Nmax)
    print(f'The result from steepest descent is: {x_sd}')
    print(f'The iteration number is {its}')

    print('\n')

    [x_newt_sd, its] = newton_method(x_sd, tol, Nmax)
    print(f'The result from newton is: {x_newt_sd}')
    print(f'The iteration number is {its}')



def F_eval(x0):
    x = x0[0]
    y = x0[1]
    z = x0[2]

    F = np.array([x + np.cos(x * y * z) - 1,
        (1 - x)**(1/4) + y + 0.05 * z**2 - 0.15 * z - 1,
        -x**2 - 0.1 * y**2 + 0.01 * y + z - 1])

    return F

def jacobian(x0):
    x = x0[0]
    y = x0[1]
    z = x0[2]
    return np.array([[1 - y * z * np.sin(x * y * z), -x * z * np.sin(x * y * z), -x * y * np.sin(x * y * z)],
        [-0.25 * (1 - x)**(-3/4), 1, 0.1 * z - 0.15],
        [-2 * x, -0.2 * y + 0.01, 1] ])


def newton_method(x0, tol, Nmax):

    for i in range(Nmax):
        J = jacobian(x0)
        Jinv = inv(J)
        F = F_eval(x0)
        x1 = x0 - Jinv.dot(F)
        if norm(x1-x0) < tol:
            return [x1, i]

        x0 = x1


def steepest_descent(x0, tol, Nmax):
    for i in range(Nmax):
        grad = jacobian(x0).T @ F_eval(x0)# Grad = Jacobian^transpose * F
        d = -grad
        alpha = 1  # Step size

        while True:
            x1 = x0 - alpha * grad
            if norm(F_eval(x1)) <= norm(F_eval(x0)) + 10**(-4) * alpha * np.dot(F_eval(x0), d):
                break
            alpha = alpha / 2


        x1 = x0 - alpha * grad

        if norm(x1 - x0) < tol:
            return [x1, i]

        x0 = x1

# 3. First Steepest Descent then Newton's Method
def steepest_descent_with_initial_guess(x0, tol = 5* 10 ** (-2), Nmax=100):
    return steepest_descent(x0, tol, Nmax)

driver()