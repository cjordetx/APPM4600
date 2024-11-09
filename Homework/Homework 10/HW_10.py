import numpy as np
import matplotlib.pyplot as plt
import math

def driver():

    x_values = np.linspace(0, 5, 500)
    coeffs33 = [0, 1, 0, -7/60, 0, 1/20, 0]
    coeffs24 = [0, 1, 0, 0, 1/6, 0, 7/360]
    coeffs42 = [0, 1, 0, -7/60, 0, 0, 1/20]
    P_33 = P33(x_values, coeffs33)
    P_24 = P24(x_values, coeffs24)
    P_42 = P42(x_values, coeffs42)
    maclaurin = taylor_approx(x_values)

    error_maclaurin = np.abs(f(x_values) - maclaurin)
    error_P33 = np.abs(f(x_values) - P_33)
    error_P24 = np.abs(f(x_values) - P_24)
    error_P42 = np.abs(f(x_values) - P_42)

    plt.figure()
    plt.plot(x_values, f(x_values), 'bo--', label = 'exact')
    plt.plot(x_values, maclaurin, 'y.--', label='Maclaurin')
    plt.plot(x_values, P_33, 'rs--', label='P33')
    plt.plot(x_values, P_24, 'g+--', label='P24')
    plt.plot(x_values, P_42, 'm*--', label='P42')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('function evaluated at x')
    plt.title('Exact Function and Approximations of sin(x) over [0, 5]')
    plt.grid(True)
    plt.show()


    plt.figure()
    plt.semilogy(x_values, error_P33, 'rs--', label='Error of P33')
    plt.semilogy(x_values, error_P24, 'g+--', label='Error of P24')
    plt.semilogy(x_values, error_P42, 'm*--', label='Error of P42')
    plt.semilogy(x_values, error_maclaurin, 'y.--', label='Error of Maclaurin')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Error Comparison of Approximations to sin(x) over [0, 5]')
    plt.grid(True)
    plt.show()



def f(x):
    return np.sin(x)

def taylor_approx(x_values):
    taylor = np.zeros(len(x_values))
    for i in range(len(x_values)):
        taylor[i] = x_values[i] - (x_values[i] ** 3)/math.factorial(3) + (x_values[i] ** 5)/math.factorial(5)

    return taylor


def P33(x_values, coeffs):
    a0 = coeffs[0]
    a1 = coeffs[1]
    a2 = coeffs[2]
    a3 = coeffs[3]
    b1 = coeffs[4]
    b2 = coeffs[5]
    b3 = coeffs[6]
    p = np.zeros(len(x_values))
    for i in range(len(x_values)):
        p[i] = (a0 + a1 * x_values[i] + a2 * (x_values[i] ** 2) + a3 * (x_values[i] ** 3)) / (1 + b1 * x_values[i] + b2 * (x_values[i] ** 2) + b3 * (x_values[i] ** 3))

    return p

def P24(x_values, coeffs):
    a0 = coeffs[0]
    a1 = coeffs[1]
    a2 = coeffs[2]
    b1 = coeffs[3]
    b2 = coeffs[4]
    b3 = coeffs[5]
    b4 = coeffs[6]
    p = np.zeros(len(x_values))
    for i in range(len(x_values)):
        p[i] = (a0 + a1 * x_values[i] + a2 * (x_values[i] ** 2)) / (1 + b1 * x_values[i] + b2 * (x_values[i] ** 2) + b3 * (x_values[i] ** 3) + b4 * (x_values[i] ** 4))

    return p

def P42(x_values, coeffs):
    a0 = coeffs[0]
    a1 = coeffs[1]
    a2 = coeffs[2]
    a3 = coeffs[3]
    a4 = coeffs[4]
    b1 = coeffs[5]
    b2 = coeffs[6]
    p = np.zeros(len(x_values))
    for i in range(len(x_values)):
        p[i] = (a0 + a1 * x_values[i] + a2 * (x_values[i] ** 2) + a3 * (x_values[i] ** 3) + a4 * (x_values[i] ** 4)) / (1 + b1 * x_values[i] + b2 * (x_values[i] ** 2))

    return p

driver()