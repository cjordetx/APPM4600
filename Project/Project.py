import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

# given splines and nodes how does that affect matrix EXAM 2
# Least squares on 3x3, Pade approx,

def driver():

    f = lambda x: np.sin(x)
    a = -1
    b = 1
    N = 15
    x_values = (a+b)/2 + (a-b) * (np.cos((2 * np.arange(1, N + 2) - 1) * np.pi / (2 * N)))/2
    f_values = f(x_values)
    print(f_values[5])
    Neval = 1000
    x_eval = np.linspace(a, b, Neval + 2)
    fex = f(x_eval)
    coefficients = coeffs(x_values, f_values, N)
    x_eval = np.linspace(a, b, Neval + 2)
    f_approx = eval_monomial(x_eval, coefficients, N, Neval)
    err = np.max(f_approx - fex)
    print('f_approx', err)

    remez_coefficients = remez_coeffs(x_values, f_values, N, err)
    f_remez = eval_monomial(x_values, remez_coefficients, N, Neval)
    err_remez = abs(fex - f_remez)
    plt.plot(x_eval, fex, 'rs-', label='exact')
    plt.plot(x_eval, f_approx, 'o-', label=f'approx')
    plt.legend()
    plt.show()


    plt.semilogy(x_eval, err, 'o-', label=f'error')
    plt.legend()
    plt.show()

    plt.plot(x_eval, fex, 'rs-', label='exact')
    plt.plot(x_eval, f_remez, 'o-', label=f'approx')
    plt.legend()
    plt.show()

    plt.semilogy(x_eval, err_remez, 'o-', label=f'error')
    plt.legend()
    plt.show()

def vandermonde_matrix(x_values, N):
    V = np.zeros((N+1, N+1))

    ''' fill the first column'''
    for j in range(N+1):
        V[j][0] = 1.0

    for i in range(1, N+1):
        for j in range(N+1):
            V[j][i] = x_values[j] ** i
    return V

def coeffs(x_values, f_values,N):
    V = vandermonde_matrix(x_values, N)
    a = inv(V) @ f_values
    return a

def eval_monomial(x_eval, coefficients, N, Neval):
    y_eval = coefficients[0] * np.ones(Neval + 2)

    #    print('yeval = ', yeval)

    for j in range(1,N):
        for i in range(Neval + 1):
            #        print('yeval[i] = ', yeval[i])
            #        print('a[j] = ', a[j])
            #        print('i = ', i)
            #        print('xeval[i] = ', xeval[i])
            y_eval[i] = y_eval[i] + coefficients[j] * x_eval[i] ** j

    return y_eval

def remez_vandermonde_matrix(x_values, N, f_values, err):
    V = np.zeros((N + 2, N + 2))
    #V[0][N+2] = x_values[0] - f_values[0]

    ''' fill the first column'''
    for j in range(N+1):
        V[j][0] = 1.0
        V[j][N+1] = err * (-1) ** j


    for i in range(1, N):
        for j in range(N + 1):
            V[j][i] = x_values[j] ** i
    return V

def remez_coeffs(x_values, f_values,N, err):
    V = remez_vandermonde_matrix(x_values, N, f_values, err)
    a = inv(V) @ f_values
    return a

driver()