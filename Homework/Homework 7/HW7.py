import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

def driver():

    f = lambda x: 1/(1+((10*x)**2))
    plt.figure(figsize=(30,15))
    for x in range(1, 20):
        N = x + 1
        a = -1
        b = 1
        h = 2/(N - 1)
        x_values = np.array([-1 + (i - 1)*h for i in range(1, N + 1)])
        f_values = f(x_values)
        # Find coefficients
        coefficients = coeffs(x_values, f_values,N)
        Neval = 1000
        x_eval = np.linspace(a, b, Neval + 1)
        f_approx = eval_monomial(x_eval,coefficients,N,Neval)
        fex = f(x_eval)
        plt.plot(x_eval, fex, 'rs-', label='exact')
        plt.plot(x_eval, f_approx, 'o-', label=f'approx{x+1}')
        plt.legend()
    plt.show()

    plt.figure(figsize=(30,15))
    for x in range(1, 20):
        N = x + 1
        h = 2 / (N - 1)
        x_values = np.array([-1 + (i - 1) * h for i in range(1, N + 1)])
        Neval = 1000
        x_eval = np.linspace(a, b, Neval + 1)
        y_values = f(x_values)
        fex = f(x_eval)
        f_approx_bary = np.zeros(Neval + 1)
        count = 0
        for l in range(Neval):
            count += 1
            print(count)
            f_approx_bary[l] = barycentric_lagrange(x_values, y_values, N, x_eval[l])
        plt.plot(x_eval, fex, 'rs-', label='exact')
        plt.plot(x_eval, f_approx_bary, 'o-', label=f'approx_bary{N}')
        plt.legend()
    plt.show()

    plt.figure()
    N = 100
    x_cheb = np.cos((2 * np.arange(1, N + 1) - 1) * np.pi / (2 * N))
    Neval = 1000
    x_eval = np.linspace(a, b, Neval + 1)
    y_cheb = f(x_cheb)
    fex = f(x_eval)
    f_cheb_bary = np.zeros(Neval + 1)
    count = 0
    for l in range(Neval):
        count += 1
        print(count)
        f_cheb_bary[l] = barycentric_lagrange(x_cheb, y_cheb, N, x_eval[l])
    plt.plot(x_eval, fex, 'rs-', label='exact')
    plt.plot(x_eval, f_cheb_bary, 'o-', label=f'approx_bary_chebyshev{N}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(30,15))
    for x in range(1, 20):
        N = x + 1
        x_cheb = np.cos((2 * np.arange(1, N + 1) - 1) * np.pi / (2 * N))
        Neval = 1000
        x_eval = np.linspace(a, b, Neval + 1)
        y_cheb = f(x_cheb)
        fex = f(x_eval)
        f_cheb_bary = np.zeros(Neval + 1)
        count = 0
        for l in range(Neval):
            count += 1
            print(count)
            f_cheb_bary[l] = barycentric_lagrange(x_cheb, y_cheb, N, x_eval[l])
        plt.plot(x_eval, fex, 'rs-', label='exact')
        plt.plot(x_eval, f_cheb_bary, 'o-', label=f'approx_bary_chebyshev{x}')
        plt.legend()
    plt.show()


def vandermonde_matrix(x_values, N):
    V = np.zeros((N, N))

    ''' fill the first column'''
    for j in range(N):
        V[j][0] = 1.0

    for i in range(1, N):
        for j in range(N):
            V[j][i] = x_values[j] ** i
    return V

def coeffs(x_values, f_values,N):
    V = vandermonde_matrix(x_values, N)
    a = inv(V) @ f_values
    return a

def eval_monomial(x_eval, coefficients, N, Neval):
    y_eval = coefficients[0] * np.ones(Neval + 1)

    #    print('yeval = ', yeval)

    for j in range(1,N):
        for i in range(Neval + 1):
            #        print('yeval[i] = ', yeval[i])
            #        print('a[j] = ', a[j])
            #        print('i = ', i)
            #        print('xeval[i] = ', xeval[i])
            y_eval[i] = y_eval[i] + coefficients[j] * x_eval[i] ** j

    return y_eval


def barycentric_lagrange(x_values, y_values, N, x):
    f = lambda x: 1 / (1 + ((10 * x) ** 2))
    w = np.zeros(N)

    for i in range(N):
        w[i] = 1 / np.prod(x_values[i] - np.delete(x_values, i))

    if x in x_values:
        return f(x)
    # Initialize p(x)
    p_x = 0.0
    denom = 0.0

    for j in range(N):
        numer = w[j] * y_values[j] / (x - x_values[j])
        p_x += numer
        denom += w[j] / (x - x_values[j])

    return p_x / denom



driver()
