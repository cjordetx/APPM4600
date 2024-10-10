import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    f = lambda x: 1/(1+((10*x)**2))

    N = 18
    a = -1
    b = 1
    x_values = np.linspace(a, b, N + 1)
    f_values = f(x_values)
    # Find coefficients
    coefficients = coeffs(x_values, f_values,N)

    # Evaluate polynomial at a new point
    Neval = 1000
    x_eval = np.linspace(a, b, Neval + 1)
    y_eval = eval_monomial(x_eval, coefficients, N, Neval)
    y_exact = f(x_eval)
    err = norm(y_exact - y_eval)
    print('err = ', err)

    result = evaluate_polynomial(coefficients, x_eval)

    print("Coefficients:", coefficients)
    print("p_n(x_eval):", result)

    yeval_l = np.zeros(Neval + 1)
    yeval_dd = np.zeros(Neval + 1)

    y = np.zeros((N + 1, N + 1))

    for j in range(N + 1):
        y[j][0] = f_values[j]

    y = dividedDiffTable(x_values, y, N + 1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval + 1):
        yeval_l[kk] = eval_lagrange(x_eval[kk], x_values, f_values, N)
        yeval_dd[kk] = evalDDpoly(x_eval[kk], x_values, y, N)

    ''' create vector with exact values'''
    fex = f(x_eval)

    plt.figure()
    plt.plot(x_eval, fex, 'ro-', label = 'exact')
    plt.plot(x_eval, yeval_l, 'bs--', label = 'lagrange')
    plt.plot(x_eval, yeval_dd, 'c.--', label = 'dd')
    plt.plot(x_eval, y_eval, 'g>', label='monomial')
    plt.legend()

    plt.figure()
    err_l = abs(yeval_l - fex)
    err_dd = abs(yeval_dd - fex)
    plt.semilogy(x_eval, err_l, 'ro--', label='lagrange')
    plt.semilogy(x_eval, err_dd, 'bs--', label='Newton DD')
    plt.legend()
    plt.show()


def eval_lagrange(xeval, xint, yint, N):
    lj = np.ones(N + 1)

    for count in range(N + 1):
        for jj in range(N + 1):
            if (jj != count):
                lj[count] = lj[count] * (xeval - xint[jj]) / (xint[count] - xint[jj])

    yeval = 0.

    for jj in range(N + 1):
        yeval = yeval + yint[jj] * lj[jj]

    return (yeval)


''' create divided difference matrix'''


def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                       (x[j] - x[i + j]));
    return y;


def evalDDpoly(xval, xint, y, N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N + 1)

    ptmp[0] = 1.
    for j in range(N):
        ptmp[j + 1] = ptmp[j] * (xval - xint[j])

    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N + 1):
        yeval = yeval + y[0][j] * ptmp[j]

    return yeval


def vandermonde_matrix(x_values, N):
    V = np.zeros((N + 1, N + 1))

    ''' fill the first column'''
    for j in range(N + 1):
        V[j][0] = 1.0

    for i in range(1, N + 1):
        for j in range(N + 1):
            V[j][i] = x_values[j] ** i

    return V

def coeffs(x_values, f_values,N):
    V = vandermonde_matrix(x_values, N)
    a = inv(V) @ f_values
    return a

def evaluate_polynomial(a, x):
    return sum(a[j] * (x ** j) for j in range(len(a)))


def eval_monomial(x_eval, coefficients, N, Neval):
    y_eval = coefficients[0] * np.ones(Neval + 1)

    #    print('yeval = ', yeval)

    for j in range(1, N + 1):
        for i in range(Neval + 1):
            #        print('yeval[i] = ', yeval[i])
            #        print('a[j] = ', a[j])
            #        print('i = ', i)
            #        print('xeval[i] = ', xeval[i])
            y_eval[i] = y_eval[i] + coefficients[j] * x_eval[i] ** j

    return y_eval


driver()
