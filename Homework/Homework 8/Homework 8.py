import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm


def driver():
    f = lambda x: 1 / (1 + x ** 2)
    df = lambda x: -(2*x) / ((1+x**2)**2)
    a = -5
    b = 5

    ''' number of intervals'''
    N = [5, 10, 15, 20]
    for Nint in N:
        x_equi = np.linspace(a, b, Nint + 1)
        x_cheb = np.cos((2 * np.arange(1, Nint + 2) - 1) * np.pi / (2 * Nint))
        xint = [x_equi, x_cheb]

        for xs in xint:
            yint = f(xs)
            df_a = df(a)
            df_b = df(b)

            ''' create points you want to evaluate at'''
            Neval = 500
            xeval = np.linspace(xs[0], xs[Nint], Neval + 1)

            y_int = np.zeros(Nint+1)
            yp_int = np.zeros(Nint+1)
            for jj in range(Nint+1):
                y_int[jj] = f(xs[jj])
                yp_int[jj] = df(xs[jj])


            yevalL = np.zeros(Neval + 1)
            yevalH = np.zeros(Neval + 1)
            for kk in range(Neval + 1):
                yevalL[kk] = eval_lagrange(xeval[kk], xs, y_int, Nint)
                yevalH[kk] = eval_hermite(xeval[kk], xs, y_int, yp_int, Nint)

            ''' create vector with exact values'''
            fex = np.zeros(Neval + 1)
            for kk in range(Neval + 1):
                fex[kk] = f(xeval[kk])


            errL = abs(yevalL - fex)
            errH = abs(yevalH - fex)

            (M, C, D) = create_natural_spline(yint, xs, Nint)

            print('M =', M)
            print('C =', C)
            print('D=', D)

            yeval_nat = eval_cubic_spline(xeval, Neval, xs, Nint, M, C, D)

            print('yeval_nat = ', yeval_nat)

            ''' evaluate f at the evaluation points'''
            fex = f(xeval)

            nerr_nat = norm(fex - yeval_nat)
            print('nerr_nat = ', nerr_nat)

            print('\n')

            err_nat = abs(yeval_nat - fex)

            (M_clamp, C_clamp, D_clamp) = create_clamped_spline(yint, xs, Nint, df_a, df_b)

            print('M_clamp =', M_clamp)
            print('C_clamp =', C_clamp)
            print('D_clamp =', D_clamp)

            yeval_clamp = eval_cubic_spline(xeval, Neval, xs, Nint, M_clamp, C_clamp, D_clamp)

            print('yeval_clamp = ', yeval_clamp)

            nerr_clamp = norm(fex - yeval_clamp)
            print('nerr_clamp = ', nerr_clamp)

            print('\n')

            plt.figure()
            plt.plot(xeval, fex, 'ro-', label='exact function')
            plt.plot(xeval, yevalL, 'bs--', label='Lagrange')
            plt.plot(xeval, yevalH, 'c.--', label='Hermite')
            plt.plot(xeval, yeval_clamp, 'g^--', label='clamped spline')
            plt.plot(xeval, yeval_nat, 'm+--', label='natural spline')

            plt.legend()
            plt.title(f'n = {Nint}')
            plt.show()

            err_clamp = abs(yeval_clamp - fex)
            plt.figure()
            plt.semilogy(xeval, errL, 'bs--', label='Lagrange Error')
            plt.semilogy(xeval, errH, 'c.--', label='Hermite Error')
            plt.semilogy(xeval, err_clamp, 'g^--', label='absolute error clamp')
            plt.semilogy(xeval, err_nat, 'm+--', label='absolute error natural')
            plt.legend()
            plt.title(f'n = {Nint}')
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


def eval_hermite(xeval, xint, yint, ypint, N):
    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if (jj != count):
                lj[count] = lj[count] * (xeval - xint[jj]) / (xint[count] - xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N + 1)
    #    lpj2 = np.ones(N+1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if (jj != count):
                #              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
                lpj[count] = lpj[count] + 1. / (xint[count] - xint[jj])

    yeval = 0.

    for jj in range(N + 1):
        Qj = (1. - 2. * (xeval - xint[jj]) * lpj[jj]) * lj[jj] ** 2
        Rj = (xeval - xint[jj]) * lj[jj] ** 2
        #       if (jj == 0):
        #         print(Qj)

        #         print(Rj)
        #         print(Qj)
        #         print(xeval)
        #        return
        yeval = yeval + yint[jj] * Qj + ypint[jj] * Rj

    return (yeval)


def create_natural_spline(yint, xint, N):
    #    create the right  hand side for the linear system
    b = np.zeros(N + 1)
    #  vector values
    h = np.zeros(N + 1)
    for i in range(1, N):
        hi = xint[i] - xint[i - 1]
        hip = xint[i + 1] - xint[i]
        b[i] = (yint[i + 1] - yint[i]) / hip - (yint[i] - yint[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    #  create matrix so you can solve for the M values
    # This is made by filling one row at a time
    A = np.zeros((N + 1, N + 1))
    A[0][0] = 1.0
    for j in range(1, N):
        A[j][j - 1] = h[j - 1] / 6
        A[j][j] = (h[j] + h[j - 1]) / 3
        A[j][j + 1] = h[j] / 6
    A[N][N] = 1

    Ainv = inv(A)

    M = Ainv.dot(b)

    #  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6
    return (M, C, D)


def eval_local_spline(xeval, xi, xip, Mi, Mip, C, D):
    # Evaluates the local spline as defined in class
    # xip = x_{i+1}; xi = x_i
    # Mip = M_{i+1}; Mi = M_i

    hi = xip - xi
    yeval = (Mi * (xip - xeval) ** 3 + (xeval - xi) ** 3 * Mip) / (6 * hi) \
            + C * (xip - xeval) + D * (xeval - xi)
    return yeval


def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval + 1)

    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp = xint[j + 1]

        #   find indices of values of xeval in the interval
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

        # evaluate the spline
        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j + 1], C[j], D[j])
        #        print('yloc = ', yloc)
        #   copy into yeval
        yeval[ind] = yloc

    return (yeval)


def create_clamped_spline(yint, xint, N, df_a, df_b):
    #    create the right  hand side for the linear system
    b = np.zeros(N + 1)
    #clamped conditions
    b[0] = df_a
    b[N] = df_b
    #  vector values
    h = np.zeros(N + 1)
    for i in range(1, N):
        hi = xint[i] - xint[i - 1]
        hip = xint[i + 1] - xint[i]
        b[i] = (yint[i + 1] - yint[i]) / hip - (yint[i] - yint[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    #  create matrix so you can solve for the M values
    # This is made by filling one row at a time
    A = np.zeros((N + 1, N + 1))
    A[0][0] = 1.0
    for j in range(1, N):
        A[j][j - 1] = h[j - 1] / 6
        A[j][j] = (h[j] + h[j - 1]) / 3
        A[j][j + 1] = h[j] / 6
    A[N][N] = 1

    Ainv = inv(A)

    M = Ainv.dot(b)

    #  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6
    return (M, C, D)


def eval_local_clamp_spline(xeval, xi, xip, Mi, Mip, C, D):
    # Evaluates the local spline as defined in class
    # xip = x_{i+1}; xi = x_i
    # Mip = M_{i+1}; Mi = M_i

    hi = xip - xi
    yeval = (Mi * (xip - xeval) ** 3 + (xeval - xi) ** 3 * Mip) / (6 * hi) \
            + C * (xip - xeval) + D * (xeval - xi)
    return yeval


def eval_cubic_clamped_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval + 1)

    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp = xint[j + 1]

        #   find indices of values of xeval in the interval
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

        # evaluate the spline
        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j + 1], C[j], D[j])
        #        print('yloc = ', yloc)
        #   copy into yeval
        yeval[ind] = yloc

    return (yeval)


driver()

