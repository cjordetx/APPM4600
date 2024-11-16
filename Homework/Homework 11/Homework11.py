import numpy as np
import scipy.integrate as integrate


def driver():

    f = lambda s: 1/(1 + s ** 2)
    a = -5
    b = 5

    I_ex =np.arctan(b) - np.arctan(a)
    print("I_ex", I_ex)
    I_scipy1 = integrate.quad(f, a, b, epsabs = 10 ** (-4))
    I_scipy2 = integrate.quad(f, a, b, epsabs = 10 ** (-6))
    print("scipy with tol 10 ** (-4):", I_scipy1)
    print("scipy with tol 10 ** (-6):", I_scipy2)


    n = 1291
    m = 108
    I_trap = CompTrap(a, b, n, f)
    print('I_trap= ', I_trap)

    err = abs(I_scipy1 - I_trap)

    print('scipy error = ', err)

    I_simp = CompSimp(a, b, m, f)

    print('I_simp= ', I_simp)

    err = abs(I_scipy1 - I_simp)

    print('scipy error = ', err)



def CompTrap(a, b, n, f):
    h = (b - a) / n
    xnode = a + np.arange(0, n + 1) * h

    trap = h * f(xnode[0]) * 1 / 2

    for j in range(1, n):
        trap = trap + h * f(xnode[j])
    trap = trap + 1 / 2 * h * f(xnode[n])

    return trap


def CompSimp(a, b, n, f):
    h = (b - a) / n
    xnode = a + np.arange(0, n + 1) * h
    simp = f(xnode[0])

    nhalf = n / 2
    for j in range(1, int(nhalf) + 1):
        # even part
        simp = simp + 2 * f(xnode[2 * j])
        # odd part
        simp = simp + 4 * f(xnode[2 * j - 1])
    simp = simp + f(xnode[n])

    simp = h / 3 * simp

    return simp


driver()
