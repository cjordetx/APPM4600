import numpy as np

# Define f and fp as lambda functions
f = lambda x: np.sin(x)
fp = lambda x: np.cos(x)

def error(f, fp, xeval, coeffs, N, Neval):
    approx = coeffs[0] * np.ones(Neval + 1)
    dapprox = np.zeros(Neval + 1)
    error = np.zeros(Neval + 1)
    derror = np.zeros(Neval + 1)

    for j in range(1, N + 1):
        for i in range(Neval + 1):
            approx[i] = approx[i] + coeffs[j] * xeval[i] ** j
            dapprox[i] = dapprox[i] + j * coeffs[j] * xeval[i] ** (j - 1)

    # Ensure you iterate over Neval + 1 (not Neval + 2)
    for i in range(Neval + 1):
        error[i] = approx[i] - f(xeval[i])  # f(x) is callable
        derror[i] = dapprox[i] - fp(xeval[i])  # fp(x) is callable

    return error, derror

# Example usage:
xeval = np.linspace(0, np.pi, 100)
coeffs = np.random.rand(10)  # Random coefficients
N = 9  # Degree of the polynomial
Neval = 99  # Number of evaluation points

error_values, derror_values = error(f, fp, xeval, coeffs, N, Neval)
print(error_values)
print(derror_values)
