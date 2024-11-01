import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

def driver():

#  function you want to approximate
    f = lambda x: math.exp(x)

# Interval of interest    
    a = -1
    b = 1
# weight function    
    w = lambda x: 1.

# order of approximation
    n = 2

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)
    approx = np.zeros(N+1)

    #p = eval_legendre(3, 5)
    #print(type(p))

    for kk in range(N+1):
      pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      #approx[kk] = coeffs(f, a, b, w, n, xeval[kk])

    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
    print(approx)
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err_l = abs(pval-fex)
    plt.semilogy(xeval,err_l,'ro--',label='error')
    plt.legend()
    plt.show()

def eval_legendre(n, x):
    # Initialize a list to store the values of the Legendre polynomials
    p = np.zeros(n + 1)

    # Base cases
    p[0] = 1.0  # P_0(x)
    if n > 0:
        p[1] = x  # P_1(x)

    # Use recurrence relation to compute P_n(x) for n >= 2
    for i in range(2, n + 1):
        p[i] = ((2 * i - 1) / i) * x * p[i - 1] - ((i - 1) / i) * p[i - 2]

    return p
      
def coeffs(f, a, b, w, n, x):
    eval = 0
    p = np.zeros(n + 1)
    p[0] = 1
    p[1] = eval_legendre(1, x)[1]
    for j in range(2, n + 1):
        phi_j = lambda x: eval_legendre(j, x)[j]
        phi_j_sq = lambda x: ((phi_j(x)) ** 2) * w(x)
        norm_j = lambda x: (phi_j(x) * f(x) * w(x))
        coeffs_num = quad(norm_j, a, b)
        coeffs_denom = quad(phi_j_sq, a, b)
        coeffs = coeffs_num / coeffs_denom
        eval = eval + coeffs * p[j]

    return eval

def eval_legendre_expansion(f,a,b,w,n,x): 

#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab 
  p = np.zeros(n + 1)
  # initialize the sum to 0 
  pval = 0.0
  p[0] = 1
  p[1] = eval_legendre(1, x)[1]
  for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: eval_legendre(j, x)[j]
      # make a function handle for evaluating phi_j^2(x)*w(x)
      phi_j_sq = lambda x: ((phi_j(x)) ** 2) * w(x)
      # use the quad function from scipy to evaluate normalizations
      norm_fac,err = quad(phi_j_sq,a,b)
      # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
      func_j = lambda x: (phi_j(x) * f(x) * w(x)) / norm_fac
      # use the quad function from scipy to evaluate coeffs
      aj,err = quad(func_j,a,b)
      # accumulate into pval
      pval = pval+aj*p[j] 
       
  return pval

    
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()               
