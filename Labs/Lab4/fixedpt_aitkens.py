# import libraries
import numpy as np


def driver():

# test functions 
     f1 = lambda x: (10/(x+4))**(1/2)
# fixed point is x = 1.3652300134140976

     f2 = lambda x: x - ((x ** 5 - 7) / 12)
# fixed point is x = 7^(1/5)

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     [xstar,ier,x,n] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
     print(f"This list of itseratios is: {x}")
     print(f"The number of iterations is {n}")
     fit = compute_order(x,xstar)
     print("Fit is:",fit)
     fit = compute_order(aitkens(x,tol,Nmax),xstar)
     print("Aitkens Fit is:",fit)
    
#test f2 '''
     x0 = 1.0
     [xstar,ier,x,n] = fixedpt(f2,x0,tol,Nmax)
     print("################################################################")
     print('the approximate fixed point is:',xstar)
     print('f2(xstar):',f2(xstar))
     print('Error message reads:',ier)
     print(f"This list of itseratios is: {x}")
     print(f"The number of iterations is {n}")


# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    x = [x0]
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       x.append(x1)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          x=np.array(x)
          n = len(x) - 1
          return [xstar,ier,x,n]
       x0 = x1

    xstar = x1
    ier = 1
    x = np.array(x)
    n = len(x) - 1
    return [xstar, ier,x,n]

def compute_order(x, xstar):
    """Approximates order of convergence given:
    x: array of iterate values
    xstar: fixed point/solution"""
    # |x_n+1 - x*|
    diff1 = np.abs(x[1:-1:]-xstar)

    # |x_n - x*|
    diff2 = np.abs(x[0:-2]-xstar)

    #take linear fit of logs to find slope (alpha) and intercept
    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    _lambda = np.exp(fit[1])
    alpha = fit[0]
    print(f"lambda is {_lambda}")
    print(f"alpha is {alpha}")
    return fit

def aitkens(x,tol,Nmax):
    pn1 = x[1:-1:]
    pn = x[0:-2:]
    pn2 = x[2::]
    p = pn - ((pn1 - pn)** 2)/ (pn2 - 2 * pn1 + pn)
    return p

driver()