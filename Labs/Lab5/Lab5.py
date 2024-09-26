# import libraries
import numpy as np
import math
        
def driver():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2

    f = lambda x: math.e ** (x**2 + 7*x -30) - 1
    deriv_f = lambda x: (2*x + 7) * math.e ** (x**2 + 7*x -30)
    double_deriv_f = lambda x: 2*math.e ** (x**2 + 7*x -30) + ((2*x + 7)**2) * math.e ** (x**2 + 7*x -30)
    a = 2
    b = 4.5

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-7

    [astar,ier, count] = bisection(f,deriv_f,double_deriv_f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'The total number of iterations is: {count}')
    print("\n")
####

    #f = lambda x: (x-2)*(x-5)*np.exp(x)
    #fp = lambda x: (x-2)*(x-5)*np.exp(x)+(2*x-7)*np.exp(x)
    p0 = astar

    Nmax = 100
    tol = 1.e-14

    (p,pstar,info,it) = newton(f,deriv_f,p0,tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('f(pstar) =', f(pstar))
    print('Number of iterations:', '%d' % it)


def newton(f,deriv_f,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/deriv_f(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]


def bisection(f, deriv_f, double_deriv_f, a, b, tol):
    #    Inputs:
    #     f,a,b       - function and endpoints of initial interval
    #      tol  - bisection stops when interval length < tol

    #    Returns:
    #      astar - approximation of root
    #      ier   - error message
    #            - ier = 1 => Failed
    #            - ier = 0 == success

    #     first verify there is a root we can find in the interval
    count = 0
    fa = f(a)
    fb = f(b);
    if (fa * fb > 0):
        ier = 1
        astar = a
        return [astar, ier, count + 1]

    #   verify end points are not a root
    if (fa == 0):
        astar = a
        ier = 0
        return [astar, ier, count + 1]

    if (fb == 0):
        astar = b
        ier = 0
        return [astar, ier, count + 1]

    d = 0.5 * (a + b)
    while (abs((-f(d) * double_deriv_f(d)) / ((deriv_f(d)) ** 2)) > 1):
        fd = f(d)
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier, count + 1]
        if (fa * fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count = count + 1
    #      print('abs(d-a) = ', abs(d-a))

    astar = d
    ier = 0
    return [astar, ier, count + 1]
        
driver()
