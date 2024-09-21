# import libraries
import numpy as np

def driver():

# use routines    
    f1 = lambda x: 2*x - 1 - np.sin(x)
    a1 = 0
    b1 = 1

    tol1 = 1e-8

    [astar,ier,count] = bisection(f1,a1,b1,tol1)
    print('the approximate root for f1 is',astar)
    print('the error message reads:',ier)
    print('f1(astar) =', f1(astar))
    print(f"the total number of iterations is {count}")
    print("\n")

    f2 = lambda x: (x-5) ** 9
    a2 = 4.82
    b2 = 5.2

    tol2 = 1e-4

    [astar,ier,count] = bisection(f2,a2,b2,tol2)
    print('the approximate root for f2 is',astar)
    print('the error message reads:',ier)
    print('f2(astar) =', f2(astar))
    print(f"the total number of iterations is {count}")
    print("\n")


    f3 = lambda x: x**9 - 45*(x**8) + 900*(x**7) - 10500*(x**6) + 78750*(x**5) - 393750*(x**4) + 1312500*(x**3) - 2812500*(x**2) + 3515625*x - 1953125
    a3 = 4.82
    b3 = 5.2

    tol3 = 1e-4

    [astar, ier, count] = bisection(f3, a3, b3, tol3)
    print('the approximate root for f3 is', astar)
    print('the error message reads:', ier)
    print('f3(astar) =', f3(astar))
    print(f"the total number of iterations is {count}")
    print("\n")

    f4 = lambda x: x**3 + x - 4
    a4 = 1.0
    b4 = 4.0

    tol4 = 1e-3

    [astar, ier, count] = bisection(f4, a4, b4, tol4)
    print('the approximate root for f4 is', astar)
    print('the error message reads:', ier)
    print('f4(astar) =', f4(astar))
    print(f"the total number of iterations is {count}")
    print("\n")


# define routines
def bisection(f,a,b,tol):
    
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
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]

    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)

      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]
      
driver()               

