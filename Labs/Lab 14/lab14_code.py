import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''
     size = [100, 500, 1000, 2000, 4000, 5000]
     result = np.zeros(6)
     count = 0
     for n in size:
          '''' N = size of system'''
          N = n
          print(N)
          ''' Right hand side'''
          b = np.random.rand(N,1)
          A = np.random.rand(N,N)

          startsolve = time.time()
          x = scila.solve(A,b)
          endsolve = time.time()
          timesolve = endsolve - startsolve

          startlu = time.time()
          lu, p = scila.lu_factor(A)
          endlu = time.time()
          timelu = endlu - startlu

          startlusolve = time.time()
          x_lu = scila.lu_solve((lu, p), b)
          endlusolve = time.time()
          timelusolve = endlusolve-startlusolve

          result[count] = timelu + timelusolve - timesolve
          count += 1
          print("Solve time:", timesolve)
          print("LU time:", timelu)
          print("LU solve time", timelusolve)
          print("LU total time", timelu + timelusolve)
          print(timelu +timelusolve - timesolve)

     
          test = np.matmul(A,x)
          test2 = np.matmul(A,x_lu)
          r = la.norm(test-b)
          r2 = la.norm(test2-b)
     
          print(r)
          print(r2)
     print("N =", size)
     print("Solve time for each N respectively:", result)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
