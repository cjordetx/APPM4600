Last login: Thu Aug 29 13:19:32 on ttys000
(base) christianordetx@Christians-MacBook-Pro ~ % python3
Python 3.11.5 (main, Sep 11 2023, 08:31:25) [Clang 14.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.

# This code computes the multiplication of two different vectors through matrix multiplication
# Imported the packages numpy as np and matplotlib.pyplot as plt

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> X = np.linspace(0, 2 * np.pi, 100)
>>> Ya = np.sin(X)
>>> Yb = np.cos(X)
>>> plt.plot(X, Ya)
[<matplotlib.lines.Line2D object at 0x138c54e10>]
>>> plt.show(block=False)
>>> plt.plot(X, Yb)
[<matplotlib.lines.Line2D object at 0x138ca1190>]
>>> plt.show(block=False)
>>> plt.plot(X, Ya)
[<matplotlib.lines.Line2D object at 0x138a97e10>]
>>> plt.plot(X, Yb)
[<matplotlib.lines.Line2D object at 0x138cbffd0>]
>>> plt.xlabel(’x’)
  File "<stdin>", line 1
    plt.xlabel(’x’)
               ^
SyntaxError: invalid character '’' (U+2019)
>>> plt.ylabel('y')
Text(44.44444444444443, 0.5, 'y')
>>> plt.xlabel('x')
Text(0.5, 47.04444444444444, 'x')
>>> plt.show(block = False)
>>> X = np.linspace(0, 2 * np.pi, 100)
>>> Ya = np.sin(X)
>>> Yb = np.cos(X)
>>> plt.plot(X, Ya)
[<matplotlib.lines.Line2D object at 0x138cf0210>]
>>> plt.plot(X, Yb)
[<matplotlib.lines.Line2D object at 0x138c90790>]
>>> plt.xlabel('x')
Text(0.5, 47.04444444444444, 'x')
>>> plt.ylabel('y')
Text(44.44444444444443, 0.5, 'y')
>>> plt.show(block = False)
>>> x = np.linspace(0, 10, 10)
>>> x
array([ 0.        ,  1.11111111,  2.22222222,  3.33333333,  4.44444444,
        5.55555556,  6.66666667,  7.77777778,  8.88888889, 10.        ])
>>> x[0:3]
array([0.        , 1.11111111, 2.22222222])
>>> print('the first the entries of x are:',x[0],',',x[1],', and',x[2])
the first the entries of x are: 0.0 , 1.1111111111111112 , and 2.2222222222222223
>>> w = 10**(-np.linspace(1,10,10))
>>> w
array([1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06, 1.e-07, 1.e-08,
       1.e-09, 1.e-10])
>>> x = np.linspace(1, len(w) + 1, len(w))
>>> x
array([ 1.        ,  2.11111111,  3.22222222,  4.33333333,  5.44444444,
        6.55555556,  7.66666667,  8.77777778,  9.88888889, 11.        ])
>>> x = np.linspace(1, len(w), len(w))
>>> x
array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
>>> plt.semilogy([x], w)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/christianordetx/anaconda3/lib/python3.11/site-packages/matplotlib/pyplot.py", line 2880, in semilogy
    return gca().semilogy(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/anaconda3/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 1923, in semilogy
    return self.plot(
           ^^^^^^^^^^
  File "/Users/christianordetx/anaconda3/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 1688, in plot
    lines = [*self._get_lines(*args, data=data, **kwargs)]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/anaconda3/lib/python3.11/site-packages/matplotlib/axes/_base.py", line 311, in __call__
    yield from self._plot_args(
               ^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/anaconda3/lib/python3.11/site-packages/matplotlib/axes/_base.py", line 504, in _plot_args
    raise ValueError(f"x and y must have same first dimension, but "
ValueError: x and y must have same first dimension, but have shapes (1, 10) and (10,)

# keep geeting an error that the dimensions of the vectors are not compatible with eachother for vector multiplication
 
>>> plt.semilogy(x, w)
[<matplotlib.lines.Line2D object at 0x1387b9010>]
>>> plt.show(block = False)
>>> plt.semilogy(x, w)
[<matplotlib.lines.Line2D object at 0x14b083b10>]
>>> plt.show(block = False)
>>> plt.semilogy(x, w)
[<matplotlib.lines.Line2D object at 0x14bc924d0>]
>>> plt.xlabel('x')
Text(0.5, 47.04444444444444, 'x')
>>> plt.ylabel('w')
Text(64.52680558946396, 0.5, 'w')
>>> plt.show(block = False)
>>> s = 3 * w
>>> plt.semilogy(x, w)
[<matplotlib.lines.Line2D object at 0x14bc86f90>]
>>> plt.semilogy(x, s)
[<matplotlib.lines.Line2D object at 0x14b0c9850>]
>>> plt.legend(block = False)

# Obtain an unexpected error trying to create a legend

No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/christianordetx/anaconda3/lib/python3.11/site-packages/matplotlib/pyplot.py", line 2710, in legend
    return gca().legend(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/anaconda3/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 318, in legend
    self.legend_ = mlegend.Legend(self, handles, labels, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/anaconda3/lib/python3.11/site-packages/matplotlib/_api/deprecation.py", line 454, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
TypeError: Legend.__init__() got an unexpected keyword argument 'block'
>>> plt.legend()

# Obtain legend error again. Do I have to add in legend art directly?

No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
<matplotlib.legend.Legend object at 0x14fe4ba10>
>>> plt.semilogy(x, w)
[<matplotlib.lines.Line2D object at 0x14fe41f10>]
>>> plt.semilogy(x, s)
[<matplotlib.lines.Line2D object at 0x14bc29690>]
>>> plt.legend()
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
<matplotlib.legend.Legend object at 0x14fe41f90>
>>> plt.show(block = False)
>>> s
array([3.e-01, 3.e-02, 3.e-03, 3.e-04, 3.e-05, 3.e-06, 3.e-07, 3.e-08,
       3.e-09, 3.e-10])
>>> plt.semilogy(x, w)
[<matplotlib.lines.Line2D object at 0x14bc427d0>]
>>> plt.semilogy(x, s)
[<matplotlib.lines.Line2D object at 0x15b9e7150>]
>>> plt.xlabel('x')
Text(0.5, 47.04444444444444, 'x')
>>> plt.ylabel('w, s')
Text(64.52680558946396, 0.5, 'w, s')
>>> plt.show(block = False)
>>> exit()
(base) christianordetx@Christians-MacBook-Pro ~ % vim testDot.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 testDot.py
Traceback (most recent call last):
  File "/Users/christianordetx/testDot.py", line 34, in <module>
    driver()
  File "/Users/christianordetx/testDot.py", line 8, in driver
    x = linspace(0, np.pi, n)
        ^^^^^^^^
NameError: name 'linspace' is not defined
(base) christianordetx@Christians-MacBook-Pro ~ % testDot.py
zsh: command not found: testDot.py
(base) christianordetx@Christians-MacBook-Pro ~ % vim testDot.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 testDot.py
the dot product is :  212660.24304678725
(base) christianordetx@Christians-MacBook-Pro ~ % vim testDot.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 testDot.py
the dot product is :  0.0
(base) christianordetx@Christians-MacBook-Pro ~ % vim testDot.py
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 VectorMult.py
  File "/Users/christianordetx/VectorMult.py", line 28
    vm = np.zeros([np.size(m, 0), np.size(v, 1)]
                 ^
SyntaxError: '(' was never closed
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 VectorMult.py
Traceback (most recent call last):
  File "/Users/christianordetx/VectorMult.py", line 34, in <module>
    driver()
  File "/Users/christianordetx/VectorMult.py", line 15, in driver
    m = np.array([1, 4], [8, 2])
        ^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Field elements must be 2- or 3-tuples, got '8'
(base) christianordetx@Christians-MacBook-Pro ~ % m = np.array([1, 4], [8, 2])
zsh: unknown file attribute:  
(base) christianordetx@Christians-MacBook-Pro ~ % python3         
Python 3.11.5 (main, Sep 11 2023, 08:31:25) [Clang 14.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> m = np.array([1, 4], [8, 2])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Field elements must be 2- or 3-tuples, got '8'
>>> vim VectorMult.py
  File "<stdin>", line 1
    vim VectorMult.py
        ^^^^^^^^^^
SyntaxError: invalid syntax
>>> exit()
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 VectorMult.py
Traceback (most recent call last):
  File "/Users/christianordetx/VectorMult.py", line 34, in <module>
    driver()
  File "/Users/christianordetx/VectorMult.py", line 19, in driver
    vm = VectorMult(m, v, n)
         ^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/VectorMult.py", line 28, in VectorMult
    vm = np.zeros([np.size(m, 0), np.size(v, 1)])
                           ^
NameError: name 'm' is not defined. Did you mean: 'vm'?
(base) christianordetx@Christians-MacBook-Pro ~ % python3
Python 3.11.5 (main, Sep 11 2023, 08:31:25) [Clang 14.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> m = np.array([1, 4], [8, 2])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Field elements must be 2- or 3-tuples, got '8'
>>> m = np.array([[1, 4], [8, 2]])
>>> size(m)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'size' is not defined. Did you mean: 'slice'?
>>> np.size(m)
4
>>> np.size(m, 0)
2
>>> exit()
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 VectorMult.py
Traceback (most recent call last):
  File "/Users/christianordetx/VectorMult.py", line 30, in <module>
    driver()
  File "/Users/christianordetx/VectorMult.py", line 15, in driver
    vm = VectorMult(m, v, n)
         ^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/VectorMult.py", line 24, in VectorMult
    vm = np.zeros([m.size[0], v.size[1]])
                   ^
NameError: name 'm' is not defined. Did you mean: 'vm'?
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 VectorMult.py
Traceback (most recent call last):
  File "/Users/christianordetx/VectorMult.py", line 30, in <module>
    driver()
  File "/Users/christianordetx/VectorMult.py", line 15, in driver
    vm = VectorMult(M, V, n)
         ^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/VectorMult.py", line 24, in VectorMult
    vm = np.zeros([M.size[0], V.size[1]])
                   ~~~~~~^^^
TypeError: 'int' object is not subscriptable
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 VectorMult.py
  File "/Users/christianordetx/VectorMult.py", line 24
    vm = np.zeros(int([M.size[0]), int(V.size[1]]))
                                ^
SyntaxError: closing parenthesis ')' does not match opening parenthesis '['
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMilt.py
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 VectorMult.py
Traceback (most recent call last):
  File "/Users/christianordetx/VectorMult.py", line 30, in <module>
    driver()
  File "/Users/christianordetx/VectorMult.py", line 15, in driver
    vm = VectorMult(M, V, n)
         ^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/VectorMult.py", line 24, in VectorMult
    vm = np.zeros([int(M.size[0]), int(V.size[1])])
                       ~~~~~~^^^
TypeError: 'int' object is not subscriptable
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py
(base) christianordetx@Christians-MacBook-Pro ~ % python3 VectorMult.py
Traceback (most recent call last):
  File "/Users/christianordetx/VectorMult.py", line 30, in <module>
    driver()
  File "/Users/christianordetx/VectorMult.py", line 15, in driver
    vm = VectorMult(M, V, n)
         ^^^^^^^^^^^^^^^^^^^
  File "/Users/christianordetx/VectorMult.py", line 24, in VectorMult
    vm = np.zeros((int(M.size[0]), int(V.size[1])))
                       ~~~~~~^^^
TypeError: 'int' object is not subscriptable
(base) christianordetx@Christians-MacBook-Pro ~ % vim VectorMult.py

import numpy as np
import numpy.linalg as la
import math

def driver():

        n = 2
# this is a function handle. You can use it to define functions instead of using a subroutine
# like you have to in a true low level language

        M = np.array([[1, 4], [8, 2]])
        V = np.array([[2], [4]])

    # evaluate themultiplication of matrix m and vector v
        vm = VectorMult(M, V, n)

    # print the output
        print('the product is : ', vm)

        return

def VectorMult(V, M, n):
# Computes the dot product of the n x 1 vectors of x and y
        vm = np.zeros((int(M.size[0]), int(V.size[1])))
        for j in range(n):
           vm[j] = M[j] * V

        return vm

driver()


~                                                                                                                                
~                                                                                                                                
~                                                                                                                                
"VectorMult.py" 32L, 700B
