# Homework 1
# Christian Ordetx

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import random

## Problem 1

### part i.

x = np.arange(1.920, 2.080, 0.001); #creates a list of values in the interval [1.920, 2.080] with step 0.001

papprox = x ** 9 - 18 * x ** 8 + 144 * x ** 7 - 672 * x ** 6 + 2016 * x ** 5 - 4032 * x ** 4 + 5376 * x ** 3 - 4608 * x ** 2 + 2304 * x - 512; #expanded polynomial of p(x)

plt.plot(x, papprox) #plots expanded polynomial of p(x)
plt.show()

### part ii.

p = (x - 2 ) ** 9; #factored polynomial of p(x)

plt.plot(x, p) #plots factored form of p(x)
plt.show()

### part iii.

#### Comparing the graph of the expanded polynomial p(x) and the graph of the factored polynomial p(x) we can see that the two graphs differ drastically. The factored form creates a smooth continuous looking curve while the graph of the expanded form is very rough and we can visually see that the plot has a discrete amount of plotted points. I conclude that this difference between the two is due to the fact that the machine must compute more calculations accumulating more error when calculating the expanded form than the factored form. I can conclude that the graph of the factored form is the more accurate graph as a polynomial of degree n can have up to (n-1) turning points and the graph of the expanded form obviously has more than 8 turning points for the polynomial p(x) of degree 9.

## Problem 2

### NOTE PROBLEM 2 WAS DONE BY HAND AND IS BELOW

## Problem 3

def f(x0):

	return (1 + x0 + x0 ** 3) * math.cos(x0)

f0 = f(0);

def df(x0):

	return (1 + 3 * x0 ** 2) * math.cos(x0) - (1 + x0 + x0 ** 3) * math.sin(x0);
	
df0 = df(0);

def d2f(x0):
	return 6 * x0 * math.cos(x0) - (1 + 3 * x0 ** 2)  * math.sin(x0) -(1 + 3 * x0 ** 2) * math.sin(x0) - (1 + x0 + x0 ** 3) * math.cos(x0);

d2f0 = d2f(0);

def P2f(x0, x):

	return f0 + (df0 / math.factorial(1)) * (x - x0) + (d2f0 / math.factorial(2)) * ((x - x0) ** 2)

### part a.

print("The approximation for f(0.5) using a second degree taylor polynomial about x = 0 is:", P2f(0, 0.5))

print("This is f(0.5):", f(0.5))

error = abs(f(0.5) - P2f(0, 0.5));

print("this is the upper bound error and the greatest discrepency we can have for a second degree taylor polynomial approximation of f(x) about x = 0 evaluated at x = 0.5:", error)

### part b.

def errbnd(f, P2f):
	return abs(f -P2f)

errorcheck = errbnd(f(0.5), P2f(0, 0.5));

print(errorcheck) #prints the error between f(0.5) and P2(0.5) using the error function to verify its correct

### part c.

del x
x0 = 0;
P2fx0 = lambda x: f0 + (df0 / math.factorial(1)) * (x) + (d2f0 / math.factorial(2)) * ((x) ** 2)

integral = quad(P2fx0, 0, 1)

print("This is the approximation for the integral of f(x) from 0 to 1 with respect to x using P2(x):", integral[0])

### part d. 

print("This is the error in the approximation for the integral of f(x) from 0 to 1 with respect to x using P2(x):", integral[1])

## Problem 4

a = 1;
b = -56;
c = 1;

quadratic = lambda x: a * x ** 2 + b * x + c

### part a.

r1loss = (-b + round(math.sqrt(b ** 2 - 4 * a * c), 3)) / 2 * a;
r2loss = (-b - round(math.sqrt(b ** 2 - 4 * a * c), 3)) / 2 * a;

print("r1 loss:", r1loss)
print("r2 loss:", r2loss)

r1 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a);
r2 = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a);

print("r1:", r1, "r2:", r2)

# We can see that there is not an error between the first root with loss of precision and the actual value for accounting for up to 3 decimal places since the value for the first root is relatively large compared to the second root. We can however see that there is a error in the second root lost due to precision error. If we were able to only see each value rounded to 3 decimal places, we would not notice this loss of precision but that does not mean it doesn't exist as we have just shown.

### part b.

r1actual = (2 * c) / (-b - round(math.sqrt(b ** 2 - 4 * a * c), 3));
r2actual = (2 * c) / (-b + round(math.sqrt(b ** 2 - 4 * a * c), 3));

print("actual value of r1:", r1actual, "actual value of r2:", r2actual)

# We can see now that the second root is accurate to 6 digits of precision which is way more than we had before. Therefore this computation produces a more accurate second root. Using this process for the first root however produces a less accurate root. Therefore we can conclude that this computation is better for small roots near x = 0 while the quadratic formula is better for larger roots away from x = 0.

## Problem 5

x1 = 0.001;
x2 = 0.002;

deltax1 = 0.003;
deltax2 = 0.001;

y = x1 + x2;

print("this is y:", y)

yalt = y + (deltax1 - deltax2);

print("this is y tilda:", yalt)

# Setting x1 = 0.1, x2 = 0.2, deltax1 = 0.003, and deltax2 = 0.001, we obtain that y = 0.30000000000000004 and ytilda = 0.30200000000000005

# Setting x1 = 100, x2 = 200, and keeping delta1/2 the same, we obtain that y = 300 and ytilda = 300.002

# Comparing large values of x and small values of x as inputs, we can see that for small x, the error accumulated while small is greater than the error accumulated for large values of x. I think this has to do with the total digits of precision allowed. For small x, the machine has more digits of precision in the decimal region while for large x, some of those digits of precision are traded for integer digits of precision.

### part a.

abserror = abs(yalt - y); #computes the absolute error of y or absolute value of deltay

relerror = abs(yalt - y) / abs(y);

print("Absolute Error:", abserror) 

print("Relative Error:", relerror)

#After testing the error for large and small x, I found that the relative error is signifigently greater for small x than large x as for small x, y is small and a small denominator makes a large overall value and therefore the relative error is larger for small x.

### part b.

def cosalt(x, delta):
	return math.cos(x + delta) + math.cos(x + math.pi); #manipulated expression of cos(x + delta) - cos(x)
	
delta = []
	
for n in range(0,17): #calculates deltas and puts them into a list
	if n == 0:
		delta.append(1)
	else:
		delta.append(10 ** (-n))
		
delta = delta[::-1]; #reverses the delta list order

cospi = []

for n in delta:
	cospi.append(cosalt(math.pi, n)) #manipulated function evalulated at x = pi for all deltas
	
cos10 = []

for n in delta:
	cos10.append(cosalt(10 ** 6, n)) #manipulated function evalulated at x = 10 ** 6 for all deltas
	
plt.plot(delta, cospi) #plots manipulated function at x = pi as a function of delta
plt.show()

plt.plot(delta, cos10) #plots manipulated function at x = 10 ** 6 as a function of delta
plt.show()

# We can see for small x as delta gets greater, the error in the output increases but for large x as delta gets greater, the error actually gets smaller and the error is instead larger for small delta. Therefore small x and small delta as well as large x and large delta lead to small error outputs while small x and large delta as well as large x and small delta lead to large error outputs.

### part c.

def costaylor(x, delta):
	return delta * (-math.sin(x)) + ((delta ** 2) / 2) * (-math.cos(random.uniform(x, x + delta))) #Taylor expansion of the given function choosing a random float between x and x + delta for the cursive epsilon to evaluate
	
costaylorpi = []

for n in delta:
	costaylorpi.append(costaylor(math.pi, n))

costaylor10 = []

for n in delta:
	costaylor10.append(costaylor(10 ** 6, n))
	
plt.plot(delta, costaylorpi)
plt.show()

plt.plot(delta, costaylor10)
plt.show()

#Comparing the graphs of taylors method with the method I used removing the subtraction sign from the function by the identity -cos(x) = cos(x + pi), I don't notice any differences besides in the x=pi graphs where the value of the difference is greater but I assume that could be due to the fact that the epsilon cosen to evaluate cos at is random and changes with every calculation. Not sure if this is supposed to be fixed to one value for all values of delta or is also a parameter free to change as delta changes as long as it is in the interval [x, x + delta].

