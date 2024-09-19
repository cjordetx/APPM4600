import math
import numpy as np
import matplotlib.pyplot as plt
import random

# Problem 4

## part a:
t = np.arange(0.0, math.pi+0.1, math.pi/30)

y = np.cos(t)

n = 31
s = 0
for i in range(n):
    s+= t[i]*y[i]

print("The sum is S:",s)

## part b:

r = 1.2
deltar = 0.1
f = 15
p = 0

theta = np.linspace(0, 2 * math.pi, 1000)
x = r * (1 + deltar * np.sin(f * theta + p)) * np.cos(theta)
yb = r * (1 + deltar * np.sin(f * theta + p)) * np.sin(theta)

plt.plot(x, yb)
plt.show()

### for loop
del r, deltar, f, p

deltar = 0.05

for i in range(1,11):
    r = i
    f = 2 + i
    p = random.uniform(0,2)
    x = r * (1 + deltar * np.sin(f * theta + p)) * np.cos(theta)
    yb = r * (1 + deltar * np.sin(f * theta + p)) * np.sin(theta)
    plt.plot(x, yb)
plt.show()
