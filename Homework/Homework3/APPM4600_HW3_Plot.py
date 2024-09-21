# import libraries
import numpy as np
import math
import matplotlib.pyplot as plt

x = np.linspace(-2*np.pi, 2*np.pi, 100)

y = x - 4* np.sin(2*x) - 3

plt.plot(x,y)
plt.plot(x,np.zeros((100,1)))
plt.title("f(x) plotted on the interval [-2pi,2pi]")
plt.show()