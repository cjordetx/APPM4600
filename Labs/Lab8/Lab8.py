import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
N = 11;
A = np.zeros((N + 1, N + 1))
print(f'this is diagonal{np.diag(A, k=int(N))}')