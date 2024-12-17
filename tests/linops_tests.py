# %%
import matplotlib.pyplot as plt
import numpy as np
from DeconTools.operators.fftops import LinearOperator

# %%
h = np.zeros((2, 2))

h[0, 0] = 1
h[1, 0] = -1

op = LinearOperator(h, shape=(100, 100))

x = np.arange(100)
yy, xx = np.meshgrid(x, x)
s = 2.5
G = np.exp(-0.5 * (((xx - 50) / s) ** 2 + ((yy - 50) / s) ** 2))


lapG = op.dot(G)

invG = op.invdot(lapG, eps=1e-7 + 0j)

plt.imshow(lapG)
plt.show()
