import numpy as np
import matplotlib.pyplot as plt
# -1과 1 사이로 수렴

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()