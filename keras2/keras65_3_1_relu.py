import numpy as np
import matplotlib.pyplot as plt
# relu의 특징: 양수만 나온다. 그래서 음수 부분은 다 통일되지만 큰 음수값이 있으면 문제가 있음. 그래서 음수 부분을 약간 살려놓는다.

def relu(x):
    return np.maximum(0, x)  

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()