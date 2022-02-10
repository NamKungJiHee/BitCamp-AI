import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x):
    return np.maximum(0.01*x,x) 

x = np.arange(-5, 5, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# Leaky Relu: 0보다 작은 경우, 0에 근접하는 매우 작은 값으로 변환되도록 하지만 relu에 비해 연산의 복잡성이 크다.