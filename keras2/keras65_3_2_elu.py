import numpy as np
import matplotlib.pyplot as plt

def elu(x, alp):
    return (x>0)*x + (x<=0)*(alp + (np.exp(x) -1))  

x = np.arange(-5, 5, 0.1)
y = elu(x, alp= 0.1)

plt.plot(x, y)
plt.grid()
plt.show()

# Exponential Linear Unit: 개형은 RELU와 유사하며 0보다 작은 경우는 alpha값을 이용해서 그래프를 부드럽게 만든다. Relu와 달리 ELU는 미분해도 부드럽게 이어진다.