# 난 정말 시그모이드!
# 0 , 1 사이

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 밑이 자연상수 e인 지수함수의 그래프

x = np.arange(-5, 5, 0.1) # -5 ~ 5 까지의 범위   
print(len(x)) # 100

y = sigmoid(x)

plt.plot(x,y)
plt.grid()
plt.show()