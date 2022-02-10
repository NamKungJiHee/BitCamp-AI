import numpy as np
import matplotlib.pyplot as plt

def selu(x, alpha, scale):
    return np.where (x <= 0, scale * alpha * (np.exp(x)-1), scale * x)

x = np.arange(-5, 5, 0.1)
alpha = 1
scale = 1
y = selu(x, alpha, scale)

plt.plot(x, y)
plt.grid()
plt.show()

# Selu(Scaled Exponential Linear Unit): 알파값으로 2개를 파라미터로 넣어 학습시키면, 활성화 함수의 분산이 일정하게 나와 성능이 좋아진다.
# 하지만 알파 값에 따라 활성화 함수의 결과값이 일정하지 않아 층을 많이 쌓을 수 없다고 한다.