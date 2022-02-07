import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4*x + 6

x = np.linspace(-1, 6, 100)  # -1부터 6 사이로 # np.linspace 함수(구간 시작점, 구간 끝점, 구간 내 숫자개수)
print(x, len(x)) #100등분

y = f(x)

plt.plot(x, y, 'k-')
plt.plot(2,2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()