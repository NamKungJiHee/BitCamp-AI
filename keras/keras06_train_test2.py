from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#01.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

## 과제 ##
#train과 test의 비율을 8:2으로 분리하시오.
x_train = x[:8]
x_test = x[8:]
y_train = y[:8]
y_test = y[8:]


print(x_train)
print(x_test)
print(y_train)
print(y_test)
