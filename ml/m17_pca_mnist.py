import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
########################################
# 실습) pca를 통해 0.95 이상인 n_componenents가 몇개??
# 0.99
# 0.999
#np.argmax
########################################
(x_train, _ ), (x_test, _ ) = mnist.load_data() # y의 data는 가져오지 않을것! 그래서 _로 공백처리 해준다.

#print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis = 0) # train과 test를 행(axis = 0)으로 합치겠다 
#print(x.shape) # (70000, 28, 28) # 784
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
#print(x.shape) # (70000, 784)

#pca = PCA(n_components = 784)
pca = PCA(n_components = 784) 
x = pca.fit_transform(x)
#print(x)

pca_EVR = pca.explained_variance_ratio_  # 설명가능한 변화률
#print(pca_EVR)
#print(sum(pca_EVR))  # 1.0000000000000022

cumsum = np.cumsum(pca_EVR)
print(cumsum) # 누적합   

print(np.argmax(cumsum)+1) # 713 #####자리가 0부터 시작하므로 1 더해줘야한다. 그래야 1번째, 2번째 .. 일케 셀 수 있음 (1.0)  ########### argmax: 최댓값의 자리 반환##########
#print(np.argmax(cumsum>=0.99)+1) # 331
print(np.argmax(cumsum>=0.95) +1) # 154
#print(np.argmax(cumsum>0.999) +1 ) # 486

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# #plt.plot(pca_EVR)
# plt.grid() # 격자
# plt.show()
