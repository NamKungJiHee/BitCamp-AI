# 파라미터의 수
# (3, 3) 필터 한개에는 3 x 3 = 9개의 파라미터가 있음(Numpy연산 방식으로 이해)
# parameter 공식 : (3*3) * 입력채널(장수)*출력채널 + 상수(바이어스) = 노드...

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 5)           455
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 7)           147
_________________________________________________________________

# 1. (2*2) * 1(한겹) * 10(출력채널) + 10(바이어스) =50
# 2. (3*3) * 10 (4,4,10으로 바꼈은까) * 5 + 5 = 455
# 커널이 하나의 파라미터

"""

# EX) model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10, 10, 1)))             
# 10 = 출력채널
# Filter와 Kernel은 같음 
# 1 = channel : 컬러 이미지는 3개의 채널로 구성



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout  #Convolution


'''
model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5, 5, 1)))     # ===> (N,4,4,10)이 된다
model.add(Conv2D(5, (2,2), activation = 'relu'))     # ===> (3,3,5)가 된다
model.add(Conv2D(7, (2,2), activation = 'relu'))     # ===> (2,2,7)가 된다
                                                                # 이미지  /  kernel_size = 자르는 사이즈 / 가로 x 세로 x 장수<컬러> (장수가 1이므로 흑백사진)
                                                                   # Conv2D는 무조건 3차원
'''

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10, 10, 1)))     # 9,9,10      
model.add(Conv2D(5, (3,3), activation = 'relu'))     # 7,7,5          
model.add(Conv2D(7, (2,2), activation = 'relu'))       # 6,6,7
model.add(Flatten())    # 4차원에서 2차원으로 만들어주려고 펴주는 기능
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation = 'softmax'))


                                       
model.summary()

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 5)           455
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 7)           147
=================================================================
Total params: 652
Trainable params: 652
Non-trainable params: 0

"""