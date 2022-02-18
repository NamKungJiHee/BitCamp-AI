import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

#model = VGG16()
model = VGG16(weights = 'imagenet', include_top = False, input_shape = (32, 32, 3)) # include_top : 가장 상단의 fully connected계층들을 포함 시킬지의 여부
                                                                                    # include_top = True가 default
                                                                                    # include_top = False로 주면 내가 원하는 shape로 change할 수 있다.
model.summary()

print(len(model.weights))  # 32 즉 layer가 16개 라는 뜻!
print(len(model.trainable_weights))   # 32

""" 
 ########################### include_top = True ###########################
 1. FC layer 원래꺼 그대로 쓴다.
 2. input_shape = (224, 224, 3)으로 고정 = 즉 바꾸지 못한다. / output도 1000개로 고정된다.
 
  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928
 ===================================================================
  fc2 (Dense)                 (None, 4096)              16781312          ====> 이 부분 차이!!!

 predictions (Dense)         (None, 1000)              4097000

=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
########################### include_top = False ###########################
 1. FC layer 원래꺼 삭제! = 앞으로 난 customize할거닷!
 2. input_shape = (32, 32, 3)으로 바꿀 수 있다. = customize할 수 있다.
 
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0

 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928
 ===================================================================
 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0
 ===================================================================
 Total params: 14,714,688
 Trainable params: 14,714,688
 Non-trainable params: 0
"""
# 점심과제) FC layer에 대해 정리해놓기!!
'''
Fully connected layer의 목적은 Convolution/Pooling 프로세스의 결과를 취하여 이미지를 정의된 라벨로 분류하는 데 사용하는 것
Fully Connected Layer(이하 FCL)은 이미지를 분류하는 인공신경망
완전히 연결 되었다라는 뜻으로,

한층의 모든 뉴런이 다음층의 모든 뉴런과 연결된 상태로

2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층
'''