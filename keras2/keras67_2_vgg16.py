from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights = 'imagenet', include_top= False, input_shape= (32, 32, 3))

#vgg16.trainable = False  # 가중치를 동결시킨다.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

#model.trainable = False

model.summary()
         # Trainable = True 상태    vgg false  model false
print(len(model.weights))  # 30       -> 30      -> 30
print(len(model.trainable_weights))# 30 -> 4     -> 0  