from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights = 'imagenet', include_top= False, input_shape= (32, 32, 3))

# vgg16.trainable = False  # 가중치를 동결시킨다.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

# model.trainable = False

model.summary()
         # Trainable = True 상태    vgg false  model false
print(len(model.weights))  # 30       -> 30      -> 30
print(len(model.trainable_weights))# 30 -> 4     -> 0  

######################### 2번에서 아래만 추가 #########################
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

''' 
  pd.set_option('max_colwidth', -1)
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x00000280DC842580>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x00000280DC838310>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x00000280DC895E80>       dense      True
3  <keras.layers.core.dense.Dense object at 0x00000280DC89B610>       dense_1    True
'''