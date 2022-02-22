from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

#model.trainable = False
    
#model.layers[0].trainable = False # Dense    (Summary 상)  각 layer마다 동결시킬 수 있다!!
#model.layers[1].trainable = False # Dense_1
model.layers[2].trainable = False # Dense_2

model.summary()

print(model.layers)

#  1번째 Dense                                                      2번째 Dense                                                    3번째 Dense
# [<keras.layers.core.dense.Dense object at 0x0000025313EBCDC0>, <keras.layers.core.dense.Dense object at 0x00000253208379D0>, <keras.layers.core.dense.Dense object at 0x0000025326CB2F10>]