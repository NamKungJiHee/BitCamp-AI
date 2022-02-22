from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights = 'imagenet') # include_top = True이므로 shape 수정 불가능

img_path = '../_data/cat_dog.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)   # img_to_array: 이미지 수치화
print("===================image.img_to_array(img)=======================")
print(x, '\n', x.shape) # (224, 224, 3)

x = np.expand_dims(x, axis = 0)
print("===================np.expand_dims(x, axis = 0)=======================")
print(x, '\n', x.shape) #  (1, 224, 224, 3)

x = preprocess_input(x)   ##### 가장 이상적으로 스케일링 시키기!!!#####
print("===================preprocess_input(x)=======================")
print(x, '\n', x.shape) #  (1, 224, 224, 3)

preds = model.predict(x)
print(preds, '\n', preds.shape) #  (1, 1000)

print('결과는: ', decode_predictions(preds, top = 5)[0]) # top=5는 상위 5개 뽑아주기! / decode_predictions = prediction을 복호화해주기!!
''' 
결과는:  [('n02099601', 'golden_retriever', 0.13870242), ('n02091244', 'Ibizan_hound', 0.08043555), 
('n02808304', 'bath_towel', 0.06647505), ('n02088466', 'bloodhound', 0.06608965), ('n02099712', 'Labrador_retriever', 0.051036626)]
'''