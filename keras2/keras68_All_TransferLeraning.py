from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

# model = VGG16()
# model = VGG19()
# model = ResNet50()
# model = ResNet50V2()
# model = ResNet101()
#model = ResNet101V2()
#model = ResNet152()
#model = ResNet152V2()
#model = DenseNet121()
#model = DenseNet169()
#model = DenseNet201()
#model = InceptionV3()
#model = InceptionResNetV2()
#model = MobileNet()
#model = MobileNetV2()
#model = MobileNetV3Small()
#model = MobileNetV3Large()
#model = NASNetLarge()
#model = NASNetMobile()
#model = EfficientNetB0()
#model = EfficientNetB1()
#model = EfficientNetB7()
model = Xception()

model.trainable = False
model.summary()

print("====================================================")
#print("모델명: ", model)
print("전체 가중치 개수: ", len(model.weights))
print("훈련 가능 가중치 갯수: ", len(model.trainable_weights))

''' 
1. 모델명: VGG16
Total params: 138,357,544
Trainable params: 0
Non-trainable params: 138,357,544
_________________________________________________________________
====================================================
전체 가중치 개수:  32
훈련 가능 가중치 갯수:  0

2. 모델명: VGG19
Total params: 143,667,240
Trainable params: 0
Non-trainable params: 143,667,240
_________________________________________________________________
====================================================
전체 가중치 개수:  38
훈련 가능 가중치 갯수:  0

3. 모델명: ResNet50
Total params: 25,636,712
Trainable params: 0
Non-trainable params: 25,636,712
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  320
훈련 가능 가중치 갯수:  0

4. 모델명: ResNet50V2
Total params: 25,613,800
Trainable params: 0
Non-trainable params: 25,613,800
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  272
훈련 가능 가중치 갯수:  0

5. 모델명: ResNet101
Total params: 44,707,176
Trainable params: 0
Non-trainable params: 44,707,176
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  626
훈련 가능 가중치 갯수:  0

6. 모델명: ResNet101V2

Total params: 44,675,560
Trainable params: 0
Non-trainable params: 44,675,560
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  544
훈련 가능 가중치 갯수:  0

7. 모델명: ResNet152

Total params: 60,419,944
Trainable params: 0
Non-trainable params: 60,419,944
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  932
훈련 가능 가중치 갯수:  0

8. 모델명: ResNet152V2

Total params: 60,380,648
Trainable params: 0
Non-trainable params: 60,380,648
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  816
훈련 가능 가중치 갯수:  0

9. 모델명: DenseNet121

Total params: 8,062,504
Trainable params: 0
Non-trainable params: 8,062,504
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  606
훈련 가능 가중치 갯수:  0

10. 모델명: DenseNet169

Total params: 14,307,880
Trainable params: 0
Non-trainable params: 14,307,880
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  846
훈련 가능 가중치 갯수:  0

11. 모델명: DenseNet201

Total params: 20,242,984
Trainable params: 0
Non-trainable params: 20,242,984
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  1006
훈련 가능 가중치 갯수:  0

12. 모델명: InceptionV3

Total params: 23,851,784
Trainable params: 0
Non-trainable params: 23,851,784
__________________________________________________________________________________________________
전체 가중치 개수:  378
훈련 가능 가중치 갯수:  0

13. 모델명: InceptionResNetV2

Total params: 55,873,736
Trainable params: 0
Non-trainable params: 55,873,736
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  898
훈련 가능 가중치 갯수:  0

14. 모델명: MobileNet

Total params: 4,253,864
Trainable params: 0
Non-trainable params: 4,253,864
_________________________________________________________________
====================================================
전체 가중치 개수:  137
훈련 가능 가중치 갯수:  0

15. 모델명: MobileNetV2

Total params: 3,538,984
Trainable params: 0
Non-trainable params: 3,538,984
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  262
훈련 가능 가중치 갯수:  0

16. 모델명: MobileNetV3Large

Total params: 5,507,432
Trainable params: 0
Non-trainable params: 5,507,432
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  266
훈련 가능 가중치 갯수:  0

17. 모델명: MobileNetV3Small

Total params: 2,554,968
Trainable params: 0
Non-trainable params: 2,554,968
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  210
훈련 가능 가중치 갯수:  0

18. 모델명: NASNetLarge

Total params: 88,949,818
Trainable params: 0
Non-trainable params: 88,949,818
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  1546
훈련 가능 가중치 갯수:  0

19. 모델명: NASNetMobile

Total params: 5,326,716
Trainable params: 0
Non-trainable params: 5,326,716
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  1126
훈련 가능 가중치 갯수:  0

20. 모델명: EfficientNetB0

Total params: 5,330,571
Trainable params: 0
Non-trainable params: 5,330,571
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  314
훈련 가능 가중치 갯수:  0

21. 모델명: EfficientNetB1

Total params: 7,856,239
Trainable params: 0
Non-trainable params: 7,856,239
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  442
훈련 가능 가중치 갯수:  0

22. 모델명: EfficientNetB7

Total params: 66,658,687
Trainable params: 0
Non-trainable params: 66,658,687
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  1040
훈련 가능 가중치 갯수:  0

23. 모델명: Xception

Total params: 22,910,480
Trainable params: 0
Non-trainable params: 22,910,480
__________________________________________________________________________________________________
====================================================
전체 가중치 개수:  236
훈련 가능 가중치 갯수:  0
'''