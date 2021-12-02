<<<<<<< HEAD
import numpy as np
import pandas as pd   


path = "./_data/titanic/"
train = pd.read_csv(path + "train.csv", index_col=0, header=0)  #0번째로 passengerid가 고고..
                                                                #header=컬럼명  / passenger가 index (데이터 x)
                                                                #index_col의 디폴트 = None
print(train[:1])   #(891, 11)
print(train.shape)   # [891 rows x 12 columns]  (891, 12)


test =  pd.read_csv(path + "test.csv", index_col=0, header=0)  
gender_submission = pd.read_csv(path + "gender_submission.csv", index_col=0, header=0)
#print(test[:1])
print(test.shape)   # (418, 11)     ---> (418, 10)
#print(gender_submission)
print(gender_submission.shape)  # (418, 2)-----> (418, 1)
#print(gender_submission)

#print(train.info())
print(train.describe()) 

















'''
         Survived      Pclass         Age       SibSp       Parch        Fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

결측치 Nan
y는 Survived
'''
=======
import numpy as np
import pandas as pd   


path = "./_data/titanic/"
train = pd.read_csv(path + "train.csv", index_col=0, header=0)  #0번째로 passengerid가 고고..
                                                                #header=컬럼명  / passenger가 index (데이터 x)
                                                                #index_col의 디폴트 = None
print(train[:1])   #(891, 11)
print(train.shape)   # [891 rows x 12 columns]  (891, 12)


test =  pd.read_csv(path + "test.csv", index_col=0, header=0)  
gender_submission = pd.read_csv(path + "gender_submission.csv", index_col=0, header=0)
#print(test[:1])
print(test.shape)   # (418, 11)     ---> (418, 10)
#print(gender_submission)
print(gender_submission.shape)  # (418, 2)-----> (418, 1)
#print(gender_submission)

#print(train.info())
print(train.describe()) 

>>>>>>> b6273d91f0d2a8bda64398dfce3bbe5e3e083b07
