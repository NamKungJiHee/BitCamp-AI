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

