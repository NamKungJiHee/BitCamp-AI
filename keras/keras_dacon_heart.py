<<<<<<< HEAD
import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score,classification_report
#F1 score는 통상 이진분류에서 사용한다.
# precision - 정밀도 : '예측값'을 기준으로 한 '정답인 예측값'의 비율
# recall - 재현율 : 실제 값'을 기준으로 한 '정답인 예측값'의 비율
# f1-score - 정확도 : precision과 recall의 가중 조화평균
# support :  각 라벨의 실제 샘플 개수
#macro avg :평균에 평균을 내는 개념. 단순 평균. /  weighted avg : 각 클래스에 속하는 표본의 개수로 가중 평균을 내서 계산하는 방법


path = "../_data/dacon/heart/"
train = pd.read_csv(path + 'train.csv')

test_file = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path + 'sample_submission.csv')  

y = train['target']
x = train.drop(['id', 'target'], axis=1)  

#print(x)
#print(x.shape, y.shape) # (151, 13) (151,)
#print(np.unique(y))  # [0 1]

# print(train.feature_names)  #컬럼 이름
# print(x.DESCR)

#xx = pd.DataFrame(x, columns= x.feature_names)                     
#print(type(xx))                                      
#print(xx)                                             

#print(xx.corr())   


test_file = test_file.drop(['id'], axis=1) 
#y = y.to_numpy()

# print(train.info())    
# print(train.describe())  

#y = pd.get_dummies(y)   
#print(y.shape)   # (151, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, random_state=50)

model = Sequential()
model.add(Dense(8, input_dim=13, activation='relu')) 
model.add(Dense(30,activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(21,activation='relu'))
model.add(Dense(1,activation= 'relu'))
model.add(Dropout(0.3)) 
model.add(Dense(22, activation= 'relu'))
model.add(Dense(1,activation='sigmoid'))



#3)컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.25, callbacks=[es]) 

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

####  제출용 ####
results = model.predict(test_file)
results_int = np.argmax(results,axis=1) #결과값이 0.2 0.3 0.25 0.8 0.9 등등 이렇게 나오는데 이걸
                                        # 0.5보다 작으면 -> 0   0.5보다 크면 -> 1로 바꿔준다.
submit_file['target'] = results_int
submit_file.to_csv(path + "subfile.csv", index=False)
=======
import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score,classification_report
#F1 score는 통상 이진분류에서 사용한다.
# precision - 정밀도 : '예측값'을 기준으로 한 '정답인 예측값'의 비율
# recall - 재현율 : 실제 값'을 기준으로 한 '정답인 예측값'의 비율
# f1-score - 정확도 : precision과 recall의 가중 조화평균
# support :  각 라벨의 실제 샘플 개수
#macro avg :평균에 평균을 내는 개념. 단순 평균. /  weighted avg : 각 클래스에 속하는 표본의 개수로 가중 평균을 내서 계산하는 방법


path = "../_data/dacon/heart/"
train = pd.read_csv(path + 'train.csv')

test_file = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path + 'sample_submission.csv')  

y = train['target']
x = train.drop(['id', 'target'], axis=1)  

#print(x)
#print(x.shape, y.shape) # (151, 13) (151,)
#print(np.unique(y))  # [0 1]

# print(train.feature_names)  #컬럼 이름
# print(x.DESCR)

#xx = pd.DataFrame(x, columns= x.feature_names)                     
#print(type(xx))                                      
#print(xx)                                             

#print(xx.corr())   


test_file = test_file.drop(['id'], axis=1) 
#y = y.to_numpy()

# print(train.info())    
# print(train.describe())  

#y = pd.get_dummies(y)   
#print(y.shape)   # (151, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle=True, random_state=50)

model = Sequential()
model.add(Dense(8, input_dim=13, activation='relu')) 
model.add(Dense(30,activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(21,activation='relu'))
model.add(Dense(1,activation= 'relu'))
model.add(Dropout(0.3)) 
model.add(Dense(22, activation= 'relu'))
model.add(Dense(1,activation='sigmoid'))



#3)컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.25, callbacks=[es]) 

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

####  제출용 ####
results = model.predict(test_file)
results_int = np.argmax(results,axis=1) #결과값이 0.2 0.3 0.25 0.8 0.9 등등 이렇게 나오는데 이걸
                                        # 0.5보다 작으면 -> 0   0.5보다 크면 -> 1로 바꿔준다.
submit_file['target'] = results_int
submit_file.to_csv(path + "subfile.csv", index=False)
>>>>>>> f62920a5b2fe717b4b950597110b3151c02f0314
