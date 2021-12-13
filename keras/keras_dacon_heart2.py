import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns  
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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

test_file = test_file.drop(['id'], axis=1) 
#y = y.to_numpy()

#print(train.info())    
#print(train.describe())  

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)


model = Sequential()
model.add(Dense(90, input_dim=13))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1,activation='sigmoid'))  # sigmoid는 output이 1개!



#3)컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam')

es = EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'min', verbose=1, restore_best_weights=True) 

model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.25, callbacks=[es]) 

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred = y_pred.round(0).astype(int)   # argmax효과
f1 = f1_score(y_pred, y_test)
print('f1 스코어', f1)

results = model.predict(test_file)
results = results.round(0).astype(int)

####  제출용 ####

#print(results[:5])
submit_file['target'] = results
submit_file.to_csv(path + "subfile.csv", index=False)

# f1 스코어 0.8205128205128205 