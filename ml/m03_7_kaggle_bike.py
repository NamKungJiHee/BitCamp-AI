from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC    # 되는지 안되는지 check
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression  # 분류모델 (0과 1 사이)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd 

#1) 데이터

path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
test_file = test_file.drop(['datetime'], axis=1)
y = train['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

#2) 모델구성 
#model = Perceptron()
#model = LinearSVC() 
#model = SVC()
#model = KNeighborsClassifier()
#model = LogisticRegression()
#model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3) 훈련
model.fit(x_train, y_train) 

#4) 평가, 예측
result = model.score(x_test, y_test)  # 결과가 accuracy로 나옴 # metrics의 개념(accuracy값을 돌려준다.)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)  

print("KNeighborsRegressor: ", result)
print("accuracy: ", r2)


""" 
1.Perceptron:  0.0021432945499081446
accuracy:  0.0021432945499081446
2. LinearSVC:  0.0021432945499081446
accuracy:  0.0021432945499081446                                               
3. SVC:  0.017146356399265157
accuracy:  0.017146356399265157        
4.KNeighborsClassifier:  0.012859767299448868
accuracy:  0.012859767299448868
5.LogisticRegression:  0.017146356399265157
accuracy:  0.017146356399265157
6.DecisionTreeClassifier:  0.006429883649724434
accuracy:  0.006429883649724434
7.RandomForestClassifier:  0.006429883649724434
accuracy:  0.006429883649724434
"""


