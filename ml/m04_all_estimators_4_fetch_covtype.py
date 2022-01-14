from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter = 'classifier')  # classifier에 대한 모든 측정기
print("allAlgorithms: ", allAlgorithms)  # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ..]
print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수:  41

#allAlgorithms = all_estimators(type_filter = 'regressor')  # regressor에 대한 모든 측정기

#print("allAlgorithms: ", allAlgorithms)  # [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), ..]
#print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수: 55

for (name, algorithm) in allAlgorithms:   # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),...
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률: ', acc)
    except:                     # 에러나는 것 빼고 계속해라.
        #continue   
        print(name, '은 에러') 
""" 
모델의 갯수:  41
AdaBoostClassifier 의 정답률:  0.5028613719095032
BaggingClassifier 의 정답률:  0.9620491725686945
BernoulliNB 의 정답률:  0.631833945767321
CalibratedClassifierCV 의 정답률:  0.7122621619063191
CategoricalNB 의 정답률:  0.6321437484402296
ClassifierChain 은 에러
ComplementNB 의 정답률:  0.6225742880992745
DecisionTreeClassifier 의 정답률:  0.9395368450040016
DummyClassifier 의 정답률:  0.48625250638968015
ExtraTreeClassifier 의 정답률:  0.8627229933822707
ExtraTreesClassifier 의 정답률:  0.9538307961068131
GaussianNB 의 정답률:  0.09079800005163378
GaussianProcessClassifier 은 에러
"""