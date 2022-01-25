from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
# import warnings
# warnings.filterwarnings('ignore')

#1. 데이터
# datasets = fetch_covtype()

# x = datasets.data
# y = datasets['target']
# #print(x.shape, y.shape) # (581012, 54) (581012,)
import pickle
path = './_save/'
datasets = pickle.load(open(path + 'm26_pickle1_save_datasets.dat', 'rb'))

x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=66, shuffle=True) 

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# import pickle
# path = './_save/'
# pickle.dump(datasets, open(path +'m26_pickle1_save_datasets.dat', 'wb')) 

#2. 모델
#model = XGBClassifier()                 
model = XGBClassifier(n_estimators = 2000, learning_rate =0.025, n_jobs = -1, 
                     max_depth = 4,         
                     min_child_weight = 1,  
                     subsample = 1,      
                     colsample_bytree = 1,  
                     reg_alpha = 1,    
                     reg_lambda = 0,)  
                                                             
#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 1,
          eval_set = [(x_train, y_train),(x_test, y_test)],  
          eval_metric= 'mlogloss',
          early_stopping_rounds= 10  
          )  
end = time.time()

print("걸린시간: ", end - start)

results = model.score(x_test, y_test)
print("results: ", round(results,4))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("r2: ", round(acc,4))

# print("=========================")
# hist = model.evals_result()
# print(hist)

''' 
걸린시간:  1315.2063262462616
results:  0.8267
r2:  0.8267
'''