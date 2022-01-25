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
model = XGBClassifier(n_estimators = 10000, learning_rate =0.025, # n_jobs = -1, 
                     max_depth = 4,         
                     min_child_weight = 1,  
                     subsample = 1,      
                     colsample_bytree = 1,  
                     reg_alpha = 1,    
                     reg_lambda = 0,
                     tree_method = 'gpu_hist',                ###  이 세부분만 추가! 
                     predictor = 'gpu_predictor',             ###
                     gpu_id = 0,                              ### 첫번째 gpu사용 / gpu_id =1: 두번째 gpu 사용
                     )  
                                                             
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

''' 
# CPU
걸린시간:  65.43204140663147
results:  0.7356
r2:  0.7356

# GPU
걸린시간:  5.076215028762817
results:  0.7357
r2:  0.7357
===========================
# n_estimators = 10000 일 때
걸린시간:  383.4351215362549
results:  0.909
r2:  0.909
'''