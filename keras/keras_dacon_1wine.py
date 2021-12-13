import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, seaborn as sns  
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = "../_data/dacon/wine/"
train = pd.read_csv(path + 'train.csv')

test_file = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path + 'sample_submission.csv')  

y = train['quality']
x = train.drop(['id', 'quality'], axis=1)  

test_file = test_file.drop(['id'], axis=1) # 'sulphates' 'citric acid', 'pH',
y = y.to_numpy()

le = LabelEncoder()
label = x['type']
le.fit(label)
x['type'] = le.transform(label)

label2 = test_file['type']
le.fit(label2) 
test_file['type'] = le.transform(label2)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

model2 = RandomForestClassifier(oob_score= True, n_estimators=910, random_state=78,verbose=1)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(test_file)

print(model2.oob_score_)
print(y_pred2)

submit_file['quality'] = y_pred2
submit_file.to_csv(path+'subfile.csv', index=False)






