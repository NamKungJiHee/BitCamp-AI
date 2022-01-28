from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

path = "../_data/dacon/Jobcare/"

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

submit_file = pd.read_csv(path + 'sample_submission.csv')  

train = train.drop(['id','person_rn','contents_open_dt'], axis=1) 

test = test.drop(['id', 'person_rn','contents_open_dt'], axis=1)

from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
model = RandomForestClassifier(n_estimators=1000, max_depth=300, n_jobs=-1)
#model = Perceptron(max_iter=3000, n_jobs=-1, early_stopping=True, validation_fraction=0.3,n_iter_no_change=7)
#model = KNeighborsClassifier(n_neighbors=100, leaf_size=30,n_jobs=-1)
#model = DecisionTreeClassifier(splitter="best", max_depth=1000, min_samples_split=4, min_samples_leaf=30)
x = train.iloc[:, :-1]
y = train.iloc[:, -1]

model.fit(x,y)

preds = model.predict(test)

#print(preds) # [1 0 0 ... 1 1 0]

submit_file['target'] = preds

submit_file.to_csv(path + 'sample_submission.csv', index = False)