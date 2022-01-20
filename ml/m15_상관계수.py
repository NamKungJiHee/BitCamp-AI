import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1) 데이터
datasets = load_iris()
#print(datasets.DESCR) 
#print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target

#df = pd.DataFrame(x, columns=datasets['feature_names'])  # x에 컬럼 이름을 넣어주기
#df = pd.DataFrame(x, columns=datasets.feature_names)  # x에 컬럼 이름을 넣어주기
df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])  # x에 컬럼 이름을 넣어주기
print(df)

df['Target(Y)'] = y  # y컬럼 추가
print(df)

print("=====상관계수 히트 맵 =====")
print(df.corr())

''' 
=====상관계수 히트 맵 =====
                  sepal length (cm) sepal width (cm) petal length (cm) petal width (cm) Target(Y)
sepal length (cm)          1.000000        -0.117570          0.871754         0.817941  0.782561
sepal width (cm)          -0.117570         1.000000         -0.428440        -0.366126 -0.426658
petal length (cm)          0.871754        -0.428440          1.000000         0.962865  0.949035
petal width (cm)           0.817941        -0.366126          0.962865         1.000000  0.956547
Target(Y)                  0.782561        -0.426658          0.949035         0.956547  1.000000
'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

