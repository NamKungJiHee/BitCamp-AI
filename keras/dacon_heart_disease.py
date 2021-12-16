# Import library
import pandas as pd
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import csv

# Model list
def models(model):
    if model == 'knn':
        mod = KNeighborsClassifier(2)
    elif model == 'svm':
        mod = SVC(kernel="linear", C=0.025)
    elif model == 'svm2':
        mod = SVC(gamma=2, C=1)
    elif model == 'gaussian':
        mod = GaussianProcessClassifier(1.0 * RBF(1.0))
    elif model == 'tree':
        mod =  DecisionTreeClassifier(max_depth=5)
    elif model == 'forest':
        mod =  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    elif model == 'mlp':
        mod = MLPClassifier(alpha=1, max_iter=1000)
    elif model == 'adaboost':
        mod = AdaBoostClassifier()
    elif model == 'gaussianNB':
        mod = GaussianNB()
    elif model == 'qda':
        mod = QuadraticDiscriminantAnalysis()
    return mod

## Data load
datapath = "../_data/dacon/heart/"
train_data = pd.read_csv(datapath + 'train.csv').to_numpy()
test_data = pd.read_csv(datapath + 'test.csv').to_numpy()

#make model list in models function
model_list = ['knn', 'svm', 'svm2', 'gaussian', 'tree', 'forest', 'mlp', 'adaboost', 'gaussianNB', 'qda']

cnt = 0
empty_list = [] #empty list for progress bar in tqdm library
for model in tqdm(model_list, desc = 'Models are training and predicting ... '):
    empty_list.append(model) # fill empty_list to fill progress bar
    #classifier
    clf = models(model)

    #Training
    clf.fit(train_data[:,1:-1], train_data[:,-1:].T[0]) #학습할때는 id와 target을 제외하고 학습! 마지막 column이 라벨이므로 라벨로 설정!

    #Predict
    pred = clf.predict(test_data[:,1:]) #마찬가지로 예측을 할 때에도 id를 제외하고 나머지 feature들로 예측

    #Make answer sheet
    savepath = datapath + '' #정답지 저장 경로
    with open(savepath + '%s_answer.csv' % model_list[cnt], 'w', newline='') as f:
        sheet = csv.writer(f)
        sheet.writerow(['id', 'target'])
        for idx, p in enumerate(pred):
            sheet.writerow([idx+1, p])

    cnt += 1
