# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:34:00 2018

@author: NI389899
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from time import time
     
def evaluate(truth, pred):
    if len(truth) != len(pred):
        return -1, -1, -1

    mat = [ [0, 0], [0, 0] ]
    MRAE, cnt = 0, 0
    for (t, p) in zip(truth, pred):
        mat[int(t == 0)][int(p == 0)] += 1
        if t > 0:
            cnt += 1
            if p == -1:
                MRAE += 1
            else:
                MRAE += abs(p - t) / t

    MRAE /= cnt
    if mat[1][1] == 0:
        F1 = 0
    else:
        precision = float(mat[1][1]) / (mat[1][1] + mat[0][1])
        recall = float(mat[1][1]) / (mat[1][1] + mat[1][0])
        F1 = precision * recall * 2 / (precision + recall)

    return F1 + (1 - MRAE), F1, MRAE

time0 = time()
trainfile = input("Enter train csv file path: ")
testfile = input("Enter test csv file path: ")

#Dataset 
dataset = pd.read_csv(trainfile)
dataset = dataset.sample(frac=1)
X = dataset.iloc[:,0:-1].values

#Pre-processing region
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
X[:, 0] = le1.fit_transform(X[:, 0])
X[:, 1] = le2.fit_transform(X[:, 1])
X[:, 3] = le3.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [0,1,3])
X = onehotencoder.fit_transform(X).toarray()
y = dataset.iloc[:,-1].values
y_1 = []
y_2 = []
y_3 = []
X_2 = []
X_3 = []
for i,j in enumerate(y):
    if (j!=0):
        y_1.append(1)
        if(j>0):
            y_2.append(1)
            X_2.append(X[i])
            y_3.append(j)
            X_3.append(X[i])
        else:
            y_2.append(-1)
            X_2.append(X[i])
    else:
        y_1.append(0)
        
X_2=np.array([np.array(x) for x in X_2])
X_3=np.array([np.array(x) for x in X_3])


#ML model region
clf = RandomForestClassifier(n_estimators = 100, random_state = 0)
clf2 = RandomForestClassifier(n_estimators = 100, random_state = 0)
clf3 = RandomForestRegressor(n_estimators = 100, random_state = 0)

#Train region
clf.fit(X,y_1)
clf2.fit(X_2,y_2)
clf3.fit(X_3,y_3)

'''
#K-Fold Region
kf = KFold(n_splits=10)   
scores = cross_val_score(clf, X, y_1, cv=kf)
avg_score = np.mean(scores)

scores2 = cross_val_score(clf2, X_2, y_2, cv=kf)
avg_score2 = np.mean(scores2)

scores3 = cross_val_score(clf3, X_3, y_3, cv=kf)
avg_score3 = np.mean(scores3)
'''

#Test region
test_dataset = pd.read_csv(testfile)

test_dataset.iloc[:, 0] = test_dataset.iloc[:, 0].map(lambda s: '<unknown>' if s not in le1.classes_ else s)
le1.classes_ = np.append(le1.classes_, '<unknown>')

test_dataset.iloc[:, 1] = test_dataset.iloc[:, 1].map(lambda s: '<unknown>' if s not in le2.classes_ else s)
le2.classes_ = np.append(le2.classes_, '<unknown>')

test_dataset.iloc[:, 3] = test_dataset.iloc[:, 3].map(lambda s: '<unknown>' if s not in le3.classes_ else s)
le1.classes_ = np.append(le3.classes_, '<unknown>')

X_test = test_dataset.iloc[:,:].values
X_test[:, 0] = le1.transform(X_test[:, 0])
X_test[:, 1] = le2.transform(X_test[:, 1])
X_test[:, 3] = le3.transform(X_test[:, 3])

X_test = onehotencoder.transform(X_test).toarray()
y_pred = clf.predict(X_test)
for j,i in enumerate(y_pred):
    if i == 1:
        y_pred[j]=clf2.predict(X_test[j].reshape(1,-1))
        if(y_pred[j]!=-1):
            y_pred[j]=clf3.predict(X_test[j].reshape(1,-1))

pred_frame = pd.DataFrame(y_pred)
pred_frame.to_csv("prediction.csv", index = False)            
#score, f1, mrae = evaluate(y,y_pred)
#print("Score:", score)
#print("Model 1 Accuracy:",avg_score)
#print("Model 2 Accuracy:",avg_score2)
#print("Model 3 Accuracy:",avg_score3)
print("Done in %0.3fs " %(time() - time0))