# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:21:24 2019

@author: zahmed
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

h = .02  # step size in the mesh

#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\classification\classifier\dataset'
fname = 'binary_dataset'

file_path = (os.path.join(path_dir, fname))
df = pd.read_csv(file_path, sep = ',',  engine = 'python' )

for i in range(df.num_peaks.count()):
    if df.num_peaks[i] != 1:
        df.num_peaks[i] = 2
#[ df.num_peaks[i]=2  for i in range(df.num_peaks.count()) if df.num_peaks[i] != 1 ]
# clean up data
print(df.isna().sum())
df = df.fillna(0)
df.drop('Unnamed: 0', axis=1, inplace =True)
df.drop('Unnamed: 0.1', axis=1, inplace =True)
df.drop('Unnamed: 0.1.1', axis=1, inplace =True)

# normalize Q variable
mean_q = df['Q'].mean()
std_q = df['Q'].std()
df['q_scale'] = (df['Q'] - mean_q)/std_q

df['target'] = df['device'].map({ 'fbg':0 , 'ring_resonator_multimode': 1    })

X = df.drop(columns = ['device', 'Q'])
y = df.target
# y = pd.factorize(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# y = label_binarize(y, classes = [0,1,2,3])

names = ["Nearest Neighbors",
         "Decision Tree", "Random Forest", 'DecisionTreeRegressor']

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features='auto'),
    ]

                          
 # iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    score = cross_val_score(clf, X_train, y_train, cv=5)
    print(clf)
    print(np.mean(score))
    print(metrics.classification_report(y_test, clf.predict(X_test)))
    # clf.predict_proba(X_test)[:,1]
    i += 1


# z = label_binarize(y, classes =[0,1,2]) # not working
# 
# X_train, X_test, z_train,z_test = train_test_split(X, z)

clf = svm.SVC(gamma = 'auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, clf.predict(X_test)))



