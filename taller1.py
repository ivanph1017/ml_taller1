# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 08:39:24 2018

@author: a2203
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, model_selection, linear_model
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from pandas.plotting import scatter_matrix


def load():
    data = np.genfromtxt("dataR2.csv",delimiter=",")
    data2=pd.read_csv("dataR2.csv",names=['Age', 'BMI','Glucose',
                                          'Insulin', 'HOMA', 
                                          'Leptin', 'Adiponectin'
                                          'Resistin', 'MCP.1',
                                          'Classification'])
    # plot_corr(data2)
    
    X = np.concatenate((data[:,:-2], data[:,-1].reshape((-1, 1))), axis=1)
    #X[X == '?'] = '-99999' #bad idea does not work
    X[X == '?'] = 'NaN'
    imputer = Imputer()
    X = imputer.fit_transform(X)
    print X
    y = data[:, -1].astype(int)
    logisticR(X,y)

def plot_corr(data2):
    correlation=data2.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')

def logisticR(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    lg=linear_model.LogisticRegression(n_jobs = 10)
    lg.fit(X_train,y_train)
    predictions = lg.predict(X_test)
    print predictions
    cm=confusion_matrix(y_test, predictions)
    print(cm)
    score = lg.score(X_test, y_test)
    print(score)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
    print('Accuracy de Sklearn {}'.format(accuracy_score(y_test, predictions)))
      
def main():
    load()

if __name__=="__main__":
    main()