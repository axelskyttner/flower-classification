from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.decision_trees import regression_tree
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
import argparse
from joblib import dump, load
import datetime
le = preprocessing.LabelEncoder()

def get_time():
    now = datetime.datetime.now()
    return "%d%d%d-%d%d" % (now.year, now.month, now.day, now.hour, now.minute)

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = read_csv('data/training-data.csv')
X = iris.values[:, :4]
Y = iris.values[:, -1]
le.fit(Y)

Y_numeric = le.transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_numeric)

n_neighbours = 1
predictor = KNeighborsClassifier(n_neighbours)
predictor.fit(X_train, Y_train)

pred = predictor.predict(X_test)
score = accuracy_score(Y_test, pred)

scores = cross_validate(predictor, X_test, Y_test, cv=5, return_train_score=True)

print('score', scores['test_score'])
dump(predictor, 'models/model-%s.joblib' % get_time())
