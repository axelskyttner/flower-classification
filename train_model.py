from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


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
print('score', score)
dump(predictor, 'models/model-%s.joblib' % get_time())
