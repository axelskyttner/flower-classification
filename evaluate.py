from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
import argparse
from joblib import dump, load

le = preprocessing.LabelEncoder()

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = read_csv('data/test-data.csv')
X = iris.values[:, :4]

Y = iris.values[:, -1]
le.fit(Y)

Y_numeric = le.transform(Y)

predictor2 = load('models/model.joblib')
pred = predictor2.predict(X)

print(pred)
print(Y_numeric)
score = accuracy_score(Y_numeric, pred)
print(score)
