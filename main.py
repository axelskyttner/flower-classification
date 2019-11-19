from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--petal_length', metavar='petal_length', type=float, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--petal_width', metavar='petal_width', type=float, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sepal_length', metavar='sepal_length', type=float, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sepal_width', metavar='sepal_width', type=float, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
petal_length = args.petal_length[0]
petal_width = args.petal_width[0]
sepal_length = args.sepal_length[0]
sepal_width = args.sepal_width[0]

print(petal_length, petal_width, sepal_length, sepal_width)

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = read_csv('data/training-data.csv', names=names)
print(iris)
X = iris[:, :4]
print(X)
print(iris.target_names)

Y = iris[:, 4]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

my_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

n_neighbours = 25

predictor = KNeighborsClassifier(n_neighbours)
predictor.fit(X_train, Y_train)

res = predictor.predict(my_data)
print(iris.target_names[res])
