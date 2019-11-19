import pandas
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv('data/training-data.csv')
print(dataset.shape)
print(dataset)

print(dataset.describe())

print(dataset.head(20))
dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()
