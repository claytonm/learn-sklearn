import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
# load iris dataset
iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

# there are three target classes
np.unique(iris_y)

# split into test and control
np.random.seed(42)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
knn.predict(iris_X_test)
iris_y_test
