from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
X = iris.data[:, :3]  # we only take the first three features.
Y = iris.target

# X = np.array([[0.2, 0.553519174456596, 0.657920807600021],
#               [0.3, 0.830065906047821, 0.686320632696152],
#               [0.2, 0.999999940395355, 1.0],
#               [0.15, 0.983746290206909, 0.827079772949219],
#               [0.5, 1.0, 0.999999940395355],
#               [0.1, 0.960313141345978, 0.769430041313171],
#               [0.04, 0.4, 0.4],
#               [0.1, 0.985110819339752, 0.704302211602529]])
# Y = np.array([0, 1, 0, 1, 1, 1, 0, 0])
print(X)
print(Y)
print(type(X))
print(type(Y))
# for i in X:
#     i[0] = i[0] * 10
#     i[1] = i[1] * 5
#     i[2] = i[2] * 5

# make it binary classification problem
X = X[np.logical_or(Y == 0, Y == 1)]
Y = Y[np.logical_or(Y == 0, Y == 1)]

model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)

# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# Solve for w3 (z)
z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

tmp = np.linspace(-5, 5, 30)
x, y = np.meshgrid(tmp, tmp)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
ax.plot_surface(x, y, z(x, y))
ax.view_init(30, 60)

print(clf.predict([[0.3, 0.4, 0.5]]))
plt.show()
