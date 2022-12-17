import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plot_decision_function(classifier, sample_weight, axis, title):
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(-8, 8, 500), np.linspace(-8, 8, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=100 * sample_weight,
        alpha=0.9,
        cmap=plt.cm.bone,
        edgecolors="black",
    )

    axis.axis("off")
    axis.set_title(title)


# we create 20 points
np.random.seed(0)
# X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
# y = [1] * 10 + [-1] * 10
# X = np.array([[0, 0], [0, 2], [0, 4], [2, 4], [4, 4], [4, 2], [4, 0], [2, 0]])
# y = np.array([0, 1, 0, 1, 0, 1, 0 , 1])

X = np.array([[0.2, 0.553519174456596],
              [0.3, 0.830065906047821],
              [0.2, 0.999999940395355],
              [0.15, 0.983746290206909],
              [0.5, 1.0],
              [0.1, 0.960313141345978],
              [0.04, 0.4],
              [0.1, 0.985110819339752]])
y = np.array([0, 1, 0, 1, 1, 1, 0, 0])

print(X)
sample_weight_last_ten = abs(np.random.randn(len(X)))
sample_weight_constant = np.ones(len(X))
# and bigger weights to some outliers
# sample_weight_last_ten[15:] *= 5
# sample_weight_last_ten[3] *= 15

sample_weight_last_ten= np.array([1, 1, 5, 1, 1, 0.1, 1, 1])
print(sample_weight_last_ten)
print(sample_weight_constant)

# for reference, first fit without sample weights

# fit the model
clf_weights = svm.SVC(gamma=1)
clf_weights.fit(X, y)

clf_no_weights = svm.SVC(gamma=1)
clf_no_weights.fit(X, y)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
(plot_decision_function(
    clf_no_weights, sample_weight_constant, axes[0], "Constant weights"
))
plot_decision_function(clf_weights, sample_weight_last_ten, axes[1], "Modified weights")

plt.show()