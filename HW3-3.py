import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generate non-circular (moon-shaped) data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.2, random_state=0)

# Train SVM
svm_clf = SVC(kernel='rbf', gamma=0.5)
svm_clf.fit(X, y)

# Plotting
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 100), np.linspace(-1, 1.5, 100))
Z = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap="coolwarm", alpha=0.2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("2D SVM with Non-Circular Distribution")
plt.legend()
plt.show()