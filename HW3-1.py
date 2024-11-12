import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate 1D synthetic data
np.random.seed(0)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = (X > 5).astype(int).ravel()  # Binary labels based on threshold

# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X, y)
y_pred_log = log_reg.predict(X)

# Train SVM
svm_clf = SVC(kernel='linear')
svm_clf.fit(X, y)
y_pred_svm = svm_clf.predict(X)

# Calculate accuracy
accuracy_log = accuracy_score(y, y_pred_log)
accuracy_svm = accuracy_score(y, y_pred_svm)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, log_reg.predict_proba(X)[:, 1], color="green", label="Logistic Regression")
plt.plot(X, svm_clf.decision_function(X), color="red", label="SVM Decision Boundary")
plt.xlabel("Feature")
plt.ylabel("Class")
plt.legend()
plt.title(f"Logistic Regression Acc: {accuracy_log:.2f} vs SVM Acc: {accuracy_svm:.2f}")
plt.show()


