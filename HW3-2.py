import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# Sidebar controls for parameters
st.sidebar.title("SVM Parameter Tuning")
gamma = st.sidebar.slider("Gamma (RBF kernel)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
mesh_limit = st.sidebar.slider("Mesh Grid Limit", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
elevation = st.sidebar.slider("Elevation Angle", min_value=0, max_value=90, value=30, step=5)
azimuth = st.sidebar.slider("Azimuth Angle", min_value=0, max_value=360, value=45, step=15)

# Generate circular data
np.random.seed(0)
X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# Train SVM with RBF kernel
svm_clf = SVC(kernel='rbf', gamma=gamma)
svm_clf.fit(X, y)

# Create mesh for plotting decision function
xx, yy = np.meshgrid(np.linspace(-mesh_limit, mesh_limit, 100), np.linspace(-mesh_limit, mesh_limit, 100))
zz = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Streamlit plot
st.title("2D SVM with Circular Distribution and Adjustable View")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz, cmap="coolwarm", edgecolor="k", alpha=0.5)
ax.scatter(X[y == 0, 0], X[y == 0, 1], 0, color='blue', label="Class 0")
ax.scatter(X[y == 1, 0], X[y == 1, 1], 0, color='red', label="Class 1")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Decision Function")
ax.legend()

# Apply rotation
ax.view_init(elev=elevation, azim=azimuth)
st.pyplot(fig)
