Here's an enhanced and detailed `README.md` structured according to CRISP-DM, providing thorough explanations for each section and file.

---

# HW3 - Comparison of Logistic Regression and SVM on Synthetic Datasets

This project demonstrates the comparison between Logistic Regression and Support Vector Machine (SVM) models on synthetic datasets with varying dimensions and shapes. The project includes interactive visualization using Streamlit, allowing users to understand the impact of tuning SVM parameters on model performance.

## CRISP-DM Methodology

The project is structured following the CRISP-DM methodology, which provides a structured approach to tackle data science problems.

### 1. Business Understanding

The purpose of this project is to explore and compare the effectiveness of two popular machine learning algorithms, Logistic Regression and SVM, across different types of synthetic data distributions. The objectives are:
- To understand how Logistic Regression and SVM differ in handling 1D and 2D data.
- To analyze SVM’s behavior on circular and non-circular 2D distributions using various parameter settings.
- To make the model behavior interpretable by allowing parameter adjustments in a Streamlit app, enhancing learning and comprehension.

### 2. Data Understanding

We use synthetic data in three different setups to simulate varied data distributions:
- **1D Simple Case**: A one-dimensional dataset for a straightforward comparison between Logistic Regression and SVM.
- **2D Circular Distribution**: A two-dimensional dataset where data points are distributed in a circular pattern, allowing us to observe SVM’s performance with radial kernel functions.
- **2D Non-Circular Distribution**: A two-dimensional dataset with a non-circular (moon-shaped) distribution, demonstrating the advantages of SVM’s non-linear decision boundaries on complex data shapes.

Each dataset is generated within the script, making it reproducible and eliminating the need for external data files.

### 3. Data Preparation

Since the datasets are synthetically generated, minimal data preparation is required:
- **1D Data**: Data points are generated along a single feature axis with binary labels based on a threshold.
- **2D Circular Data**: Data points are generated in a circular pattern with binary XOR logic, producing a more complex distribution.
- **2D Non-Circular Data**: Data points are generated in two interlocking moon shapes to test non-linear separability.

### 4. Modeling

Each Python script represents a different modeling experiment with specific model settings:

1. **HW3-1.py**: Logistic Regression vs. SVM on 1D Data
   - **Description**: Compares the two models on a simple 1D dataset where data points are separated along a single feature.
   - **Models**:
     - Logistic Regression: A linear model that predicts probabilities based on a logistic function.
     - SVM: A linear SVM model that finds an optimal hyperplane for classification.
   - **Output**: A plot comparing the decision boundaries of both models, showing where they differ in classification on 1D data.

2. **HW3-2.py**: SVM on 2D Circular Data with Streamlit (Interactive 3D Plot)
   - **Description**: Uses SVM with an RBF (Radial Basis Function) kernel on a circular distribution and visualizes the decision function in a 3D plot.
   - **Model**:
     - SVM (RBF Kernel): Allows for non-linear decision boundaries, suited to circular data patterns.
   - **Interactivity**:
     - `gamma`: Controls the influence of each data point in the RBF kernel, adjustable via a slider in Streamlit.
     - `mesh_limit`: Adjusts the zoom level for the decision boundary plot.
     - `elevation` and `azimuth`: Control the viewing angle of the 3D plot, enabling rotation.
   - **Output**: An interactive 3D plot in Streamlit, where users can modify parameters and observe how `gamma` and viewing angles affect the decision surface.

3. **HW3-3.py**: SVM on 2D Non-Circular Data
   - **Description**: Applies SVM with an RBF kernel on a non-circular (moon-shaped) distribution.
   - **Model**:
     - SVM (RBF Kernel): Non-linear kernel to handle complex distributions, highlighting the decision boundary flexibility.
   - **Output**: A 2D contour plot showing the decision boundary on the moon-shaped data, illustrating the model's effectiveness in separating non-linearly distributed data points.

### 5. Evaluation

The evaluation metrics and visualizations focus on understanding model behavior across different data types:
- **HW3-1**: Compares model accuracy and visually inspects decision boundaries. This illustrates how SVM and Logistic Regression differ in classification when features are linearly separable.
- **HW3-2**: The Streamlit app provides interactive visualization of SVM's decision surface in 3D, with `gamma` adjustments affecting decision boundary curvature. This helps in visualizing SVM's behavior on circular data.
- **HW3-3**: Shows the SVM decision boundary on a non-circular dataset, demonstrating the power of non-linear kernels for separating complex patterns.

### 6. Deployment

Each file can be executed independently:

- **HW3-1.py**: Run in a Python environment to compare Logistic Regression and SVM on 1D data:
   ```bash
   python HW3-1.py
   ```

- **HW3-2.py**: Run with Streamlit to deploy an interactive 3D plot for SVM on circular data:
   ```bash
   streamlit run HW3-2.py
   ```
   - This opens an interactive interface where users can adjust SVM parameters like `gamma`, `mesh_limit`, and view angles to explore the decision boundary.

- **HW3-3.py**: Run in a Python environment to see the SVM's decision boundary on a non-circular 2D dataset:
   ```bash
   python HW3-3.py
   ```

## Requirements

To run these scripts, ensure you have the following dependencies installed:
- **Python 3.x**
- **Libraries**:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `streamlit` (only for `HW3-2.py`)

Install dependencies using:

```bash
pip install numpy matplotlib scikit-learn streamlit
```

## Usage

1. **Download** the files in this repository.
2. **Run the scripts** as per the deployment steps above to explore the different modeling approaches and visualizations.
3. Use the **Streamlit interface** in `HW3-2.py` to adjust parameters interactively, gaining insight into how SVM’s `gamma` parameter and viewing angles affect the decision boundary.

## Insights and Learnings

This project demonstrates how different models and parameters perform on varied data distributions:
- Logistic Regression works well on linearly separable data but struggles with complex patterns.
- SVM, especially with RBF kernel, adapts to complex data distributions like circular and moon shapes, showcasing its flexibility.
- Interactive visualization through Streamlit provides a deeper understanding of SVM’s behavior, helping users grasp the impact of parameter tuning on model performance.

This repository is a practical guide to exploring machine learning models on synthetic data and understanding model interpretability through visualization.

--- 

This `README.md` provides comprehensive guidance on the purpose, methodology, and usage of each file in the project, making it easier for others to understand and run the code effectively.
