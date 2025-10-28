🧠 Principal Component Analysis (PCA) on Breast Cancer Dataset
📋 Project Overview

This project demonstrates how to perform Principal Component Analysis (PCA) on the Breast Cancer dataset from scikit-learn.
PCA is a powerful dimensionality reduction technique that helps visualize and understand high-dimensional data by converting it into a smaller number of principal components while retaining most of the important information (variance).

🎯 Objectives

Load and explore the Breast Cancer dataset

Standardize the data before applying PCA

Apply PCA to reduce data dimensions from 30 features to 2 principal components

Visualize the results using scatter plots and heatmaps

Understand how each original feature contributes to the new principal components

🧩 Technologies Used

Python 3.x

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

⚙️ Installation

Make sure Python is installed, then install the required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn

🧮 Steps Performed

Import Libraries: Import all necessary Python libraries.

Load Dataset: Use load_breast_cancer() from scikit-learn.

Create DataFrame: Convert the dataset to a pandas DataFrame.

Standardize Data: Use StandardScaler to normalize the feature values.

Apply PCA: Reduce features to 2 principal components using PCA(n_components=2).

Visualize Data:

Plot the data points in a 2D scatter plot (using the two components).

Create a heatmap to show how features contribute to each principal component.

📊 Visualization Results

PCA Scatter Plot:
Displays the two principal components with color-coded classes (malignant vs benign).
It helps visualize how well the two classes are separated in reduced dimensions.

PCA Components Heatmap:
Shows how each original feature contributes to the principal components.
Each row = one component, each column = one feature.

📈 Example Output
Shape of scaled data: (569, 30)
Shape of PCA data: (569, 2)


569 samples

30 original features

Reduced to 2 components

🧠 Key Learnings

PCA helps simplify complex datasets while preserving most of the variance.

Scaling is crucial before applying PCA to ensure fairness among features.

The first few components often capture most of the important information.

PCA is commonly used for visualization, noise reduction, and feature extraction.

🧑‍💻 Author

Name: Danish Khatri
Role: Student
Project: PCA - Dimensionality Reduction using Breast Cancer Dataset
