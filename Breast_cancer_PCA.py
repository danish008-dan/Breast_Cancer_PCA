# ==============================
# PCA on Breast Cancer Dataset
# ==============================

# Importing required libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Importing the dataset from sklearn
from sklearn.datasets import load_breast_cancer

# Loading the dataset
can = load_breast_cancer()

# Checking the type of the dataset object
print(type(can))  # usually it's a sklearn.utils.Bunch object (like a dictionary)

# Viewing all available keys in the dataset
print(can.keys())

# Printing dataset description
print(can['DESCR'])

# Creating a DataFrame from the dataset features
df = pd.DataFrame(can['data'], columns=can['feature_names'])

# Checking target names (0 = malignant, 1 = benign)
print(can['target_names'])

# PCA helps us find the principal components (new features)
# that explain most of the variance in the dataset.

# Importing StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Creating a scaler object
s = StandardScaler()

# Fitting the scaler to the data (it learns the mean and std)
s.fit(df)

# Transforming the data (scaling it)
s_data = s.transform(df)


# Applying PCA (Principal Component Analysis)

# Importing PCA class
from sklearn.decomposition import PCA

# Creating a PCA object to reduce to 2 components
pca = PCA(n_components=2)

# Fitting PCA to the scaled data
pca.fit(s_data)

# Transforming the data into the 2 principal components
x_pca = pca.transform(s_data)

# Checking the shape of scaled data (original dimensions)
print("Shape of scaled data:", s_data.shape)

# Checking the shape of PCA data (reduced dimensions)
print("Shape of PCA data:", x_pca.shape)


# Visualizing the PCA result

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=can['target'], cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - Breast Cancer Dataset')
plt.show()

# Components are not real features — they are combinations
# of the original features that explain the most variance.

# Displaying PCA components
print(pca.components_)

# Creating a DataFrame of PCA components
df_comp = pd.DataFrame(pca.components_, columns=can['feature_names'])

# Visualizing PCA components using a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_comp, cmap='plasma', annot=False)
plt.title('PCA Components Heatmap')
plt.xlabel('Original Feature Names')
plt.ylabel('Principal Components')
plt.show()
