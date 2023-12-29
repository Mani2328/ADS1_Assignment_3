"""
Applied Data Science Assignment 3 by K.Manivannan
The data set from data.worldBank org. Perform cluster and prediction
for CO2 emission based on countries
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime 
import os
import seaborn as sns
import matplotlib
import errors as err
import scipy.optimize as opt
from scipy.stats import chi2

path = "API_EN.ATM.CO2E.PC_DS2_en_excel_v2_6298783.xlsx"
path1 = "WorldBank_data_extract.xlsx"

df = pd.read_excel(path1)
df.dtypes

percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df)

df = df.dropna().reset_index(drop=True)
df.dtypes

percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df)

import sklearn.cluster as cluster
import sklearn.metrics as skmet

corr = df.corr(numeric_only=True)
print(corr.round(3))

corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True , cmap = 'RdBu', annot_kws={"fontsize":20})

pd.plotting.scatter_matrix(df, figsize=(10,10), s=10)
plt.show()

# Choose CO2 vs Forest Area as they are not highly correlation. 
plt.scatter(df["CO2"],df["Forest area"], 10, marker="o")
plt.xlabel("CO2 emissions (metric tons per capita)")
plt.ylabel("Forest area (sq. km)")

plt.show()

import sklearn.preprocessing as pp

#setup a scalar object
scaler = pp.RobustScaler()
df_clust = df[["CO2", "Forest area"]]

#extract columns
scaler.fit(df_clust)

#apply the scaling
norm = scaler.transform(df_clust)
print(norm)

plt.figure(figsize=(8, 8))

plt.scatter(norm[:,0], norm[:,1], 10, marker="o")

plt.xlabel("CO2 emissions (metric tons per capita)")
plt.ylabel("Forest area (sq. km)")
plt.show()

def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score


# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}") 

# set up the clusterer with the number of expected clusters
cm = matplotlib.colormaps["Paired"]
kmeans = cluster.KMeans(n_clusters=7, n_init=20)

# Fit the data, results are stored in the kmeans object
kmeans.fit(norm)     # fit done on x,y pairs

# extract cluster labels
labels = kmeans.labels_

# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]    

# extract x and y values of data points
x = df_clust["CO2"]
y = df_clust["Forest area"]

plt.figure(figsize=(8.0, 8.0))

# plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o", cmap=cm)
    
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    
plt.xlabel("CO2 Emission")
plt.ylabel("Forest Area")
plt.show()

    
    
    
    
    