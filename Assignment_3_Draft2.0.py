"""
Applied Data Science Assignment 3 by K.Manivannan
The data set from data.worldBank org. Perform cluster and prediction
for CO2 emission based on countries
"""

#Import package for the assignment3
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
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp


path1 = "API_EN.ATM.CO2E.PC_DS2_en_excel_v2_6298783.xlsx"
path2 = "WorldBank_Data_Extract_2010.xlsx"

df = pd.read_excel(path2)
df.dtypes
df.describe()

#Check for missing data
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df)


#Drop row with missing data
df = df.dropna().reset_index(drop=True)

#Check for missing data again after drop row
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 '%_missing_data': percent_missing})
print(missing_value_df)

#Check for correlation in data set features
corr = df.corr(numeric_only=True)
corr = corr.round(3)
print(corr)

#plot the correlation heatmap for visualisation
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True , cmap = 'RdBu', annot_kws={"fontsize":20})

#Scatter plot of data features 
pd.plotting.scatter_matrix(df, figsize=(10,10), s=10)
plt.show()

#Scatter plot of data not highly correlated
plt.scatter(df["Forest area"],df["CO2"], 10, marker="o")
plt.xlabel("Forest area (% of land area)")
plt.ylabel("CO2 emissions (metric tons per capita)")
plt.show()


#setup a scalar object
scaler = pp.RobustScaler()
df_clust = df[["Forest area", "CO2"]]

#extract columns
scaler.fit(df_clust)

#apply the scaling
normalise = scaler.transform(df_clust)
print(normalise)

plt.figure(figsize=(8, 8))

plt.scatter(normalise[:,0], normalise[:,1], 10, marker="o")
plt.xlabel("Forest area (% of land area)")
plt.ylabel("CO2 emissions (metric tons per capita)")
plt.show()

#function for one_silhoutte
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
    score = one_silhoutte(normalise, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")  
# allow for minus signs

# set up the clusterer with the number of expected clusters
cm = matplotlib.colormaps["Paired"]
kmeans = cluster.KMeans(n_clusters=4, n_init=20)

# Fit the data, results are stored in the kmeans object
kmeans.fit(normalise)     # fit done on x,y pairs

# extract cluster labels
labels = kmeans.labels_

# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]    

# extract x and y values of data points
x = df_clust["Forest area"]
y = df_clust["CO2"]

plt.figure(figsize=(8.0, 8.0))

# plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o", cmap=cm)
    
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    
plt.xlabel("Forest area (% of land area)")
plt.ylabel("CO2 emissions (metric tons per capita)")
#plt.legend()
plt.show()

#append cluster label to df dataframe
df["cluster"] = labels
df.head()
df.to_excel("df_Cluster_label.xlsx")

#extract cluster label for cluster legend
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]


plt.scatter(df1["Forest area"], df1['CO2'], color = 'green', label=0)
plt.scatter(df2["Forest area"], df2['CO2'], color = 'red', label=1)
plt.scatter(df3["Forest area"], df3['CO2'], color = 'black', label=2)
plt.scatter(df4["Forest area"], df4['CO2'], color = 'blue', label=3)

# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "cyan", marker="d", label="Centres")

#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], color='purple', marker='*', label='centroid')
plt.xlabel("Forest area (% of land area)")
plt.ylabel("CO2 emissions (metric tons per capita)")
plt.legend()
plt.show()

#Create cluster 2 
plt.scatter(df["Electric_power"],df["Forest area"], 10, marker="o")
plt.xlabel("Electric power consumption (kWh per capita)")
plt.ylabel("Forest area (% of land area)")
plt.show()

#setup a scalar object
scaler = pp.RobustScaler()
df_clust2 = df[["Electric_power", "Forest area"]]

#extract columns
scaler.fit(df_clust2)

#apply the scaling
normalise1 = scaler.transform(df_clust2)
print(normalise1)

plt.figure(figsize=(8, 8))

plt.scatter(normalise1[:,0], normalise1[:,1], 10, marker="o")
plt.xlabel("Electric power consumption (kWh per capita)")
plt.ylabel("Forest area (% of land area)")
plt.show()

# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(normalise1, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")   
# allow for minus si

# set up the clusterer with the number of expected clusters
cm = matplotlib.colormaps["Paired"]
kmeans = cluster.KMeans(n_clusters=5, n_init=20)

# Fit the data, results are stored in the kmeans object
kmeans.fit(normalise1)     # fit done on x,y pairs

# extract cluster labels
labels = kmeans.labels_

# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]    

# extract x and y values of data points
x = df_clust2["Electric_power"]
y = df_clust2["Forest area"]

plt.figure(figsize=(8.0, 8.0))

# plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o", cmap=cm)
    
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    
plt.xlabel("Electric power consumption (kWh per capita)")
plt.ylabel("Forest area (% of land area)")
#plt.legend()
plt.show()

df["cluster2"] = labels
df.head()
df.to_excel("df_Cluster2_label.xlsx")

#Create cluster with label
df1 = df[df.cluster2==0]
df2 = df[df.cluster2==1]
df3 = df[df.cluster2==2]
df4 = df[df.cluster2==3]
df5 = df[df.cluster2==4]

plt.scatter(df1["Electric_power"], df1['Forest area'], color = 'green', label=0)
plt.scatter(df2["Electric_power"], df2['Forest area'], color = 'red', label=1)
plt.scatter(df3["Electric_power"], df3['Forest area'], color = 'black', label=2)
plt.scatter(df4["Electric_power"], df4['Forest area'], color = 'blue', label=3)
plt.scatter(df5["Electric_power"], df5['Forest area'], color = 'pink', label=4)

# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "cyan", marker="d", label="Centres")

#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], color='purple', marker='*', label='centroid')
plt.xlabel("Electric power consumption (kWh per capita)")
plt.ylabel("Forest area (% of land area)")
plt.legend()
plt.show()
