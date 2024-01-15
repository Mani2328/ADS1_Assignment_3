"""
Applied Data Science Assignment 3 by K.Manivannan
The data set from data.worldBank org. Perform cluster and prediction
for CO2 emission based on countries
"""

#Import package for the assignment3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import errors as err
import scipy.optimize as opt
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
plt.xlabel("Electric power consumption (kWh per capita)")
plt.ylabel("Forest area (% of land area)")
plt.legend()
plt.show()

#The rest of the Code will prediction for CO2 
def Year_Country(path1):
    """
    This function take the file name as argument 
    and produce 2 data frame, one with years as columns and 
    transpose the first dataframe and produce another dataframe
    with countries as columns
    
    Parameters
    arg_1 : dataframe #from world bank format
    arg_2 : int
    
    Returns:
    Dataframe1 : with year as column
    Dataframe2 : with country as column
    """
    df = pd.read_excel(path1, header=3)
    df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1,
            inplace = True)
    df_country = df
    df_year = df.transpose()
    df_year.reset_index(drop=False, inplace=True)
    new_header = df_year.iloc[0]
    df_year.drop([0], axis = 0, inplace=True)
    df_year.columns = new_header
    df_year = df_year.rename(columns = {'Country Name':'Year'})
    return df_year, df_country
    
df_year, df_country = Year_Country(path1)
print(df_year)
df_year = df_year[30:57]
df_year.reset_index(drop=True, inplace = True)
# drops columns that have less than 3 non-NaNs 
df_year = df_year.dropna(axis='columns', thresh=3) 
df_year.dtypes  #Check the data types

#Change data type from object to numeric
df_year = df_year.apply(pd.to_numeric)
print(df_year.dtypes)

def exp_decay(t, scale, decay):
    """ Function for fitting exponential decay
        t: independent variable
        scale and decay are the parameters to be fitted
    """
    t0 = 1990
    f = scale * np.exp(decay * (t - t0))
    return f

#Inital Curve fit for CO2 emission by Germany
popt, pcov = opt.curve_fit(exp_decay, df_year['Year'], df_year['Germany'])
print('Fit parameters', popt)

df_year['CO2_Germany_fit'] = exp_decay(df_year['Year'], *popt)
df_year

plt.figure()
plt.plot(df_year['Year'], df_year['Germany'], label = 'data')
plt.plot(df_year['Year'], df_year['CO2_Germany_fit'], label = 'fit')
plt.ylim(6,12)
plt.title('CO2 Emission - Germary')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.legend()
plt.show()

#Second attempt with inital parameters
popt, pcov = opt.curve_fit(exp_decay, df_year['Year'], df_year['Germany'],
                           p0=[11.42, -0.00962])
print("Fit parameter", popt)

df_year['C02_Germany_fit2'] = exp_decay(df_year['Year'], *popt) #Predict Y values

#Predict CO2 Emission for Germary in year 2030 and 2040
print("Germany CO2 emission in 2030: {:0.3f}".format(exp_decay(2030, *popt)))
print("Germany CO2 emission in 2030: {:0.3f}".format(exp_decay(2040, *popt)))

years = np.linspace(1990, 2030)
pop_decay = exp_decay(years, *popt)

sigma = err.error_prop(years, exp_decay, popt, pcov)
low = pop_decay - sigma
high = pop_decay + sigma

plt.figure()
plt.plot(df_year['Year'], df_year['Germany'], label='data')
plt.plot(years, pop_decay, label ='fit')
plt.fill_between(years, low, high, alpha=0.5, color='y')
plt.legend()
plt.title('CO2 Emission - Germary')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.show()

#CO2 Emission by Singapore
popt, pcov = opt.curve_fit(exp_decay, df_year['Year'], df_year['Singapore'])
print('Fit parameters', popt)
df_year['CO2_Singapore_fit'] = exp_decay(df_year['Year'], *popt)

#Plot CO2 Emission actual data and Fit data
plt.figure()
plt.plot(df_year['Year'], df_year['Singapore'], label = 'data')
plt.plot(df_year['Year'], df_year['CO2_Singapore_fit'], label = 'fit')
plt.ylim(6,12)
plt.title('CO2 Emission - Singapore')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.legend()
plt.show()

#Second attempt with initial parameters
popt, pcov = opt.curve_fit(exp_decay, df_year['Year'], df_year['Singapore'],
                           p0=[10.67, -0.0114])
print("Fit parameter", popt)
df_year['CO2_Singapore_fit2'] = exp_decay(df_year['Year'], *popt) 

plt.figure()
plt.plot(df_year['Year'], df_year['Singapore'], label = 'data')
plt.plot(df_year['Year'], df_year['CO2_Singapore_fit2'], label = 'fit')
plt.title('CO2 Emission - Singapore')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.ylim(6,12)
plt.legend()
plt.show()

#Predict CO2 emission for SIngapore in 2030 and 2040
print("Singapore CO2 emission in 2030: {:0.3f}".format(exp_decay(2030, *popt)))
print("Singapore CO2 emission in 2040: {:0.3f}".format(exp_decay(2040, *popt)))

years = np.linspace(1990, 2030)
pop_decay = exp_decay(years, *popt)

sigma = err.error_prop(years, exp_decay, popt, pcov)
low = pop_decay - sigma
high = pop_decay + sigma

plt.figure()
plt.plot(df_year['Year'], df_year['Singapore'], label='data')
plt.plot(years, pop_decay, label ='fit')
plt.fill_between(years, low, high, alpha=0.5, color='y')
plt.title('CO2 Emission - Singapore')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.legend()
plt.show()


def logistics(t, s, k, t0):
    """
    Function for fitting logistics function   
    Parameters
    ----------
    t : time - year
        independent variable
    s : numerical
        scale factor.
    k : float
        is the logistic growth rate.
    t0 : time - year
        is the sigmoid's midpoint.

    Returns
    -------
    f : float
        Predict value with logistics function.

    """
        
    f = s / (1 + np.exp(-k * ( t - t0)))
    return f

plt.plot(df_year.Year, df_year['Korea, Rep.'])
popt, pcov = opt.curve_fit(logistics, df_year['Year'], df_year['Korea, Rep.'],
                           p0=(10, 0.5, 2000))
print("Fit parameter", popt)

popt, pcov = opt.curve_fit(logistics, df_year['Year'], df_year['Korea, Rep.'],
                           p0=(13.25, 0.089, 1990))
print("Fit parameter", popt)

df_year['Korea_Logistic'] = logistics(df_year['Year'], *popt)

plt.figure()
plt.plot(df_year['Year'], df_year['Korea, Rep.'], label = 'data')
plt.plot(df_year['Year'], df_year['Korea_Logistic'], label = 'fit')
plt.title("CO2 Emission - Korea")
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.legend()
plt.show()

print("Korea CO2 emission in 2030: {:0.3f}".format(logistics(2030, *popt)))
print("KOrea CO2 emission in 2040: {:0.3f}".format(logistics(2040, *popt)))

years = np.linspace(1990, 2030)
pop_logistics = logistics(years, *popt)

sigma = err.error_prop(years, logistics, popt, pcov)
low = pop_logistics - sigma
high = pop_logistics + sigma

plt.figure()
plt.plot(df_year['Year'], df_year['Korea, Rep.'], label='data')
plt.plot(years, pop_logistics, label ='fit')
plt.fill_between(years, low, high, alpha=0.5, color='y')
plt.title("CO2 Emission - Korea")
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.legend()
plt.show()
