# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('parking.csv')

#Deleting rows with empty name
from pandas import DataFrame
dataset_clean = dataset.dropna() 

X = dataset_clean



#Time and Date in a column
X['DateTime'] = X['Date'] + ' ' + X['Time']

#Converting time and date string format to datetime
from jdatetime import datetime
X['DateTime'] = X['DateTime'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))

#Minimun Entry time for each Name, Date
minEntry = X.loc[X['Type'] == 'En'].rename(index=str, columns={'DateTime':'En'}).groupby(['Name', 'Date'])['En'].min().to_frame()    
#Maximum Exit time for each Name, Date
maxExit = X.loc[X['Type'] == 'Ex'].rename(index=str, columns={'DateTime':'Ex'}).groupby(['Name', 'Date'])['Ex'].max().to_frame()    
#Merging min Entry time and max Exit time for each Name, Date
df = minEntry.merge(maxExit, how='inner', right_index=True, left_index=True)
#Calculating time diffrence in TimeDelta format
df['dur'] = df['Ex'] - df['En']
#Converting TimeDelta to Seconds
df['dur'] = df['dur'].apply(lambda x: x.seconds)
#Converting delta time to hours
df.loc[:,'dur'] /= 3600
Sum = df['dur'].sum()
#deleting inapporpriate numbers
df = df[ (df.dur <= 17)]
df = df[ (df.dur >= 2)]
#Entrance time
df['En_hour'] = df['En'].apply(lambda x: x.hour) + (df['En'].apply(lambda x: x.minute))/60.0
#Exite time
df['Ex_hour'] = df['Ex'].apply(lambda x: x.hour) + (df['Ex'].apply(lambda x: x.minute))/60.0


import matplotlib.pyplot as plt
import seaborn as sns

#Plotting distribution of stay at work
fig, ax = plt.subplots(dpi = 100)
ax.hist(df['dur'], 20, density=True, edgecolor='k')
ax.set_xticks(range(1,15,1))
#ax.set_xlim((1, 15))
ax.axvline(df['dur'].mean(), color='red', linestyle='dashed', linewidth=1, label='mean')
ax.set_xlabel('Stay at Work(hr)')
ax.set_ylabel('Distribution')
#sns.distplot(df['dur'], hist = False, kde = True,
#             kde_kws = {'linewidth': 2})   
plt.legend()
fig.canvas.draw()
plt.show()

#plotting distribution of entrance
fig, ax = plt.subplots(dpi = 100)
ax.hist(df['En_hour'], 24, density=True, edgecolor='k')
ax.set_xticks(range(0,24,2))
#ax.set_xlim((6,14))
ax.axvline(df['En_hour'].mean(), color='red', linestyle='dashed', linewidth=1, label='mean')
ax.set_xlabel('Entrance(hr)')
ax.set_ylabel('Distribution')
#sns.distplot(df['En_hour'], hist = False, kde = True,
#             kde_kws = {'linewidth': 2})  
plt.legend()
fig.canvas.draw()
plt.show()

#plotting distribution of exite
fig, ax = plt.subplots(dpi = 100)
ax.hist(df['Ex_hour'], 24, density=True, edgecolor='k')
ax.set_xticks(range(0,24,2))
#ax.set_xlim((14,24))
ax.axvline(df['Ex_hour'].mean(), color='red', linestyle='dashed', linewidth=1, label = 'mean')
ax.set_xlabel('Exit(hr)')
ax.set_ylabel('Distribution')
#sns.distplot(df['Ex_hour'], hist = False, kde = True,
#             kde_kws = {'linewidth': 2})  
plt.legend()
fig.canvas.draw()
plt.show()



# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
Values = df.iloc[:,[3,2]].values
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 3)
    kmeans.fit(Values)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


 # Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 3)
y_kmeans = kmeans.fit_predict(Values)

# Visualising the clusters
fig, ax = plt.subplots(dpi = 100)
plt.scatter(Values[y_kmeans == 0, 0], Values[y_kmeans == 0, 1], s = 1, c = 'red', label = 'cluster 0')
plt.scatter(Values[y_kmeans == 1, 0], Values[y_kmeans == 1, 1], s = 1, c = 'blue', label = 'cluster 1')
plt.scatter(Values[y_kmeans == 2, 0], Values[y_kmeans == 2, 1], s = 1, c = 'green', label = 'cluster 2')
#plt.scatter(Values[y_kmeans == 3, 0], Values[y_kmeans == 3, 1], s = 1, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[0:3, 0], kmeans.cluster_centers_[0:3, 1], s = 30, c = 'yellow')
plt.xlabel('Entrance(hr)')
plt.ylabel('Stay(hr)')
plt.legend()
plt.show()


#average cluster allocation 
df['cluster'] = y_kmeans
People = df.groupby(['Name'])['cluster'].mean().to_frame()
People['count'] = df.groupby(['Name'])['cluster'].count()


 










