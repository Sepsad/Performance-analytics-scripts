#K-Means Clustering

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Meetings.csv')
X = dataset
X = X.drop('DUE', axis = 1)
X = X.drop('PRIORITY', axis = 1)
X = X.drop('URL', axis = 1)


#Converting time and date string format to datetime
from datetime import datetime
X['DTSTART'] = X['DTSTART'].map(lambda x: x.rstrip(' IRST').rstrip(' IRDT'))
X['DTSTART'] = X['DTSTART'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %I:%M %p'))
#Specific Date
X = X.loc[X['DTSTART'] > '2016']
X['DTEND'] = X['DTEND'].map(lambda x: x.rstrip(' IRST').rstrip(' IRDT'))
X['DTEND'] = X['DTEND'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %I:%M %p'))
#calculation of the duration of meetings 
X['DUR'] = X['DTEND'] - X['DTSTART']
#Converting TimeDelta to Seconds
X['DUR'] = X['DUR'].apply(lambda x: x.seconds)
#Converting delta time to minutes
X.loc[:,'DUR'] /= 3600
#Data az zamane khas
#X = X[X.DTSTART >= datetime(2018, 1, 1)]
#Srat time
X['START'] = X['DTSTART'].apply(lambda x: x.hour) + (X['DTSTART'].apply(lambda x: x.minute))/60.0
#End time
X['END'] = X['DTEND'].apply(lambda x: x.hour) + (X['DTEND'].apply(lambda x: x.minute))/60.0
#Year
X['YEAR'] = X['DTSTART'].apply(lambda x: x.year)
#Month
X['MONTH'] = X['DTSTART'].apply(lambda x: x.month)


import matplotlib.pyplot as plt

#Plotting the distribution of meetings duration
fig, ax = plt.subplots(dpi = 100)
ax.hist(X['DUR'], 500, edgecolor='k')
plt.yscale('log')
ax.set_xticks(np.arange(0,10,1))
ax.set_xlim(0,10)
ax.axvline(X['DUR'].mean(), color='red', linestyle='dashed', linewidth=1, label='mean')
ax.set_xlabel('Length')
ax.set_ylabel('Dist')
plt.show()

#plotting the distribution of the start of meetings
fig, ax = plt.subplots(dpi = 100)
ax.hist(X['START'], 500, edgecolor='k')
ax.set_xticks(np.arange(8,20,2))
plt.yscale('log')
ax.set_xlim((0,24))
ax.axvline(X['START'].mean(), color='red', linestyle='dashed', linewidth=1, label='mean')
ax.set_xlabel('Start')
ax.set_ylabel('Dist')
plt.show()

#plotting average time on monthly meetings
Monthly_Report = X.groupby(['YEAR','MONTH'], as_index=False)['DUR'].sum() 
fig, ax = plt.subplots(dpi = 300)
dates = []
for i in range(len(Monthly_Report['YEAR'])):
    dates.append(str(Monthly_Report['YEAR'][i])+ ' ' + str(Monthly_Report['MONTH'][i]) 
)
    
plt.plot(dates, Monthly_Report['DUR'].values)
plt.xticks(rotation='vertical',fontsize=5)
plt.yticks(fontsize=5)
plt.grid(True)
ax.set_xlabel('Month')
ax.set_ylabel('Hours')
plt.show()

# Plotting the number of attendee
X['NUM_ATT'] = X['ATTENDEE'].str.count(';') + 1
Num_Att = X['NUM_ATT'].dropna()
fig, ax = plt.subplots(dpi = 100)
ax.hist(Num_Att, np.arange(0.5,16.5,1), edgecolor='k')
ax.set_xticks(range(0,16,1))
#ax.set_xlim(0,15)
ax.axvline(Num_Att.mean(), color='red', linestyle='dashed', linewidth=1, label='mean')
ax.set_xlabel('NUmber of Attendee per Meeting')
ax.set_ylabel('Dist')
plt.show()

#Most participations in meetings
from collections import Counter
fig, ax = plt.subplots(dpi = 300)
Y = X['ATTENDEE'].str.split(pat=';', n=-1, expand=False)
Y = Y.dropna()
Y_F = [subitem for item in Y for subitem in item]
Y_F = [item.strip() for item in Y_F]
Y_F = list(filter(None, Y_F))
Y_F_NUM = Counter(Y_F)
Y_F_NUM_MOST = Y_F_NUM.most_common(60)
labels = [item[0] for item in Y_F_NUM_MOST]
values = [item[1] for item in Y_F_NUM_MOST]
indexes = np.arange(len(labels))
width = 1
plt.bar(indexes, values, align='center', edgecolor = 'black')
plt.xticks(indexes-0.5 + width * 0.5, labels)
plt.xticks(rotation='vertical',fontsize=5)
plt.yticks(fontsize=5)
ax.set_ylabel('Number of Meetings', fontsize = '5')
plt.show()

#preparing the matrix of InteractionS for GEPHI
dim = len(Y_F_NUM)
Interaction = np.zeros(shape=(dim,dim))

Y = Y.reset_index(drop=True)

import itertools

for item in Y:
    b = [subitem.strip() for subitem in item]
    b = list(filter(None, b))
    b_pair = list(itertools.product(b,b))
    for pairs in b_pair:
        n1 = list(Y_F_NUM).index(pairs[0])
        n2 = list(Y_F_NUM).index(pairs[1])
        Interaction[n1][n2] = Interaction[n1][n2] + 1

np.fill_diagonal(Interaction, 0, wrap=False)     

#save as csv 
df = pd.DataFrame(Interaction)
df.columns = list(Y_F_NUM)
df.index = list(Y_F_NUM)
df.to_csv("netwrok.csv")

        
        
        
    
        


 
