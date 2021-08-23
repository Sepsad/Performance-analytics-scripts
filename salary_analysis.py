#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:28:07 2018

@author: alimehdizadeh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

salary94 = pd.read_csv('salary94.csv')
salary95 = pd.read_csv('salary95.csv')
salary96 = pd.read_csv('salary96.csv')
salary97 = pd.read_csv('salary97.csv')

salary94['FullName'] = salary94['نام '] + salary94['نام خانوادگی']
salary94['FullName'] = salary94['FullName'].str.replace(' ', '')
salary94 = salary94.dropna(axis=0, how='all')
salary94 = salary94.drop(['ردیف','نام خانوادگی','نام ','تمام وقت/پاره وقت','تاریخ استخدام'], axis=1)


salary95['FullName'] = salary95['نام '] + salary95['نام خانوادگی']
salary95['FullName'] = salary95['FullName'].str.replace(' ', '')
salary95 = salary95.dropna(axis=0, how='all')
salary95 = salary95.drop(['نام خانوادگی','نام ','نام و نام خانوادگی','Unnamed: 4','Unnamed: 0'], axis=1)


salary96['FullName'] = salary96['نام '] + salary96['نام خانوادگی']
salary96['FullName'] = salary96['FullName'].str.replace(' ', '')
salary96 = salary96.dropna(axis=0, how='all')
salary96 = salary96.drop(['ردیف','نام خانوادگی','نام ','تاریخ استخدام','نام  نام خانوادگی','کد ملی'], axis=1)

salary97['FullName'] = salary97['نام '] + salary97['نام خانوادگی']
salary97['FullName'] = salary97['FullName'].str.replace(' ', '')
salary97 = salary97.dropna(axis=0, how='all')
salary97 = salary97.drop(['ردیف','نام خانوادگی','نام ','تاریخ استخدام','نام  نام خانوادگی','کد ملی','تیم'], axis=1)

X1 = pd.merge(salary96, salary97, how='outer', on='FullName')
X2 = pd.merge(salary94, salary95, how='outer', on='FullName')

X = pd.merge(X2, X1, how='outer', on='FullName')
X = X.dropna(axis=0, how='all')
X = X.dropna(axis=1, how='all')
X = X.dropna(axis=0, subset=['FullName'])
X = X.set_index('FullName')

#remove nafar who are not hired now 
X = X.dropna(axis = 0, subset=['مرداد97'])

Teams = X['محصول']
X = X.drop(['محصول'], axis = 1)
X = X.replace(0, np.nan)
X.insert(0, 'Team', Teams )

GroupDivar = X.loc[X['Team'] == 'دیوار']    
GroupBazar = X.loc[X['Team'] == 'بازار']  
  
meanD = GroupDivar.mean()
meanB = GroupBazar.mean()
#PLotting 

import matplotlib.pyplot as plt
from bidi import algorithm as bidialg
import random

linestyles = ['-', '--', '-.', ':']

    

fig, (ax0, ax1) = plt.subplots(2, 1, dpi = 300, figsize=(40,20) )
Xax = list(X)
Xax = Xax[1:]

for i in range(len(X)):
    
    
    if X['Team'][i] == 'دیوار':
        ax0.plot(Xax,X.iloc[i][1:], linewidth = 7 - (random.randint(1,len(X))/len(X))*5, alpha = 0.3, LineStyle = random.choice(linestyles) )
        ax0.scatter(Xax,X.iloc[i][1:], s = 10)
        ax0.set_xticklabels(Xax, rotation=90, size = 8)
        ax0.set_title('Divar')
        ax0.grid(True)
        
    if X['Team'][i] == 'بازار':
        ax1.plot(Xax,X.iloc[i][1:], linewidth = 7 - (random.randint(1,len(X))/len(X))*5, alpha = 0.3, LineStyle = random.choice(linestyles) ) 
        ax1.scatter(Xax,X.iloc[i][1:], s = 10)
        ax1.set_xticklabels(Xax, rotation=90, size = 8)
        ax1.set_title('Bazaar')
        ax1.grid(True)

        
ax0.scatter(Xax,meanD, color = 'black')
ax1.scatter(Xax,meanB, color = 'black')        
plt.tight_layout()    
plt.show()






