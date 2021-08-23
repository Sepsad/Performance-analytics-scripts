#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 11:57:37 2018

@author: alimehdizadeh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('CafeDivar Slack Analytics Sep 02 2018.csv')

dataset['Date'] = dataset['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
dataset['Day'] = dataset['Date'].apply(lambda x: x.weekday())

X = dataset.loc[dataset['Date'] > datetime(2018, 7, 1)]
Y = [0,0,0,0,0,0,0]
for i in range(7):
    Y[i] = X.loc[X['Day'] == i]   

Yf = [0,0,0,0,0,0,0]    
for i in range(7):
    Yf[i] = Y[i]['Messages from Users'].mean()

import matplotlib.pyplot as plt
plt.plot(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'],Yf)
plt.show()


    


