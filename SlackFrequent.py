#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:19:34 2018

@author: aliakbarmehdizadeh
"""

import json
import pandas as pd
import glob

with open('2018-08-19.json') as f:
    data = json.load(f)
    
X = pd.DataFrame.from_dict(data)    

for filename in glob.glob('*.json'):
    with open(filename) as f:
    data = json.load(f)
    X = pd.DataFrame.from_dict(data)    
