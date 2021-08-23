
import sys
import os 
import inspect
from os import chdir

chdir('/Users/aliakbarmehdizadeh/Dropbox/CafeBazaar/Performance Evaluation')

#connect to the database
import sqlite3
import pandas as pd
import pdb
#from __future__ import unicode_literals

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

#convert database to pandas dataframe

table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

data_frame = {}   
for index, table_name in table_names.iterrows():
    data_frame[table_name[-1]] = pd.read_sql_query('SELECT * FROM ' + table_name[-1], conn)
    
    
last_round_id = max(data_frame['panel_participant']['round_id'])
last_round_data = data_frame['panel_participant'].loc[ data_frame['panel_participant']['round_id'] == last_round_id ].copy(deep=True)

last_round_data['rate'] = ''
last_round_data['name'] = ''
last_round_data['email'] = ''

for index, person in last_round_data.iterrows():

    try:
        last_round_data.at[index,'rate']  = data_frame['panel_supervisoroverview'].loc[ data_frame['panel_supervisoroverview']['supervisee_id'] == person['id'] ]['performance'].to_list()[0]
    except:
        last_round_data.at[index,'rate'] = 'NOT SPECIFIED'
        #pdb.set_trace()
    
    last_round_data.at[index,'name']  = data_frame['auth_user'].loc[ data_frame['auth_user']['id'] == person['user_id'] ]['first_name'].to_list()[0]
    last_round_data.at[index,'email'] = data_frame['auth_user'].loc[ data_frame['auth_user']['id'] == person['user_id'] ]['email'].to_list()[0]
    
last_round_data = last_round_data.reset_index( drop=True)   

 