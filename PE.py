#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:44:51 2019

@author: aliakbarmehdizadeh
"""

###
PMs = ['armank@CB', 'arian@D','alireza@D','ali.hanzakian@CB','hirbod@CB','amir.bornaee@CB',
       'amin@D','mirzazadeh@CB','mhjavadi@D','abouzar@D','mojdeso@CB','ali.vahdani@D','mahdis@CB',
       'armita@CB','alireza@CB']

VPs = ['pooria@CB','vahid@CB','danial@CB','keyhan@CB','behnam@D']

Tech_Bazaar = ['vahid@CB','sepehr103@CB','alirezamika@CB','and@CB','alirk@CB','ghazal@CB','pouya@CB','pdrm.taheri@CB','mandomi@CB','shayanpourvatan@CB','hamidrezasahraei@CB','saeed.masoumi@CB',
'imn@CB','ali.rajabi@CB','m.teimori@CB','morteza@CB','mnoroozi@CB','ali.amelie@CB','asalehe@CB','boomari@CB','mzarrintareh@CB','mojtaba.k@CB',
'pedramteymoori@CB','parsa.azm@CB','sahel@CB','h.hosseinvand@CB','amitis.shidani@CB','arman@CB','ramtinrostami@CB','arefmq@CB','p.abdollahi@CB','h.r.ahmadian@CB','mohammad.beygi@CB','melika@CB','moein@CB','khaari@CB','mohammadmahdi@CB','arash.shakery@CB','aliasadi@CB','mh.shabani@CB',
'e.poursaeed@CB','bardia.heydari@CB ']

Zirsakht = ['hassanz@D','seraji@CB','arya@CB','abz@CB','reith@CB','eqbal@CB','navidn@D','nikan@CB','kiana@CB','meysam.s@CB',
            'geram@CB','danialgood@CB','navid@CB','ali.orouji@CB','m.sheikh@CB','mesbahi@CB','speed@CB','razavi.naghmeh@CB','soheild@CB',
            'mirfenderesgi@CB','amirreza@CB','navidhtb@CB','Karbas@CB','artin@CB','navid@CB','sina.mirhejazi@CB','mojtaba.k@CB',
            'armin.shoushtary@CB','mehdy@CB','behnam@D','emad.mohamadi@CB','AliSoumee@CB','mohammadreza@CB',]

Divar    = ['mahdim@CB','ali.rajabi@CB','rezas@D','hoshay@CB','mohtada.h@CB','alich@CB','rezashiri@CB','mhkeshavarz@CB',
            'parsa@CB','borna@CB','behmand@D','alimz@D','hesam.r@CB','mahdinouri@CB','agah@CB','m.kharatizadeh@CB','ghasem.b@CB',
            's.hamzelooy@CB','h.fadaei@CB','hashemi.soroush@CB','agorji@CB','Kayvan@CB','sinamokhtarzadeh@CB','sadegh.mahdavi@CB',
            'msoltanian@CB','afkar@CB','karo@CB','miladn@CB','ahrzb@CB','jbalalimoghadam@CB','aghosey@CB','mory@CB','miladrezaei@CB',
            'ali.ahmadi@CB','khoshraftar@CB','alireza.8n@CB','afra.amini@D','m.reza@CB','madmadi@CB','norouzi@CB','yasamin.rahimi@CB','payam@CB']

Balad    = ['peyman.jabarzade@CB','hosein.moosavi@CB','razizadeh@CB','mbehzadi@CB','armank@CB','mahan@CB','ramin@CB','Hastifz@CB','ikhoshabi@CB',
            'parham@CB','ali.irani@CB','siavash.abdoli@CB','moh.mahdieh@CB','atofighi@CB','kimia@CB','ho3in@CB','sharifihaghighi@CB','ahlajevardi@CB',
            'ali.javadi@CB','ssajad@CB','naser@B','mohammad.razeghi@B','amir.dehghan@CB','shadihn@CB','arian@CB','alie@CB','iman.gholami@CB',
            'arastu@CB','bagher@CB','erfan@B','payam@CB','alibahmani@CB','mahsa.rahimi@CB','derakhshandeh@CB','hasti.ranjkesh@CB','s.dehghanian@CB',
            'hejazi@CB','keyhan@CB/asghari@CB']

BlackList = ['armank@CB', 'arian@D','alireza@D','ali.hanzakian@CB','hirbod@CB','amir.bornaee@CB',
      'amin@D','mirzazadeh@CB','mhjavadi@D','abouzar@D','mojdeso@CB','ali.vahdani@D','mahdis@CB',
      'armita@CB', 'yasamin.dashti@CB','arian@CB','amin@CB','sadjad@CB','alibahmani@CB','hejazi@CB','hastifz@CB',
      'hedie@CB','s.dehghanian@CB''iman@CB','ahlajevardi@CB','kaveh@CB','majid@CB','pooria@CB','amir@CB','armita@CB',
      'ashkan@D','abolfazl@CB','amehdizadeh@CB','hamidreza@CB','milad@CB','vahid@CB','behnam@D','danial@CB','keyhan@CB',
      'masoud@D','ali.alaee@CB','amin@D','rozhina@CB','mirzazadeh@CB','abbasmousavi@CB','tahere@CB','moallemi@CB']

LD = ['amehdizadeh@CB','a.nobakhti@CB','y.baghaei@CB','abolfazl@CB','amirj@CB'] 

Marketing_Bazaar=['amin@CB','haghighat@CB','reihanneh@CB','jamalahmadi@CB','shervinrrad@CB',
                 'habibolah@CB','bahram@CB','mina.cheragh@CB','farshad.mokhtari@CB','mahdi@CB',
                 'miaad@CB','alireza.ahmadi@CB','mohammad@CB','ehsan.z@CB','e.kangarani@CB',
                 'laleh@CB','sassan@CB','mreza@CB','sanaz@CB','niloofar@CB','faride@CB',
                 'mahta@CB','amir.nasrol@CB','roomana@CB','saman@CB','golnar@CB','niusha@CB',
                 'etezadi@CB','s.hasanzade@CB']

#################################################################
import sys
import os 
import inspect
from os import chdir

chdir('/Users/aliakbarmehdizadeh/Dropbox/CafeBazaar/Performance Evaluation')

#connect to the database
import sqlite3
import pandas as pd
#from __future__ import unicode_literals

conn = sqlite3.connect('db.sqlite3')

c = conn.cursor()

#convert database to pandas dataframe

table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

data_frame = {}   
for index, table_name in table_names.iterrows():
    data_frame[table_name[-1]] = pd.read_sql_query('SELECT * FROM ' + table_name[-1], conn)
    
#################################################################
#Start_Review    
#collaborations network from peer-reviews
import csv
import numpy as np
from pandas import isnull

collaborations_netwrok = pd.DataFrame(None, index=data_frame['panel_participant']['id'], columns=data_frame['panel_participant']['id'])
collaborations_netwrok = collaborations_netwrok.applymap(lambda x: {'weight': 0 ,'rate': [] } if isnull(x) else x)

for index, task in data_frame['panel_selfreview_peers'].iterrows():
    self_review_id = task['selfreview_id']
    self_review_author_id = data_frame['panel_selfreview'].loc[ data_frame['panel_selfreview']['id'] == self_review_id ]['participant_id'].tolist()[0]
    self_review_peer_id = task['participant_id']
    rate = data_frame['panel_peerreview'].loc[ (data_frame['panel_peerreview']['self_review_id'] == self_review_id) & (data_frame['panel_peerreview']['peer_id'] == self_review_peer_id) ]['success'].values
    
    collaborations_netwrok[self_review_author_id][self_review_peer_id]['weight'] += 1
    collaborations_netwrok[self_review_peer_id][self_review_author_id]['weight'] += 1
    
    collaborations_netwrok[self_review_peer_id][self_review_author_id]['rate'].append(list(rate))
    collaborations_netwrok[self_review_author_id][self_review_peer_id]['rate'].append(list(rate))

        
#divide netwroks into evaluation rounds
participants_by_round = data_frame['panel_participant'].groupby('round_id')

collaborations_netwrok_per_round = {}

for name,group in participants_by_round:
    participants_id_per_round = []
    for participant in group.iterrows():
        participants_id_per_round.append(participant[1][0]) 
        
    collaborations_netwrok_per_round[str(name)] = collaborations_netwrok[participants_id_per_round]
    collaborations_netwrok_per_round[str(name)] = collaborations_netwrok_per_round[str(name)][collaborations_netwrok_per_round[str(name)].index.isin(participants_id_per_round)]
      
    #renaming index and lables
    for participant in participants_id_per_round:
        participant_user_id = data_frame['panel_participant'].loc[data_frame['panel_participant']['id'] == participant ]['user_id'].tolist()[0]
        participant_name    = data_frame['auth_user'].loc[data_frame['auth_user']['id'] == participant_user_id ]['username'].tolist()[0]
        collaborations_netwrok_per_round[str(name)].rename(columns={participant: participant_name }, inplace=True)
        collaborations_netwrok_per_round[str(name)].rename({participant: participant_name}, inplace=True, axis='index')
            
    #saving the outputs
    #collaborations_netwrok_per_round[str(name)].to_csv("collaboration_netwrok_round"+str(name)+".csv")

#################################################################
#plotting the collabortion network
import networkx as nx
from networkx.readwrite import json_graph 
import json
from matplotlib import pyplot as plt
import webcolors
from pyvis.network import Network
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, WheelZoomTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes, NodesOnly
from bokeh.palettes import Spectral4
from bokeh.models import ColumnDataSource
from bokeh.models import LabelSet

#edfd5e
ratings = {
           "NEEDS_IMPROVEMENT" : '#ff0000', 
           "AS_EXPECTED" : '#bfff00', 
           "MORE_THAN_EXPECTED" : '#00ff00', 
           "MUCH_MORE_THAN_EXPECTED" : '#00ffff'}
    
peer_ratings = {
           "LOW" : '#ff0000', 
           "NORMAL" : '#edfd5e', 
           "GOOD" : '#bfff00', 
           "VERY_GOOD": '#00ff00',
           "EXCELLENT" : '#00ffff'}
           
           
for key, value in collaborations_netwrok_per_round.items():
    #second round does not have a network
    if key == '1': continue
    if key == '2': continue
    #if key == '3': continue
    
    network = value.applymap(lambda x: x['weight'])

    G = nx.from_numpy_matrix(network.values)

    #node coloring and labeling - base on supervisor rating

    node_labels = {}
    node_color  = {}
    
    for idx, node in enumerate(G.nodes()):
        #[:-3] removes '.ir' at the end of names
        node_labels[node]    = list(value)[node][:-3].replace('cafebazaar','CB').replace('divar','D')
        participant_user_id  = data_frame['auth_user'][data_frame['auth_user']['username'] == list(value)[node] ]['id'].tolist()[0] 
        participant_id       = data_frame['panel_participant'].loc[ (data_frame['panel_participant']['user_id'] == participant_user_id) & (data_frame['panel_participant']['round_id'] == int(key))]['id'].tolist()[0]
        #if not rated leave it gray
        try: 
            node_color[node] = ratings[data_frame['panel_supervisoroverview'].loc[ data_frame['panel_supervisoroverview']['supervisee_id'] == participant_id]['performance'].tolist()[0]]
            if node_labels[node] in BlackList:
                node_color[node] = '#cccccc'
                
        except Exception as inst:
            #print(inst)
            
            node_color[node] = '#cccccc'
            
    #node_size = [np.power(degree,2.5) for degree in G.degree().values()] 
    node_size = [int(np.power(degree,1)) for node, degree in G.degree()]
    node_size = dict(zip(node_labels.keys(), node_size))
    
    #set nodes attribiutes
    
    nx.set_node_attributes(G, node_labels , 'labels')
    nx.set_node_attributes(G, node_size   , 'sizes' )
    nx.set_node_attributes(G, node_color  , 'colors')

    edge_color_list = []
    
#   Edge coloring - base on peer rating   
    for u,v in G.edges():
        node1_label   = list(value)[u]
        node2_label   = list(value)[v]
        try: 
            edge_ratings     = value[node1_label][node2_label]['rate']
            edge_ratings[:]  = [item for item in edge_ratings if item != []]
            if len (edge_ratings) == 0 : raise ValueError('A very specific bad thing happened.')
            edge_ratings_hex = [peer_ratings[rate[0]] for rate in edge_ratings] 
            edge_ratings_rgb = [ webcolors.hex_to_rgb(hex_color) for hex_color in edge_ratings_hex] 
            edge_color_rgb = {'R':0,'G':0,'B':0}
            for rating in edge_ratings_rgb:
                edge_color_rgb['R'] += rating[0]
                edge_color_rgb['G'] += rating[1]
                edge_color_rgb['B'] += rating[2]
            for key1 in edge_color_rgb:
                edge_color_rgb[key1] = int (edge_color_rgb[key1] / len (edge_ratings))
            edge_color_hex = webcolors.rgb_to_hex((edge_color_rgb['R'],edge_color_rgb['G'],edge_color_rgb['B'])) 
            #G.add_edge(u,v, color = edge_color_hex)
            edge_color_list.append(edge_color_hex)
        except Exception as inst:
            #print(inst)
            #G.add_edge(u,v, color = '#ffffff')
            edge_color_list.append('#ffffff')
            #edge_color_list.append('#000000')
            
    edge_color_list = dict(zip(list(G.edges()), edge_color_list))            
    nx.set_edge_attributes(G, edge_color_list, 'colors')        
        
    #populate overload nodes on a circle with radious r
#    number_of_fixed_nodes = 1
#    over_load_nodes = sorted(range(len(node_size)), key=lambda i: node_size[i])[-number_of_fixed_nodes:] 
#    fixed_positions = {}
#    teta = (2*np.pi)/number_of_fixed_nodes
#    r    =  5.0
#    i = 0
#    for item in over_load_nodes:
#        x = r * np.cos(i*teta)
#        y = r * np.sin(i*teta)
#        i = i + 1     
#        fixed_positions[item] = (x,y)    
#    fixed_nodes = fixed_positions.keys()
    
    #removing/filtering free nodes and edges for simplification
    number_of_free_nodes = 0
    free_nodes = sorted(range(len(node_size)), key=lambda i: node_size[i])[:number_of_free_nodes]
    
    #remove particular nodes
    for free_node in free_nodes:
        G.remove_node(free_node)
    
#    #LD overview:
#    if key == '3':
#        
#        LD_nodes = []
#        for node in G.nodes(data=True): 
#            if node[1]['labels'] in LD: 
#                LD_nodes.append(node[0]) 
#                
#        edges = list (G.edges(data = True))     
#        
#        for edge in edges:
#            if  not (( edge[0] in LD_nodes ) | ( int(edge[1]) in LD_nodes )):
#                G.remove_edge(edge[0],edge[1])      
        
    ##Marketing Bazaar
        
    if key == '3':
        
        Marketing_Bazaar_nodes = []
        for node in G.nodes(data=True): 
            if node[1]['labels'] in Marketing_Bazaar: 
                Marketing_Bazaar_nodes.append(node[0]) 
                
        edges = list (G.edges(data = True))     
        
        for edge in edges:
            if  not (( edge[0] in Marketing_Bazaar_nodes ) | ( int(edge[1]) in Marketing_Bazaar_nodes )):
                G.remove_edge(edge[0],edge[1])      

    
    #PMs overview:
#    if key == '3':
#        
#        PM_nodes = []
#        for node in G.nodes(data=True): 
#            if node[1]['labels'] in PMs: 
#                PM_nodes.append(node[0]) 
#                
#        edges = list (G.edges(data = True))     
#        
#        for edge in edges:
#            if  not (( edge[0] in PM_nodes ) | ( int(edge[1]) in PM_nodes )):
#                G.remove_edge(edge[0],edge[1])      

 #   VPs overview:
#    if key == '3':
#        
#        VP_nodes = []
#        for node in G.nodes(data=True): 
#            if node[1]['labels'] in VPs: 
#                VP_nodes.append(node[0]) 
#                
#        edges = list (G.edges(data = True))     
#        
#        for edge in edges:
#            if  not (( edge[0] in VP_nodes ) | ( int(edge[1]) in VP_nodes )):
#                G.remove_edge(edge[0],edge[1])     
    
    #Tech_Bazaar:
#    if key == '3':
#        
#        Tech_Bazaar_nodes = []
#        for node in G.nodes(data=True): 
#            if node[1]['labels'] in Tech_Bazaar: 
#                Tech_Bazaar_nodes.append(node[0])
#            else:
#                G.nodes[node[0]]['colors'] = '#cccccc'
#                
#        edges = list (G.edges(data = True))     
#        
#        for edge in edges:
#            if  not (( edge[0] in Tech_Bazaar_nodes ) | ( int(edge[1]) in Tech_Bazaar_nodes )):
#                G.remove_edge(edge[0],edge[1]) 
#  
        
 ##Divar 
#    if key == '3':
#        
#        Divar_nodes = []
#        for node in G.nodes(data=True): 
#            if node[1]['labels'] in Divar: 
#                Divar_nodes.append(node[0])
#            else:
#                G.nodes[node[0]]['colors'] = '#cccccc'
#                
#        edges = list (G.edges(data = True))     
#        
#        for edge in edges:
#            if  not (( edge[0] in Divar_nodes ) | ( int(edge[1]) in Divar_nodes )):
#                G.remove_edge(edge[0],edge[1]) 
#          
     
 ##Zirsakht
#    if key == '3':
#        
#        Zirsakht_nodes = []
#        for node in G.nodes(data=True): 
#            if node[1]['labels'] in Zirsakht: 
#                Zirsakht_nodes.append(node[0])
#            else:
#                G.nodes[node[0]]['colors'] = '#cccccc'
#                
#        edges = list (G.edges(data = True))     
#        
#        for edge in edges:
#            if  not (( edge[0] in Zirsakht_nodes ) | ( int(edge[1]) in Zirsakht_nodes )):
#                G.remove_edge(edge[0],edge[1]) 

 ##Balad                    
#    if key == '3':
#        
#        Balad_nodes = []
#        for node in G.nodes(data=True): 
#            if node[1]['labels'] in Balad: 
#                Balad_nodes.append(node[0])
#            else:
#                G.nodes[node[0]]['colors'] = '#cccccc'
#                
#        edges = list (G.edges(data = True))     
#        
#        for edge in edges:
#            if  not (( edge[0] in Balad_nodes ) | ( int(edge[1]) in Balad_nodes )):
#                G.remove_edge(edge[0],edge[1]) 
                
    #remove particular edges
    edges = list (G.edges(data = True))
    edges = list ( filter(lambda a: a[2]['weight'] <= 1, edges) )

    for edge in edges:
            G.remove_edge(edge[0],edge[1])
#            
    isolates = nx.isolates(G)
    for node in list(isolates):
                G.remove_node(node)
    #plotting
    
    #  kamada_kawai_layout          
#    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
#    for row, data in nx.shortest_path_length(G):
#        for col, dist in data.items():
#            df.loc[row,col] = dist
#    
#    df = df.fillna(df.max().max())
    
    #pos = nx.kamada_kawai_layout(G, dist=df.to_dict(), weight='weight', scale = 1 )
    #spectral
    #pos = nx.spectral_layout(G, pos = df.to_dict(), weight='weight', scale=1, center=None, dim=2)

    #spring layout
    #pos = nx.spring_layout(G, k = 300/np.sqrt(G.order()), weight ='weight', iterations = 2500, scale=1, pos = fixed_positions, fixed = fixed_nodes )
    pos = nx.spring_layout(G, k = 200/np.sqrt(G.order()), weight ='weight', iterations = 3000, scale=1)


    fig = plt.figure(300,figsize=(120,120), facecolor='black') 

    nx.draw_networkx_nodes (G, pos, node_size  = list(nx.get_node_attributes(G,'sizes' ).values()), 
                                    node_color = list(nx.get_node_attributes(G,'colors').values()), 
                                    alpha = 0.7)
    nx.draw_networkx_edges (G, pos, width     = 1, edge_color = list(nx.get_edge_attributes(G,'colors').values()), alpha = 1)
    nx.draw_networkx_labels(G, pos, labels    = nx.get_node_attributes(G,'labels') ,font_size = 20, font_color= 'white')

    x = nx.json_graph.node_link_data(G)  
    
    for index in range(len(x['links'])):
        x['links'][index]['target'] = int(x['links'][index]['target'])
    with open('collaboration_network_round'+key+'.json','w') as outfile:    
        outfile.write(json.dumps(x,indent=4))

    
    plt.axis('off')
    plt.savefig('collaboration_network_round'+key+'.png', facecolor=fig.get_facecolor(), transparent=True) # save as png
    plt.close()
    
    source = ColumnDataSource(pd.DataFrame.from_dict({k:v for k,v in G.nodes(data=True)}, orient='index'))
    pwidth  = int (2560 *1.5)
    pheight = int (1600 *1.5)
    plot = Plot(plot_width=pwidth , plot_height=1600,
                x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
    
    plot.background_fill_color = "black"
    plot.background_fill_alpha = 1
    
    plot.title.text = "Collaboration Network"
    
    plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())
    
    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0,0))
    
    graph_renderer.node_renderer.data_source = source
    graph_renderer.node_renderer.glyph = Circle(fill_color = 'colors' ,size = 'sizes', line_color = 'colors', fill_alpha = 0.4)
    
    graph_renderer.node_renderer.data_source.data['name'] = [item[1]['labels'] for item in list(G.nodes(data = True))]
    
    graph_renderer.edge_renderer.data_source.data["weight"] = [G.get_edge_data(a,b)['weight'] for a, b in G.edges()]
    graph_renderer.edge_renderer.data_source.data["color"]  = [G.get_edge_data(a,b)['colors'] for a, b in G.edges()]
    
    graph_renderer.edge_renderer.glyph = MultiLine(line_width = 'weight', line_color = 'color', line_alpha = 0.4 )
    
    graph_renderer.node_renderer.selection_glyph = Circle(fill_color = 'colors' ,size = 'sizes', line_color = 'colors', fill_alpha = 1 )
    graph_renderer.node_renderer.hover_glyph = Circle(fill_color = 'colors' ,size = 'sizes', line_color = 'colors', fill_alpha = 1)
    
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_width = 'weight', line_color = 'color', line_alpha = 1 )
    graph_renderer.edge_renderer.hover_glyph     = MultiLine(line_width = 'weight', line_color = 'color', line_alpha = 1 )
    
    hover = HoverTool(tooltips=[("", "@name")])
    plot.add_tools(hover, TapTool(), BoxSelectTool(), WheelZoomTool())
    
    graph_renderer.inspection_policy = NodesOnly()
    graph_renderer.selection_policy  = NodesAndLinkedEdges()
    
    plot.renderers.append(graph_renderer)
    
    x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
    node_labels = nx.get_node_attributes(G, 'labels')
    
    source = ColumnDataSource({'x': x, 'y': y,
                               'labels': [node_labels[node] for node in list(G.nodes())]})
        
    labels = LabelSet(x='x', y='y', text='labels', source=source,
                      background_fill_color='black', text_color = 'white', background_fill_alpha = 0.01, text_font_size = '8pt' )
    
    plot.renderers.append(labels)
    
    output_file('collaboration_network_round'+key+'.html')
    show(plot)


#################################################################
# Rating Habit Classification
#################################################################
    
peer_ratings_habit = {
           "LOW"        : 0, 
           "NORMAL"     : 1, 
           "GOOD"       : 2, 
           "VERY_GOOD"  : 3,
           "EXCELLENT"  : 4}
           
unique_tasks = data_frame['panel_peerreview'].groupby(['self_review_id'])
participant_info = data_frame['auth_user'].copy(deep=True)

rating_habit = pd.DataFrame(columns = ['Rater', 'Ratee', 'Rater_Id', 'Task_Title', 'Mean_Dev','Avg_Task_Rate', 'Num_Raters','Overal_Bias','Overal_Avg_Dist'])


#iterrate over different tasks
for name, group in unique_tasks:
    #translate ratings to numbers 
    group['success'] = group['success'].apply(lambda x: peer_ratings_habit[x])
    #average task rating
    avg_task_rating = group['success'].mean()    
    number_of_raters = len(group)

    
    for index, peer in group.iterrows(): 
        #peer information
        peer_id         = peer['peer_id']
        peer_core_id    = int(data_frame['panel_participant'].loc[data_frame['panel_participant']['id'] == peer_id ]['user_id'])
        peer_name       = data_frame['auth_user'].loc[data_frame['auth_user']['id'] == peer_core_id ]['first_name'].values[0]
        task_id         = peer['self_review_id']
        task_tile       = data_frame['panel_selfreview'].loc[data_frame['panel_selfreview']['id'] == task_id ]['title'].values[0]
        task_owner_id = data_frame['panel_selfreview'].loc[data_frame['panel_selfreview']['id'] == task_id ]['participant_id'].values[0]
        task_owner_core_id = int(data_frame['panel_participant'].loc[data_frame['panel_participant']['id'] == task_owner_id ]['user_id'])
        task_owner_name = data_frame['auth_user'].loc[data_frame['auth_user']['id'] == task_owner_core_id ]['first_name'].values[0]
        row             = [ peer_name, task_owner_name, peer_core_id, task_tile ,peer['success']-avg_task_rating, avg_task_rating, number_of_raters, None, None]
        rating_habit.loc[len(rating_habit)] = row

unique_peers = rating_habit.groupby(['Rater'])

for name, group in unique_peers:
    
    num_rates = len(group)
    avg_dist   = group['Mean_Dev'].mean() 
    bias       = np.sqrt((group['Mean_Dev']**2).sum()/num_rates)
    rating_habit.loc[ rating_habit['Rater'] == group['Rater'].values[0], 'Overal_Bias' ] = bias
    rating_habit.loc[ rating_habit['Rater'] == group['Rater'].values[0], 'Overal_Avg_Dist' ] = avg_dist
    
    
    
#################################################################
#Text analysis 
#################################################################
#    
import sys
sys.path.insert(0, '/Users/alimehdizadeh/Dropbox/CafeBazaar/Performance Evaluation/Farsi_Text_Processing/hazm-master')
import hazm 
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import io
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import gensim
from wordcloud import WordCloud
#from PIL import Image
#import PIL.ImageOps
import random
from persian_wordcloud.wordcloud import PersianWordCloud, add_stop_words

analyser = SentimentIntensityAnalyzer()
translator = Translator()
normalizer = hazm.Normalizer()    
informal_normalizer = hazm.InformalNormalizer()

new_words = {
        'output': 2,'result':2,
        'team'  : 2,
        'quality' : 2,
        'company' : 2,
        'follow-up':2,'follow up':2,'company':2,'interact':2,'interaction':2,
        'communication ':2,'teamwork':2,'goal':2,'teammates':2,'development':2,'grow':2,
        'goals':2,'accuracy':2,'learn':2,'learning':2,'effort':2,'focus':-2,'concentrate':2,
        'management':2,'issue':-2,'issues':-2,'burnout':-2,'impact':2,'participation':2,
        'cooperation':2,'professional':2,'bravo':2,'Ishalla':2,'contributions':2,'contribution':2,
        'feedback':2, 'independent':2, 'dedication':2,'present':2,'presence':2,'attendance':2,
        'workload':2,'constructive':2,'distructive':-2,'planning':2,'guard':-2, 'cost':-2, 'expected':-2,
        'challenge':-2,'problem-solving':2,'requires':-2,'concerns':-2,'concern':-2,'co-operation':2,
        'precise':2,'expectations':-2,'care':2,'beyond':2,'timely':2,'reading':2,'skill':2,
        'skills':2, 'concentration':-2, 'distract':-2,'expectation':-2, 'differs':-2,'differ':-2,'expect':-2,
        'analytical':2,
        }
analyser.lexicon.update(new_words)
##remove
#for item in new_words:
#    analyser.lexicon.pop(item)



#divide according to supervisors overviews
sys.path.insert(0, '/Users/alimehdizadeh/Dropbox/CafeBazaar/Performance Evaluation')
    
Text = {}

# translation
#for key in ratings:
#    
#    Text[key] = data_frame['panel_supervisoroverview'].loc[data_frame['panel_supervisoroverview']['performance'] == key]     
#    
#    Text[key]['translation'] = None
#    Text[key]['compound']    = None
#    Text[key]['neg']         = None
#    Text[key]['neu']         = None
#    Text[key]['pos']         = None
#    
#    for index, overview in Text[key].iterrows():
#        
#        #informal to formal
#        sentences =[]
#        formal_text_sentences = informal_normalizer.normalize(Text[key]['overview'][index])
#        for sent in formal_text_sentences:
#            sent = [word[0] for word in sent]
#            sent = " ".join(sent)
#            sentences.append(sent)
#            
#        Text[key]['overview'][index] = " ".join(sentences)
#        
#        Text[key]['overview'][index] = normalizer.normalize(Text[key]['overview'][index])
#        sentences = hazm.sent_tokenize(Text[key]['overview'][index])
#        
#        for i in range(len(sentences)):
#            try:
#                sentences[i] = translator.translate(sentences[i]).text
#                print(index,key)
#
#            except:
#                print('error', key, index, sentences[i])
#                sentences[i] = ' '
#                
#        Text[key]['translation'][index] = ' '.join(sentences)

#for key in Text:
#    Text[key].to_csv("supervisoroverview_"+str(key)+".csv")
            
#importing translations
Text = {}
for key in ratings:
    Text[key] = pd.read_csv("supervisoroverview_"+str(key)+".csv")   
    
#semantic analysis
for key in ratings:
    
    for index, overview in Text[key].iterrows():        
        
        scores = analyser.polarity_scores(Text[key]['translation'][index])
        for keygen in scores:
            Text[key][keygen][index] = scores[keygen]
              
####### tf-idf scores
with open('stopwords.txt', encoding='utf-8') as x:
    data = x.read().replace('\n', ',')

stemmer = hazm.Stemmer()

extra_stop_words = data.split(',') 

from bidi.algorithm import get_display
import arabic_reshaper

tokenizer = RegexpTokenizer(r'\w+')
supervisor_overview_df = data_frame['panel_supervisoroverview'].copy(deep=True)


texts = []
#cleaning the text
for index, overview in supervisor_overview_df.iterrows():
    supervisor_overview_df.at['overview',index] = normalizer.normalize(supervisor_overview_df['overview'][index])
    #sentences = hazm.sent_tokenize(supervisor_overview_df['overview'][index])            
    tokens = tokenizer.tokenize(supervisor_overview_df['overview'][index]) 
    longer_tokens = [get_display(arabic_reshaper.reshape(i)) for i in tokens if not(i in extra_stop_words)]
    texts.append(longer_tokens)    

#tfidf
dictionary = corpora.Dictionary(texts)
corpus     = [dictionary.doc2bow(text) for text in texts]
tfidf      = models.TfidfModel(corpus)

stop_words = add_stop_words([get_display(arabic_reshaper.reshape('سلام'))])


#wordcloud
for index in range(len(texts)):
    
    print (index)
 
    supervisee_id       = data_frame['panel_supervisoroverview'].loc[ index , : ]['supervisee_id']
    supervisee_user_id  = data_frame['panel_participant'].loc[ data_frame['panel_participant']['id'] == supervisee_id ]['user_id'].values[0]
    evaluation_round    = data_frame['panel_participant'].loc[ data_frame['panel_participant']['id'] == supervisee_id ]['round_id'].values[0]
    supervisee_name     = data_frame['auth_user'].loc[ data_frame['auth_user']['id'] == supervisee_user_id ]['first_name'].values[0]
    
    try:
        supervisor_id       = data_frame['panel_participant'].loc[ data_frame['panel_participant']['id'] == supervisee_id ]['supervisor_id'].values[0]
        supervisor_user_id  = data_frame['panel_participant'].loc[ data_frame['panel_participant']['id'] == supervisor_id ]['user_id'].values[0]
        supervisor_name     = data_frame['auth_user'].loc[ data_frame['auth_user']['id'] == supervisor_user_id ]['first_name'].values[0]
    except:
        supervisor_name: 'ثبت نشده'
        
    top_words = np.sort(np.array(tfidf[corpus[index]],dtype = [('word',int), ('score',float)]),order='score')[::-1]
    top_words_list = [(dictionary[word],score) for word,score in top_words]
    
    TopWords = [word[0] for word in top_words_list]
    for part in supervisee_name.split(" "):
        if get_display(arabic_reshaper.reshape(part)) in TopWords:
            top_words_list.pop(TopWords.index(get_display(arabic_reshaper.reshape(part))))
    
    top_words_dict = dict(top_words_list)
             
    try:
        wc = PersianWordCloud(only_persian=True,
                          background_color="black",
                          random_state=5,
                          margin=0,
                          width =400,
                          height=400,
                          stopwords = stop_words,
                          min_font_size=1,
                          max_font_size=250,
                          max_words=100).fit_words(top_words_dict)
        

        file_name = supervisor_name + ' برای ' + supervisee_name
        
        #title = rating + neg + neu + pos + their rankings
        supervisee_performance = data_frame['panel_supervisoroverview'].loc[ index , : ]['performance']
        
        sentiment_ratio = Text[supervisee_performance].loc[ Text[supervisee_performance]['supervisee_id'] == supervisee_id ]
        
        title  = 'Performance = ' + supervisee_performance + ', ' + 'Pos = ' + str("{0:.0%}".format(sentiment_ratio['pos'].values[0])) +', Neu = ' + str("{0:.0%}".format(sentiment_ratio['neu'].values[0])) + ' ,' + ' Neg = ' + str("{0:.0%}".format(sentiment_ratio['neg'].values[0]))
        
        plt.figure(figsize=(10,10))
        plt.title(title, fontsize=15)
        plt.tight_layout(pad=0)
        plt.axis("off")        
        plt.imshow(wc)
        plt.savefig("/Users/alimehdizadeh/Dropbox/CafeBazaar/Performance Evaluation/img/"+
                   str(evaluation_round)+'/'+file_name+".png",dpi = 300)
#        wc.to_file("/Users/alimehdizadeh/Dropbox/CafeBazaar/Performance Evaluation/img/"+
#                   str(evaluation_round)+'/'+file_name+".png")      
        plt.close()
    except:
        print('error')
       
       
        
########################3
#TEXT CALSSIFICATION
        
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
#import textblob
import string
import pandas as pd

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


frames = []
for i in range(1,11):
    frames.append(pd.read_csv("sentences_data_part"+str(i)+" - sentences_data_part"+str(i)+"_Done.csv"))
    

tag_data = pd.concat(frames)
tag_data.dropna()

text  = tag_data['Sentence'].copy(deep=True)
lable = tag_data['Senriment'].copy(deep=True)

lable = lable.replace(-2, -1)
lable = lable.replace(2, 1)

# split into training and test  
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(text, lable, train_size = 0.75)

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(text)

##########################################################################
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

##########################################################################
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(text)
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(text)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(text)
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

############################################

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


###########################################
    
# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print ("NB, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("NB, CharLevel Vectors: ", accuracy)

###########################################

# Linear Classifier on Count Vectors $Best  Model so fat 
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print ("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("LR, CharLevel Vectors: ", accuracy)

###########################################

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors: ", accuracy)

###########################################

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print ("RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)

###########################################
import fasttext
	
train_x_f = train_x.copy(deep = True)
train_y_f = ['__label__'+str(item)+'\t' for item in train_y]
df = pd.DataFrame({'Senriment':train_y_f,'Sentence':train_x_f})
df['overal'] = df['Senriment'].map(str) + df['Sentence']

with open('data.train.txt', 'w') as f:
    for entry in df['overal']:
        f.write(entry+'\n')

valid_x_f = valid_x.copy(deep = True)
valid_y_f = ['__label__'+str(item)+'\t' for item in valid_y]
df_v = pd.DataFrame({'Senriment':valid_y_f,'Sentence':valid_x_f})
df_v['overal'] = df_v['Senriment'].map(str) + df_v['Sentence']

with open('test.txt', 'w') as f:
    for entry in df_v['overal']:
        f.write(entry+'\n')

classifier = fasttext.supervised('data.train.txt', 'model', label_prefix='__label__')
result = classifier.test('test.txt')

print("FastText: ", result.precision )

######### peer rating sentiments 
   
    classifier  = linear_model.LogisticRegression().fit(xtrain_count, train_y)
    predictions = classifier.predict( count_vect.transform(data_frame['panel_peerreview']['description'] ))

for index, review in data_frame['panel_peerreview'].iterrows():
    print(review['description']) 
    print(linear_model.LogisticRegression().predict(review['description'])

    self_review_id = task['selfreview_id']
    self_review_author_id = data_frame['panel_selfreview'].loc[ data_frame['panel_selfreview']['id'] == self_review_id ]['participant_id'].tolist()[0]
    self_review_peer_id = task['participant_id']
    rate = data_frame['panel_peerreview'].loc[ (data_frame['panel_peerreview']['self_review_id'] == self_review_id) & (data_frame['panel_peerreview']['peer_id'] == self_review_peer_id) ]['success'].values










#tag setence sentiment
#from random import shuffle
#SentenceList = []
#
#
#for index, overview in data_frame['panel_supervisoroverview'].iterrows():  
#    data_frame['panel_supervisoroverview']['overview'][index] = normalizer.normalize(data_frame['panel_supervisoroverview']['overview'][index])
#    
#    for i in range(len(sentences)):
#        SentenceList.append(sentences[i])
#        
#shuffle(SentenceList)
#
#SentenceDataFrame = pd.DataFrame(SentenceList, columns = ['Sentence'])
#SentenceDataFrame['Senriment'] = None
#
#n = 650  #chunk row size
#list_df = [SentenceDataFrame[i:i+n] for i in range(0,SentenceDataFrame.shape[0],n)]
#
#for i in range(len(list_df)):
#    list_df[i].to_csv("sentences_data_part"+str(i+1)+".csv")
