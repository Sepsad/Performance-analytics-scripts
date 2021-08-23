# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bidi.algorithm import get_display
import arabic_reshaper
import os
import sys
from scipy import stats
import scipy
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pdb

# Importing the dataset
working_directory  = "/Users/alimehdizadeh/Dropbox/CafeBazaar/Leadership_Evaluation"
#current_data_path  = "/Users/aliakbarmehdizadeh/Dropbox/HR-CORE/Data/Work Engagement/
os.chdir(working_directory)

dataset = pd.read_csv('CLE.csv')
dataset['محصول'] = dataset['محصول'].fillna('NA')
dataset = dataset.dropna()

X = dataset.iloc[:,:]


#replace nan_Team with sayer
#X['تیم'].fillna('سایر', inplace=True)
number_of_categorical_columns = 1

#Making report directories based on categories
for column in dataset.iloc[:,0:number_of_categorical_columns]:
    UniqueValues = dataset[column].unique()
    #filter by unique values
    for item in UniqueValues:
        if item != item : continue
        #SubData = dataset.loc[ dataset[column] == item ]
        if os.path.isdir('/Users/alimehdizadeh/Dropbox/CafeBazaar/Leadership_Evaluation/report1/'+str(column)+'/'+str(item)) == False:
            os.makedirs ('/Users/alimehdizadeh/Dropbox/CafeBazaar/Leadership_Evaluation/report1/'+str(column)+'/'+str(item))

#unique values of categories 
for column in X.iloc[:,0:number_of_categorical_columns]:
    UniqueValues = X[column].unique()
    #filter by unique values
    for item in UniqueValues:
        SubData = X.loc[ X[column] == item ]
        #plotting the new subdata
        for column2 in SubData.iloc[:,number_of_categorical_columns + 5:]:
            #pdb.set_trace()
            
            fig, ax = plt.subplots( figsize=(15, 5), dpi = 300 )
            
            rows = []
            rows.append(item)
            
            #Question
            reshaped_text = arabic_reshaper.reshape(column2)
            artext = get_display(reshaped_text)
            
            #Leadr
            reshaped_text2 = arabic_reshaper.reshape(item)
            artext2 = get_display(reshaped_text2)
            
            #Firm
            FirmName = arabic_reshaper.reshape('شرکت')
            FirmName = get_display(FirmName)
            
            #FirmAvg
            FirmavgName = arabic_reshaper.reshape('میانگین شرکت')
            FirmavgName = get_display(FirmavgName)
            
            colors = ['g','c','m','y','k']
            lines = ["-","--","-.",":"]
            
            cell = []

            if ( np.issubdtype(X[column2].dtype, np.number)): 
                
                #Firm Hist                                          
                ax.hist(X[column2].dropna(), bins = [0.5,1.5,2.5,3.5,4.5,5.5],
                        edgecolor='k', alpha = 0.1, label=FirmName)
                #Samlpe Hist
                ax.hist(SubData[column2].dropna(), bins = [0.5,1.5,2.5,3.5,4.5,5.5], 
                        color='b', label=artext2)
                
                #FirmAvg     
                FirmMeanValue = X[column2].dropna().mean()
                ax.axvline(FirmMeanValue, color='r',
                           linestyle='dashed', linewidth = 3, label = FirmavgName)
                
                #TeamAvg
                TeamMeanValue = SubData[column2].dropna().mean()
                ax.axvline(TeamMeanValue, color='b',
                       linestyle='dashed', linewidth = 3,
                        label = artext2)

                cell_text = []
                cell_text.append([round(TeamMeanValue,2), round(TeamMeanValue-FirmMeanValue,2),len(SubData[column2].dropna()) ])
                            
                #Comparision With Others
                i = 0
                for item2 in X['محصول'].unique():
                    #if item2 == item: continue
                    
                    Others = X.loc[X['محصول'] == item2]
                    OthersMean = Others[column2].mean()
                                     
                    #OthersName
                    Name = 'میانگین '+str(item2)
                    OthersName = arabic_reshaper.reshape(Name)
                    OthersName = get_display(OthersName)
                    
                    cell_text.append([round(OthersMean,2), round(OthersMean-FirmMeanValue,2),len(Others[column2].dropna()) ])
                    
            
                    ax.axvline(OthersMean,  color = colors[i%5],
                           linestyle= lines[i%4], linewidth = 2.5, alpha = 0.8,
                            label = OthersName)  
                    
                    fig.suptitle(artext)
                    ax.grid(alpha = 0.3, linestyle='-', linewidth = 1 )
                    ax.legend(loc = 'center left', prop={'size': 12}, bbox_to_anchor=(1, 0.5))     
                    
                    rows.append(item2)
                    
                    i = i + 1
            else:
                continue
            
            #pdb.set_trace()
            
            rowstext = [ arabic_reshaper.reshape(part) for part in rows ]
            rowstext1 = [ get_display(part) for part in rowstext]
            
            #Save at Bazaar:
            YLabelFarsi = arabic_reshaper.reshape('تعداد نفرات رای‌دهنده')
            YLabelFarsi = get_display(YLabelFarsi)
            plt.ylabel(YLabelFarsi)
            
            QuestionRefined = column2.replace('/',' یا ')
            
# table miangin  

            AVGFarsi = arabic_reshaper.reshape('میانگین نمونه')
            AVGFarsi = get_display(AVGFarsi)
                              
            the_table = ax.table(cellText=[[item[0]] for item in cell_text ],
                                  #rowLabels= rowstext1,
                                  colLabels=[AVGFarsi],
                                  loc='bottom',
                                  rowLoc='center',
                                  #rowColours = RColor,
                                  colLoc='center',
                                  cellLoc='center',
                                  #cellColours=plt.cm.hot(normal(vals)),
                                  bbox= [0.2, -1, 0.2, 0.9],
                                  )   
            
#table number of participants:          
            the_table = ax.table(cellText=[[item[2]]  for item in cell_text ],
                                  rowLabels= rowstext1,
                                  colLabels=[YLabelFarsi],
                                  loc='bottom',
                                  rowLoc='center',
                                  #rowColours = RColor,
                                  colLoc='center',
                                  cellLoc='center',
                                  #cellColours=plt.cm.hot(normal(vals)),
                                  bbox= [0.0, -1, 0.2, 0.9])      
            
#table fasele az miangin sherkat:
            YLabeldisavg = arabic_reshaper.reshape('فاصله میانگین نمونه از فنی‌ها')
            YLabeldisavg = get_display(YLabeldisavg)
            
            vals = np.around([ [item[1]] for item in cell_text ],2)
            normal = plt.Normalize(vals.min()-2, vals.max()+1) 
            #pdb.set_trace()
            
            the_table = ax.table(cellText=[[item[1]] for item in cell_text ],
                                  colLabels=[YLabeldisavg],
                                  loc='bottom',
                                  rowLoc='center',
                                  colLoc='center',
                                  cellLoc='center',
                                  cellColours=plt.cm.hot(normal(vals)),
                                  bbox= [0.4, -1, 0.2, 0.9])  
            
            
            plt.savefig('/Users/alimehdizadeh/Dropbox/CafeBazaar/Leadership_Evaluation/report1/'+column+'/'+item+'/'+item+'_'+QuestionRefined+'.png',
                        bbox_inches='tight')
            
            plt.close()
            #sys.exit()
        #sys.exit()
            #plt.show()
            
#
## Correlations
#import re 
## Importing the dataset
#dataset = pd.read_csv('LE.csv')
#
#Balad = dataset[ dataset['تیم'] == 'بلد'].reset_index(drop=True)
#Bazar = dataset[ dataset['تیم'] == 'بازار'].reset_index(drop=True)
#Divar = dataset[ dataset['تیم'] == 'دیوار'].reset_index(drop=True)
#Zirsakht = dataset[ dataset['تیم'] == 'زیرساخت'].reset_index(drop=True)
#
#
#Teams = [Balad,Bazar, Divar, Zirsakht]
#TeamsNames = ['Balad','Bazar', 'Divar', 'Zirsakht']
#
#X = dataset.iloc[0:115,4:]
#X = X.drop(X.columns[-3], axis=1)
#
#
#Lables = []
#
#for item in X:
#    Tag = arabic_reshaper.reshape(re.sub("(.{50})", "\\1\n", item, 0, re.DOTALL))
#    Tag = get_display(Tag)
#    Lables.append(Tag)
#    
#corr = X.corr()
#fig = plt.figure(figsize=(8, 6), dpi = 300)
#ax = fig.add_subplot(111)
#cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
#fig.colorbar(cax)
#    
#for (i, j), z in np.ndenumerate(corr):
#    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize = 5 )
#        
#ticks = np.arange(0,len(X.columns),1)
#ax.set_xticks(ticks)
#plt.xticks(rotation=90)
#ax.set_yticks(ticks)
#ax.set_xticklabels(Lables, fontsize=3)
#ax.set_yticklabels(Lables, fontsize=3)
#plt.savefig('/Users/aliakbarmehdizadeh/Dropbox/CafeBazaar/Leadership_Evaluation/'+'CafeCorr'+'.png',
#                bbox_inches='tight')
#
#numberofteams = 0
#fig = plt.figure(figsize=(14, 12), dpi = 300)
#fig.subplots_adjust(hspace=0.1, wspace=0.0)
#
#for item in Teams:
#    
#    item = item.iloc[:,4:]
#    item = item.drop(item.columns[-3], axis=1)
#    corr = item.corr()
#    #fig = plt.figure(figsize=(8, 6), dpi = 300)
#    #ax = fig.add_subplot(111)
#    ax = plt.subplot(2,2,numberofteams+1)
#    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
#    #fig.colorbar(cax)
#    
#    for (i, j), z in np.ndenumerate(corr):
#        if ( i != j):
#            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize = 5 )
#        else:
#            ax.text(j, i, '{:0.2f}'.format(item.iloc[:,i].mean()), ha='center', va='center', fontsize = 6, color = 'w' )
#            
#        
#    ticks = np.arange(0,len(item.columns),1)
#    
#    if ( numberofteams+1 == 1 ): 
#        ax.set_xticks(ticks)
#        plt.xticks(rotation=90)
#        ax.set_yticks(ticks)
#        ax.set_xticklabels(Lables, fontsize=4)
#        ax.set_yticklabels(Lables, fontsize=4)
#        ax.set_title("Balad", x = 1.02 , y = 0.8, rotation = 270 )
#        
#    if ( numberofteams+1 == 2 ): 
#        ax.set_xticks(ticks)
#        plt.xticks(rotation=90)
#        ax.set_yticks([])
#        ax.set_xticklabels(Lables, fontsize=4)
#        #ax.set_yticklabels(Lables, fontsize=3)    
#        fig.colorbar(cax)
#        ax.set_title("Bazar", x = -0.02 , y = 0.3, rotation = 90)
#
#
#    if ( numberofteams+1 == 3 ): 
#        ax.set_xticks([])
#        plt.xticks(rotation=90)
#        ax.set_yticks(ticks)
#        #ax.set_xticklabels(Lables, fontsize=3)
#        ax.set_yticklabels(Lables, fontsize=4)  
#        ax.set_title("Divar", x = 1.02 , y = 0.8, rotation = 270)
#
#        
#    if ( numberofteams+1 == 4 ): 
#        ax.set_xticks([])
#        ax.set_yticks([])
#        #ax.set_xticks(ticks)
#        #plt.xticks(rotation=90)
#        #ax.set_yticks(ticks)
#        #ax.set_xticklabels(Lables, fontsize=3)
#        #ax.set_yticklabels(Lables, fontsize=3) 
#        fig.colorbar(cax)    
#        ax.set_title("Zirsakht", x = -0.02 , y = 0.3, rotation = 90)
#
#           
#        #Saving:
#    #plt.savefig('/Users/aliakbarmehdizadeh/Dropbox/HR-CORE/Data/Work Engagement/spring 98/report_nontech/'+column+'/'+item+'/'+item+column2+'.png',
#    #                    bbox_inches='tight')        
#    
#    plt.savefig('/Users/aliakbarmehdizadeh/Dropbox/CafeBazaar/Leadership_Evaluation/report/'+TeamsNames[numberofteams]+'.png',
#                bbox_inches='tight')  
#    
#    numberofteams = numberofteams + 1
#    #plt.show()
#
##plt.savefig('/Users/aliakbarmehdizadeh/Dropbox/CafeBazaar/Leadership_Evaluation/'+'Corr_Compare'+'.png',
##                bbox_inches='tight') 
