#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:15:54 2019

@author: aliakbarmehdizadeh
"""
#curl -H "Authorization:Bearer <ACCESS TOKEN>"       https://<cafebazaar>.workable.com/spi/v3/jobs

import json
from Workable import Workable
import os
import time
from os import listdir
from os.path import isfile, join
import io
import sys
import datetime
from datetime import date

working_directory  = "/Users/alimehdizadeh/Dropbox/HR-CORE/Codes/Workable"
jobs_data_path  = "/Users/alimehdizadeh/Dropbox/HR-CORE/Data/Workable/ByJobs"

today = date.today()
last_month =  today - datetime.timedelta(days=1)
my_time = date(2019,5,17)

#merge dics in order to merge all the details 
def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

try:
    to_unicode = unicode
except NameError:
    to_unicode = str


#Workable API TOKEN
api_token = 'f9cbf94b2f0b243cdbdebee536d945efdea35c444f85a193bd27da38ff1ae400'
#api_url_base = 'https://cafebazaar.workable.com/spi/v3/{}'

workable = Workable(account ='cafebazaar',apikey = api_token)
#Get List Of All jobs
all_jobs = workable.job_list(state = '', updated_after = my_time )

#creating job directories
for job in all_jobs:
    if os.path.isdir('/Users/alimehdizadeh/Dropbox/HR-CORE/Data/Workable/ByJobs/'+str(job['title'])) == False:
        os.makedirs ('/Users/alimehdizadeh/Dropbox/HR-CORE/Data/Workable/ByJobs/'+str(job['title']))

for each_job in all_jobs:
    print (each_job['title'])   
    #candidate list for Each Job
    #created_after_candidate_list = workable.candidate_list(job = each_job["shortcode"], created_after = my_time, updated_after = '' )
    updated_after_candidate_list = workable.candidate_list(job = each_job["shortcode"], created_after = '', updated_after = my_time )
    candidate_list = updated_after_candidate_list
    #created_after_candidate_list + 

    #already saved candidates
    path = '/Users/alimehdizadeh/Dropbox/HR-CORE/Data/Workable/ByJobs/'+str(each_job['title'])
    already_saved_candidates = [f for f in listdir(path) if isfile(join(path, f))]
    
    for each_candidate in candidate_list:
        
        #check if we have already saved this candidate
        if ( not (each_candidate['name']+'.json' in already_saved_candidates) ):
            
            while True:
                B = True
                try:
                    #candidate general Information
                    each_candidate_details = workable.single_candidate_detail(candidate_id = each_candidate["id"], job = each_job["shortcode"])
                    #candiadte process Information
                    each_candidate_activities = workable.single_candidate_activities(candidate_id = each_candidate["id"])
                    #merge candidate information
                    each_candidate_full_details = merge(each_candidate_details, each_candidate_activities, path=None)
    
                    print ('\t', each_candidate['name'])
                    
                except:
                    #sleep and reiterate if error happened
                    print ('sleep')
                    B = False
                    time.sleep(60)
                    continue
                if B:
                    break    

            #write the candidate
            with io.open('/Users/alimehdizadeh/Dropbox/HR-CORE/Data/Workable/ByJobs/'+str(each_job['title'])+'/'+str(each_candidate['name'])+str(each_candidate['updated_at'])+'.json', 'w', encoding='utf-8' ) as outfile:
                str_ = json.dumps(each_candidate_full_details,
                                  sort_keys=True,
                                  separators=(',', ': '), ensure_ascii=False)
                outfile.write(to_unicode(str_))            
                    








