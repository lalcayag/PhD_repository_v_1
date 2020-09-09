# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:39:47 2020

@author: lalc
"""
import numpy as np
import scipy as sp
import pandas as pd
import os
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from scipy.spatial import Delaunay
from os.path import isfile, join, getsize
from os import listdir
import ppiscanprocess.windfieldrec as wr
import joblib

date = datetime(1904, 1, 1) 

# In[Files]   
############### files and sql engines ########################################
file_name0 = '/mnt/mimer/lalc/db/scans/phase_2/west/raw_filt_0_phase2.db'
file_name1 = '/mnt/mimer/lalc/db/scans/phase_2/west/raw_filt_1_phase2.db'
file_in_path = '/mnt/mimer/lalc/results/correlations/west/phase_2'
L_file = '/mnt/mimer/lalc/results/correlations/west/L_ph2.pkl'
file_out = '/mnt/mimer/lalc/results/correlations/west/Table_phase2_L.h5'
filelist = [join(file_in_path,filename) for filename in listdir(file_in_path)]
csv_database_0_ind = create_engine('sqlite:///'+file_name0)
csv_database_1_ind = create_engine('sqlite:///'+file_name1)
bad_files = []
for file_name in filelist:
    ###################daily dataframe 10min###################################
    store = pd.HDFStore(file_name, more = 'r')
    min10 = store.__contains__('L_10min')
    store.close()
    print(file_name)
    bad_files.append((file_name, min10))
    
joblib.dump(bad_files,'badfiles.pkl')
    
