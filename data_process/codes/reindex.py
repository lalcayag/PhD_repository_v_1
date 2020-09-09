# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:25:17 2020

@author: lalc
"""

import numpy as np
import scipy as sp
import pandas as pd
import os



file_pre  = '/mnt/mimer/lalc/results/correlations/west/phase_2/corr_phase2_d'
file_pre2 = '/mnt/mimer/lalc/results/correlations/west/phase_2/corr_spec_phase2_d'

file_name = []
file_name2 = []
for i in range(16,35):

    file1 = file_pre+str(i).zfill(2)+'.h5'
    file2 = file_pre2+str(i).zfill(2)+'.h5'
    
    corr = pd.read_hdf(file1,  key= 'corr')
    L = pd.read_hdf(file1, key='L')
    
    corr.to_hdf(file2,'corr',mode='a', data_columns =corr.iloc[[0]].reset_index().columns.tolist(),format='table', append = True)
    L.to_hdf(file2,'L',mode='a', data_columns =L.iloc[[0]].reset_index().columns.tolist(),format='table', append = True)
    
        