# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:25:17 2020

@author: lalc
"""

import numpy as np
import scipy as sp
import pandas as pd
import os



file_pre  = '/mnt/mimer/lalc/results/correlations/west/phase_2/corr_phase2_d01.h5'
file_pre2 = '/mnt/mimer/lalc/results/correlations/west/phase_2/corr_spec_phase2_d01.h5'
    
corr = pd.read_hdf(file_pre,  key= 'corr')
L = pd.read_hdf(file_pre, key='L')

corr.to_hdf(file_pre2,'corr',mode='a', data_columns =corr.iloc[[0]].reset_index().columns.tolist(),format='table', append = True)
L.to_hdf(file_pre2,'L',mode='a', data_columns =L.iloc[[0]].reset_index().columns.tolist(),format='table', append = True)

    