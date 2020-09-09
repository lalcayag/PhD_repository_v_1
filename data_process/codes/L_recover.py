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
date = datetime(1904, 1, 1) 

# In[]

with open('/mnt/mimer/lalc/results/correlations/L', 'rb') as reader:
    res_flux_2 = pickle.load(reader) 