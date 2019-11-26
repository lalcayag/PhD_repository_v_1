# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:32:44 2019

@author: lalc
"""
import mysql.connector
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import scipy as sp

# In[Directory of the input and output data]

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()

root = tkint.Tk()
file_out_path_df = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir DataFrame')
root.destroy()


# In[MySQL Skipheia]
mydb = mysql.connector.connect(
  host="IEPT0763-PC",
  user="newuser",
  database = "vinddata",
  passwd="pTq7mRjo"
)
cursor = mydb.cursor()
cursor.execute("SHOW TABLES") 
for (table_name,) in cursor:
        print(table_name)
        
        
     
# In[]

mydb = mysql.connector.connect(
  host="ri-veadbs04",
  user="lalc",
#  database = "oesterild_light_masts",
  passwd="La469lc"
)

cursor = mydb.cursor()
cursor.execute("show databases")
for (databases) in cursor:
     print(databases)
     
     
mydb2 = mysql.connector.connect(
  host="ri-veadbs03",
  user="lalc",
#  database = "oesterild_light_masts",
  passwd="La469lc"
)

cursor = mydb.cursor()
cursor.execute("show databases")
for (databases) in cursor:
     print(databases)     
     
sql_select_Query = "select * from "
cursor.execute(sql_select_Query)
records = cursor.fetchall()    
cursor.execute("SHOW TABLES") 
for (table_name,) in cursor:
        print(table_name)
        
     
     
cursor.execute("desc caldata_2016_01_20hz")
col = [column[0] for column in cursor.fetchall()]     