# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:47:22 2019

@author: Faisal Caem
"""
import numpy as np
import pandas as pd
import os
os.getcwd()
os.chdir(r'F:\NTUST\Lab Data Mining\K means')
#%%
#importing file
data=pd.read_csv('iris.csv')
head=list(data.columns)

#indexing data input and data label
inputIdx=head[:-1]
outputIdx=head[-1]

#Separating data into data input and data label
X=data[data.columns.intersection(inputIdx)]
y=data[outputIdx]

#initialing parameter
row=len(X)
column=len(inputIdx)
weight1=np.random.rand(row, column)
weight2=np.random.rand(4,1)

