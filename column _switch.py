# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:17:50 2019

@author: zahmed
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

path_dir = r'C:\Interpolation_Project\classification\raw_data\fbg_classification\weird_data'

for  fname in os.listdir(path_dir):
#    file_names.append(fname)
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 4,  engine = 'python', usecols=[0,1])
    df['x'], df['y'] = df.iloc[:,1], df.iloc[:,0]
    plt.plot(df['x'],df['y'])
    plt.show()
    df.iloc[:,2:].to_csv(fname)
    