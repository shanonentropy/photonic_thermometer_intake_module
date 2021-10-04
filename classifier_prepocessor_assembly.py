# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:36:24 2019

@author: zahmed

for the fbg data, take the transposed file, and add the label

later on we can add other attributes like Q factors and the like
"""

import os
import pandas as pd



#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\classification\classifier\fbg\fft_profile\fft'

for  fname in os.listdir(path_dir):
#    file_names.append(fname)
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = ',',  engine = 'python' )
    df['device'] = 'fbg'
    print(df.head(1))
    df.to_csv('fft_processed_'+fname)

