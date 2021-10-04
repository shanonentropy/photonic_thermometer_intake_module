# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 07:43:57 2019

@author: zahmed
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filenames = []
path_dir = r'C:\holding_cell\dispersion'

file_path = os.path.join(path_dir, 'gel_5_spike_01.csv')  
first_frame = pd.read_csv( file_path, header = 2 )    


for  fname in os.listdir(path_dir):
        if fname != 'gel_5_spike_01.csv':
            fpath = os.path.join(path_dir, fname)
            df = pd.read_csv(fpath, header = 1)
            print(df.head())
            first_frame = pd.concat([first_frame, df], axis =0, sort=False )
            
plt.plot(first_frame['(ms)'], first_frame['(mV)'])
freq_axis = np.fft.fftfreq(first_frame['(ms)'].count())
power = np.fft.fft(first_frame['(mV)']).real
plt.plot(power)

#first_frame.to_csv('binary_dataset')
    
'''
    fbg  = 20
    qpfs = 10
    ring_resonators =
    ring_resonator_full_range =
    
    
    '''
    

 

        
            