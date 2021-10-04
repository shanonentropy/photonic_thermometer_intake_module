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
path_dir = r'\\elwood.nist.gov\685\users\zahmed\My Documents\ZResearchfiles-NIST\c2c12 gel study\live_data_logging_day_9\gel_5_spike'

file_path = os.path.join(path_dir, 'gel_5_spike_01.csv')  
first_frame = pd.read_csv( file_path, header = 2 )    


for  fname in os.listdir(path_dir):
        if fname != 'gel_5_spike_01.csv':
            fpath = os.path.join(path_dir, fname)
            df = pd.read_csv(fpath, header = 1)
            print(df.head())
            first_frame = pd.concat([first_frame, df], axis =0, sort=False )
            
plt.plot(first_frame['(ms)'], first_frame['(mV)'])
freq_axis = np.fft.fftfreq(int(first_frame['(ms)'].dropna().count()*0.0000001))
power = np.fft.fft(first_frame['(mV)'].dropna().real)
plt.plot(freq_axis, power[:])
plt.ylim(0,1000000)
plt.xlim(0,10)

#first_frame.to_csv('binary_dataset')
 

 

        
            