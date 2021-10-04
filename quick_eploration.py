# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:56:42 2019

@author: zahmed
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
#import numpy as np
#from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler

# cols = ['x', 'y']
scaler= StandardScaler()
path_dir = r'C:\Interpolation_Project\classification\qpfs_classifier'

for  fname in os.listdir(path_dir):
#    file_names.append(fname)
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 0,  engine = 'python')#, usecols = [0,1] ,names =cols )
    # df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
    # plt.plot(df.x,df.y)
    # plt.show()
#    b = scaler.fit_transform(df, y=df.y)
#    c=pd.DataFrame(b, columns = ['x','y'])
##    plt.plot(c.x,c.y)
#    
#    df['x']=minmax_scale(df.x)
#    freq_axis = np.fft.fftfreq(df['x'].count())
#    power = np.fft.fft(df['y']).real
##    power_norm = minmax_scale(power, feature_range=(0,1))
#    trunc = int(len(freq_axis)/2)
##    plt.plot(freq_axis[:100], power[:100])
#    plt.plot(power[:350])
#    plt.xlim(0,)
#    plt.ylim(0,.2)