# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:38:27 2019

@author: zahmed

this program is part of the SENSOR_CLASSIFIER program's pre-processing routine
it will take in all the data from the sensor folder and display it 
"""
import os
import pandas as pd
from sklearn.preprocessing import minmax_scale
from scipy import interpolate
from scipy.interpolate import splrep, sproot
import numpy as np
import matplotlib.pyplot as plt

#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\classification\fbg_classification'
#loop over the files and then create a list of file names to later iterate over
    
''' for each spectra we need to extract the following set of information
number of peaks 
    if more than one peak, peak-to-peak distace (ppd) and delta ppd
Q of the device
from the normalized spectra, skewness
and intensity of profile of the spectra

the first part is to just feed in data with profile and label and 
see if the classifier works, if not, keep adding more features

so this program will just take in the data, fit it, create a dataset with 
known pitch of 0.003 nm and output data ninmax scaled profile data with the 
same name

'''
file_names = []
Q = []

cols = ['x', 'y']
for  fname in os.listdir(path_dir):
    file_names.append(fname)
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 4,  engine = 'python', names =cols )
    df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
#    df.plot('x','y')  
#    m = df.x.count()
#    s_val = 1/(m - np.sqrt(2*m))
    tck = interpolate.splrep(df.x,df.y,s=0.0000001) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
    x_ = np.arange (df.x.min(),df.x.max(), 0.003)
    y_ = interpolate.splev(x_, tck, der=0)
#    plt.plot(df['x'],df['y'])
#    plt.scatter(x_,y_)
#    plt.show()
    HM =(np.max(y_)-np.min(y_))/2
    w = splrep(x_, y_ - HM, k=3)
#        print(sproot(w_j))
    try:
        if len(sproot(w))%2 == 0:
            r1 , r2 = sproot(w)
#            print(r1, r2)
            FWHM = np.abs(r1 - r2)
#            print('FWHM=',FWHM)
            center_wavelength = r1 + FWHM/2
            Q.append(center_wavelength/FWHM)
    except (TypeError, ValueError):
        print(fname,'error')
        continue
    df1 = pd.DataFrame(y_, x_)
#    print(df1.head(3))
#    df1['x_scale'] = minmax_scale(x_, feature_range=(0,1))
#    df1['y_scale'] = minmax_scale(y_, feature_range=(0,1))
#    plt.plot(df1['x_scale'], df1['y_scale'])
#    df1.reset_index(inplace=True)
#    df1.drop('index', axis=1, inplace=True)
#    df2 = df1[['x_scale', 'y_scale']]
#   print(df2.head(3))
#    tmp = df2[['x_scale', 'y_scale']].transpose()
#    tmp = pd.DataFrame(tmp.loc['y_scale'].T).T
#    print(tmp)
#    tmp.to_csv(fname)
    
df_q = pd.DataFrame({'filnames':file_names, 'quality_factor':Q})
df_q.to_csv('quality_factor')
