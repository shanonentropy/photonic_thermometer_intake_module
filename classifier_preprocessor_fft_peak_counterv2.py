# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:38:27 2019

@author: zahmed

this program is part of the SENSOR_CLASSIFIER program's pre-processing routine
it will take in all the data from the sensor folder and display it 
"""
import os
import pandas as pd
#from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.interpolate import splrep, sproot
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot

scaler =StandardScaler()

#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\ring_resonator\09032015_LeTi_T_depn\Chip_7\Device_20'


#loop over the files and then create a list of file names to later iterate over
    

filenames = []
Q = []
asym = []
number_of_peaks = []
fbg_decomp = []

cols = ['x', 'y']
for  fname in os.listdir(path_dir):
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 8,  engine = 'python',usecols=[0,1], names =cols )
    df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
#    df.plot('x','y') 
    freq_axis = np.fft.fftfreq(df['x'].count())
    power = np.fft.fft(df['y']).real
    df1 = pd.DataFrame(power)
    power_scale = scaler.fit_transform(df1)
#    plt.plot(power_scale[:180]) 
    power_freq = pd.DataFrame(power_scale, columns=['power']); print(len(power_freq))
    pad = np.arange(93,180,1)
    power_freq_pad = pd.concat([power_freq, pd.DataFrame(index = pad)], axis = 0, sort=False)
    power_freq_pad.fillna(0,inplace = True)
#    plt.plot(power_freq)
    fft_profile = power_freq_pad[:180].transpose()
    indexes = peakutils.indexes(df.y, thres=0.1, min_dist=100)
    fft_profile['device'] = 'ring_resonator_single_mode'
    fft_profile['asym'] = stats.skew(df.y)
    fft_profile['num_peaks'] = len(indexes)
    pplot(df.x, df.y, indexes)
    plt.show()
#    print(indexes)
    tck = interpolate.splrep(df.x,df.y,s=0.00000001) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
    x_ = np.arange (df.x.min(),df.x.max(), 0.003)
    y_ = interpolate.splev(x_, tck, der=0)
#    plt.plot(x_,y_)
#    plt.plot(df.x, df.y)
    HM =(np.max(y_)-np.min(y_))/2
    w = splrep(x_, y_ - HM, k=3)
#        print(sproot(w_j))
    try:
        if len(sproot(w))%2 == 0:
            r1 , r2 = sproot(w)
            print(r1, r2)
            FWHM = np.abs(r1 - r2)
            center_wavelength = r1 + FWHM/2
            Q = (center_wavelength/FWHM)
        else:
            Q = 23066
    except (TypeError, ValueError):
        Q = 23066
        print(fname,'error')
        continue
    
    fft_profile['Q'] = Q
    print(fft_profile)
    #output normalized fft profile

#    fft_decomp.append(fft_profile)
#    fft_profile.to_csv('fft'+fname)
#    
    

    
#df_q = pd.DataFrame({'filnames':file_names,'fft':fft_p ,'quality_factor':Q, 'skew':asym, 'number_of_peaks': number_of_peaks})
#df_q.to_csv('peak_characteristics')