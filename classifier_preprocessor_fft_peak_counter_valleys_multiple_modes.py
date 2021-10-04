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
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.interpolate import splrep, sproot
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import peakutils
# from peakutils.plot import plot as pplot

scaler =StandardScaler()

#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\classification\raw_data\ring_resonator_multiple_modes\drop'


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
filenames = []
Q = []
asym = []
number_of_peaks = []
fbg_decomp = []

cols = ['x', 'y']
for  fname in os.listdir(path_dir):
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 4,  engine = 'python', names =cols )
    df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
#    df.plot('x','y') 
    df['x_scale'] = minmax_scale(df.x, feature_range=(0,1))
    df['y_scale'] = minmax_scale(df.y, feature_range=(0,1))    
    indexes = peakutils.indexes(df.y_scale, thres=0.1, min_dist=100); print(indexes)
#    pplot(df.x_scale,df.y_scale, indexes);plt.show()
    peak_x = indexes[-2]
    a = peak_x + 700
    b = peak_x - 700
    x2 = df.x_scale[b:a]
    y2 = df.y_scale[b:a]
#    number_of_peaks.append(len(indexes))
    skewness = stats.skew(y2)
#    plt.plot(x2,y2)
    freq_axis = np.fft.fftfreq(df['x'].count())
    power = np.fft.fft(df['y']).real
    df1=pd.DataFrame(power)
    power_scale = scaler.fit_transform(df1)
    power_freq = pd.DataFrame(power_scale[:180], columns=['power'])
    plt.plot(power_freq);plt.show()
    fft_profile = power_scale.transpose()
#    fft_profile['Q'] = 
#    fft_profile['device'] = 'ring_resonator_multimode'
    fft_profile['asym'] = skewness
    fft_profile['num_peaks'] = len(indexes)
#    fft_decomp.append(fft_profile)
#    fft_profile.to_csv('fft'+fname)
    
    

    
#df_q = pd.DataFrame({'filnames':file_names,'fft':fft_p ,'quality_factor':Q, 'skew':asym, 'number_of_peaks': number_of_peaks})
#df_q.to_csv('peak_characteristics')