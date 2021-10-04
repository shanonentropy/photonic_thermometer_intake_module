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
#from scipy import interpolate
#from scipy.interpolate import splrep, sproot
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
#import peakutils
#from peakutils.plot import plot as pplot

scaler =StandardScaler()

#path to directory with the relevant files
path_dir = r':\Interpolation_Project\classification\raw_data\qps_fbg_classification\scan_data'

 
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
#filenames = []
#Q = []
#asym = []
#number_of_peaks = []
#fbg_decomp = []

cols = ['x', 'y']
for  fname in os.listdir(path_dir):
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 6,  engine = 'python',names =cols )
    df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
    freq_axis = np.fft.fftfreq(df['x'].count())
    print(df.x.count())
    power = np.fft.fft(df['y']).real
    df1=pd.DataFrame(power)
    power_scale = scaler.fit_transform(df1)
    power_freq = pd.DataFrame(power_scale[:180], columns=['power'])
    plt.plot(power_freq);plt.show()
    fft_profile = power_freq.transpose()
    fft_profile.reset_index(drop=True, inplace=True)
    fft_profile['device'] = 'qpfs'
    fft_profile['asym'] = stats.skew(df.y)
    fft_profile['num_peaks'] = 1
    fft_profile['Q'] = np.random.uniform(13900, 14100)
    print(fft_profile)
    fft_profile.to_csv('fft'+fname)

    
