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
#from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.interpolate import splrep, sproot
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot

#scaler =StandardScaler()

#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\classification\raw_data\wbg_classification'


#loop over the files and then create a list of file names to later iterate over
    

filenames = []
Q = []
asym = []
number_of_peaks = []
fbg_decomp = []

cols = ['x', 'y']
for  fname in os.listdir(path_dir):
    filenames.append(fname)
    
df  = pd.read_csv((os.path.join(path_dir, filenames[0])), sep = '\t', header = 4,  engine = 'python', names =cols )
df1 = pd.read_csv((os.path.join(path_dir, filenames[1])), sep = '\t', header = 4,  engine = 'python', names =cols )
df2 = pd.read_csv((os.path.join(path_dir, filenames[2])), sep = '\t', header = 4,  engine = 'python', names =cols )
df3 = pd.read_csv((os.path.join(path_dir, filenames[3])), sep = '\t', header = 4,  engine = 'python', names =cols )
df4 = pd.read_csv((os.path.join(path_dir, filenames[4])), sep = '\t', header = 4,  engine = 'python', names =cols )
df5 = pd.read_csv((os.path.join(path_dir, filenames[5])), sep = '\t', header = 4,  engine = 'python', names =cols )
df6 = pd.read_csv((os.path.join(path_dir, filenames[6])), sep = '\t', header = 4,  engine = 'python', names =cols )
df7 = pd.read_csv((os.path.join(path_dir, filenames[7])), sep = '\t', header = 4,  engine = 'python', names =cols )
df8 = pd.read_csv((os.path.join(path_dir, filenames[8])), sep = '\t', header = 4,  engine = 'python', names =cols )
    
plt.plot(df.x[:70] , df.y[:70] )
plt.plot(df1.x[:70] , df1.y[:70] )
plt.plot(df2.x[:] , df2.y[:] )
plt.plot(df3.x[:] , df3.y[:] )
plt.plot(df4.x[1900:4000] , df4.y[1900:4000] )
plt.plot(df5.x[1200:3200] , df5.y[1200:3200] )
plt.plot(df6.x[2900:4800] , df6.y[2900:4800] )
plt.plot(df7.x[:110] , df7.y[:110] )
plt.plot(df8.x[50:120] , df8.y[50:120] )


s = stats.skew(df.y[:70])
s1 = stats.skew(df1.y[:70])
s2 = stats.skew(df2.y)
s3 = stats.skew(df3.y)
s4 = stats.skew(df4.y[1900:4000])
s5 = stats.skew(df5.y[1200:3200])
s6 = stats.skew(df6.y[2900:4800])
s7 = stats.skew(df7.y[:110])
s8 = stats.skew(df8.y[50:120])

x_scale = minmax_scale(df['x'][:70], feature_range=(0,1))
y_scale = minmax_scale(df.y[:70], feature_range=(0,1))
x_scale_1 = minmax_scale(df1.x[:70], feature_range=(0,1))
y_scale_1  = minmax_scale(df1.y[:70], feature_range=(0,1))
x_scale_2 = minmax_scale(df2.x[:], feature_range=(0,1))
y_scale_2  = minmax_scale(df2.y[:], feature_range=(0,1))
x_scale_3 = minmax_scale(df3.x[:], feature_range=(0,1))
y_scale_3  = minmax_scale(df3.y[:], feature_range=(0,1))
x_scale_4 = minmax_scale(df4.x[1900:4000], feature_range=(0,1))
y_scale_4  = minmax_scale(df4.y[1900:4000], feature_range=(0,1))
x_scale_5 = minmax_scale(df5.x[1200:3200], feature_range=(0,1))
y_scale_5  = minmax_scale(df5.y[1200:3200], feature_range=(0,1))
x_scale_6 = minmax_scale(df6.x[2900:4800], feature_range=(0,1))
y_scale_6  = minmax_scale(df6.y[2900:4800], feature_range=(0,1))
x_scale_7 = minmax_scale(df7.x[:110], feature_range=(0,1))
y_scale_7  = minmax_scale(df7.y[:110], feature_range=(0,1))
x_scale_8 = minmax_scale(df8.x[50:120], feature_range=(0,1))
y_scale_8  = minmax_scale(df8.y[50:120], feature_range=(0,1))

plt.plot(x_scale , y_scale )
plt.plot(x_scale_1 , y_scale_1 )
plt.plot(x_scale_2 , y_scale_2 )
plt.plot(x_scale_3 , y_scale_3 )
plt.plot(x_scale_4 , y_scale_4 )
plt.plot(x_scale_5 , y_scale_5 )
plt.plot(x_scale_6 , y_scale_6 )
plt.plot(x_scale_7 , y_scale_7 )
plt.plot(x_scale_8 , y_scale_8 )


dfa =  pd.DataFrame({ 'x' : x_scale , 'y' :  y_scale })
dfa1 =  pd.DataFrame({ 'x' :x_scale_1 , 'y' : y_scale_1} )
dfa2 =  pd.DataFrame({ 'x' :x_scale_2 , 'y' : y_scale_2} )
dfa3 =  pd.DataFrame({ 'x' :x_scale_3 , 'y' : y_scale_3} )
dfa4 =  pd.DataFrame({ 'x' :x_scale_4 , 'y' : y_scale_4} )
dfa5 =  pd.DataFrame({ 'x' :x_scale_5 , 'y' : y_scale_5} )
dfa6 =  pd.DataFrame({ 'x' :x_scale_6 , 'y' : y_scale_6} )
dfa7 =  pd.DataFrame({ 'x' :x_scale_7 , 'y' : y_scale_7} )
dfa8 =  pd.DataFrame({ 'x' :x_scale_8 , 'y' : y_scale_8} )
    
df=df[:70]
df1 = df1[:70]
df2 = df2
df3 =df3
df4 = df4[1900:4000]
df5 = df5[1200:3200]
df6 = df6[2900:4800]
df7 = df7[:110] 
df8 = df8[50:120]


indexes_0 = peakutils.indexes(dfa.y, thres=0.5, min_dist=30)
indexes_1 = peakutils.indexes(dfa1.y, thres=0.2, min_dist=30)
indexes_2 = peakutils.indexes(dfa2.y, thres=0.1, min_dist=30)
indexes_3 = peakutils.indexes(dfa3.y, thres=0.1, min_dist=30)
indexes_4 = peakutils.indexes(dfa4.y, thres=0.4, min_dist=30)
indexes_5 = peakutils.indexes(dfa5.y, thres=0.5, min_dist=30)
indexes_6 = peakutils.indexes(dfa6.y, thres=0.5, min_dist=30)
indexes_7 = peakutils.indexes(dfa7.y, thres=0.1, min_dist=30)
indexes_8 = peakutils.indexes(dfa8.y, thres=0.1, min_dist=3)

pplot(dfa.x, dfa.y, indexes_0)
pplot(dfa1.x, dfa1.y, indexes_1)
pplot(dfa2.x, dfa2.y, indexes_2)
pplot(dfa3.x, dfa3.y, indexes_3)
pplot(dfa4.x, dfa4.y, indexes_4)
pplot(dfa5.x, dfa5.y, indexes_5)
pplot(dfa6.x, dfa6.y, indexes_6)
pplot(dfa7.x, dfa7.y, indexes_7)
pplot(dfa8.x, dfa8.y, indexes_8)

fname =[]
full_width = []
peak_center = []
q_factor =[]
fft =[]
idx = [indexes_0, indexes_1, indexes_2 , indexes_3 , indexes_4 , indexes_5 , indexes_6 , indexes_7, indexes_8]
#dfidx = [dfa[:70], dfa1[:70], dfa2, dfa3, dfa4[1900:4000], dfa5[1200:3200], dfa6[2900:4800], dfa7[:110], dfa8[50:120]]
dfidx = [df, df1, df2, df3, df4, df5, df6, df7, df8]

a = idx[0]
b = np.argmax(np.diff(a))
c1,c2 = a[b],a[b+1]
d1,d2 = df.iloc[c1,0] , df.iloc[c2,0]
width = d2-d1 
center = (d1+width)
fname.append(filenames[0])
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

a=idx[1]
b=np.argmax(np.diff(a))
c1,c2=a[b],a[b+1]
d1,d2=df1.iloc[c1,0] , df1.iloc[c2,0]
width= d2-d1 
center= (d1+width)
fname.append(filenames[1])
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

a=idx[2]
b=np.argmax(np.diff(a))
c1,c2=a[b],a[b+1]
d1,d2=df2.iloc[c1,0] , df2.iloc[c2,0]
width= d2-d1 
center= (d1+width)
fname.append(filenames[2])
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

a=idx[3]
b=np.argmax(np.diff(a))
c1,c2=a[b],a[b+1]
d1,d2=df3.iloc[c1,0] , df3.iloc[c2,0]
width= d2-d1 
center= (d1+width)
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

a=idx[4]
b=np.argmax(np.diff(a))
c1,c2=a[b],a[b+1]
d1,d2=df4.iloc[c1,0] , df4.iloc[c2,0]
width= d2-d1 
center= (d1+width)
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

a=idx[5]
b=np.argmax(np.diff(a))
c1,c2=a[b],a[b+1]
d1,d2 = df5.iloc[217,0] , df5.iloc[1720,0]
width= d2-d1 
center= (d1+width)
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

a=idx[6]
b=np.argmax(np.diff(a))
c1,c2=a[b],a[b+1]
d1,d2=df6.iloc[c1,0] , df6.iloc[c2,0]
width= d2-d1 
center= (d1+width)
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

a=idx[7]
b=np.argmax(np.diff(a))
c1,c2=a[b],a[b+1]
d1,d2=df7.iloc[c1,0] , df7.iloc[c2,0]
width= d2-d1 
center= (d1+width)
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

a=idx[8]
b=np.argmax(np.diff(a))
c1,c2=a[b],a[b+1]
d1,d2=df8.iloc[c1,0] , df8.iloc[c2,0]
width= d2-d1 
center= (d1+width)
full_width.append(width)
peak_center.append(center)
q_factor.append(center/width)

freq_axis = np.fft.fftfreq(dfa['x'].count())
power = np.fft.fft(dfa['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
pad = np.arange(35,180,1)
power_freq_pad = pd.concat([power_freq[:35], pd.DataFrame(index = pad)], axis = 0, sort=False)
pp_f = power_freq_pad.fillna(0.001)
fft_profile_0 = pp_f.transpose()
fft0 = pd.DataFrame(fft_profile_0)
fft0['Q'] = q_factor[0]
fft0['device'] = 'wbg'
fft0['asym'] = s 
fft0['num_peaks'] = 1   

freq_axis = np.fft.fftfreq(dfa1['x'].count())
power = np.fft.fft(dfa1['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
pad = np.arange(35,180,1)
power_freq_pad = pd.concat([power_freq[:35], pd.DataFrame(index = pad)], axis = 0, sort=False)
pp_f = power_freq_pad.fillna(0)
fft_profile_1 = pp_f.transpose()
fft1 = pd.DataFrame(fft_profile_1)
fft1['Q'] = q_factor[1]
fft1['device'] = 'wbg'
fft1['asym'] = s1 
fft1['num_peaks'] = 1

freq_axis = np.fft.fftfreq(dfa2['x'].count())
power = np.fft.fft(dfa3['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
pad = np.arange(35,180,1)
power_freq_pad = pd.concat([power_freq[:35], pd.DataFrame(index = pad)], axis = 0, sort=False)
pp_f = power_freq_pad.fillna(0)
fft_profile_2 = power_freq.transpose()
fft2= pd.DataFrame(fft_profile_2)
fft2['Q'] = q_factor[2]
fft2['device'] = 'wbg'
fft2['asym'] = s2
fft2['num_peaks'] = 1

freq_axis = np.fft.fftfreq(dfa3['x'].count())
power = np.fft.fft(dfa3['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
fft_profile_3 = power_freq.transpose()
fft3 = pd.DataFrame(fft_profile_0)
fft3['Q'] = q_factor[3]
fft3['device'] = 'wbg'
fft3['asym'] = s3
fft3['num_peaks'] = 1

freq_axis = np.fft.fftfreq(dfa4['x'].count())
power = np.fft.fft(dfa4['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
fft_profile_4 = power_freq.transpose()
fft4 = pd.DataFrame(fft_profile_0)
fft4['Q'] = q_factor[4]
fft4['device'] = 'wbg'
fft4['asym'] = s4 
fft4['num_peaks'] = 1


freq_axis = np.fft.fftfreq(dfa5['x'].count())
power = np.fft.fft(dfa5['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
fft_profile_5 = power_freq.transpose()
fft5 = pd.DataFrame(fft_profile_0)
fft5['Q'] = q_factor[5]
fft5['device'] = 'wbg'
fft5['asym'] = s5
fft5['num_peaks'] = 1


freq_axis = np.fft.fftfreq(dfa6['x'].count())
power = np.fft.fft(dfa6['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
fft_profile_6 = power_freq.transpose()
fft6 = pd.DataFrame(fft_profile_0)
fft6['Q'] = q_factor[6]
fft6['device'] = 'wbg'
fft6['asym'] = s6
fft6['num_peaks'] = 1

freq_axis = np.fft.fftfreq(dfa7['x'].count())
power = np.fft.fft(dfa7['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
pad = np.arange(56,181,1)
power_freq_pad = pd.concat([power_freq[:56], pd.DataFrame(index = pad)], axis = 0, sort=False)
pp_f = power_freq_pad.fillna(0)
fft_profile_7 = pp_f.transpose()
fft7= pd.DataFrame(fft_profile_0)
fft7['Q'] = q_factor[7]
fft7['device'] = 'wbg'
fft7['asym'] = s7
fft7['num_peaks'] = 1

freq_axis = np.fft.fftfreq(dfa8['x'].count())
power = np.fft.fft(dfa8['y']).real
power_freq = pd.DataFrame(power[:180], columns=['power'])
plt.plot(power_freq)
pad = np.arange(35,181,1)
power_freq_pad = pd.concat([power_freq[:35], pd.DataFrame(index = pad)], axis = 0, sort=False)
ppf = power_freq_pad.fillna(0)
fft_profile_8 = ppf.transpose()
fft8 = pd.DataFrame(fft_profile_0)
fft8['Q'] = q_factor[8]
fft8['device'] = 'wbg'
fft8['asym'] = s8
fft8['num_peaks'] = 1

fft0.to_csv('fft'+filenames[0])
fft1.to_csv('fft'+filenames[1])
fft2.to_csv('fft'+filenames[2])
fft3.to_csv('fft'+filenames[3])
fft4.to_csv('fft'+filenames[4])
fft5.to_csv('fft'+filenames[5])
fft6.to_csv('fft'+filenames[6])
fft7.to_csv('fft'+filenames[7])
fft8.to_csv('fft'+filenames[8])



#ls = list(zip(filenames, q_factor, fft))


#
#
#fft_prof = pd.DataFrame({'fft':filenames, '' })
#
#    
#
#
#
#
#    freq_axis = np.fft.fftfreq(df['x_scale'].count())
#    power = np.fft.fft(df['y_scale']).real
#    power_freq = pd.DataFrame(power[:180], columns=['power'])
##    plt.plot(freq_axis[:180], power_freq)
##    plt.show()
##    plt.xlim(40,180 )
##    plt.ylim(-.1,1)
#    
#    fft_profile = power_freq.transpose()
#    fft_profile['Q'] = center_wavelength/FWHM
#    fft_profile['device'] = 'wbg'
#    fft_profile['asym'] = skewness
#    fft_profile['num_peaks'] = 1
##    fft_decomp.append(fft_profile)
##    fft_profile.to_csv('fft'+fname)
#    
#    
#
#    
##df_q = pd.DataFrame({'filnames':file_names,'fft':fft_p ,'quality_factor':Q, 'skew':asym, 'number_of_peaks': number_of_peaks})
##df_q.to_csv('peak_characteristics')