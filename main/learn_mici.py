# by josua graf

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from config import output as path_filtered
from sklearn.decomposition import FastICA

# load filtered file list
filtered_files = pd.read_csv(os.path.join(path_filtered,'filt_data_files_MinAmp0.05_PadSec0.50.csv'), index_col='rec_id')

# define sampling rates
sampling_rate_sdr = 24000
sampling_rate_daq = 32000


# load all filtered sdr files
rec_no = filtered_files.index.values
filt_mic = []
filt_f = []
filt_m = []
for i, rec_id in enumerate([0,1,5,6]):    # for faster test
#for i, rec_id in enumerate(filtered_files.index.values):       # for all files
    print(rec_id)
    data = filtered_files.loc[rec_id, ['filt_DAQmx', 'filt_SdrChannels']]
    print(data)
    if np.load(data['filt_SdrChannels']).any():
        filt_mic.append(np.load(data['filt_SdrChannels'])[:,0])
        filt_f.append(np.load(data['filt_SdrChannels'])[:,1])
        filt_m.append(np.load(data['filt_SdrChannels'])[:,2])
    else:
        filt_mic.append([])
        filt_f.append([])
        filt_m.append([])

# reshape filt_mic, filt_f and filt_m to a one dimensional array
filt_mic_all = np.concatenate(filt_mic)
filt_f_all = np.concatenate(filt_f)
filt_m_all = np.concatenate(filt_m)

# define simple function g(acc,mic) = mic - sum(acc_{j!=i})
def g_simple(accj,mic):
    mici = np.subtract(mic,accj)
    return mici

# define Independent component analysis function
def g_ICD(accj,mic):
    ica = FastICA(n_components=2)
    signal = ica.fit_transform([mic,accj])
    mici = signal[1]
    return mici

#female = g_simple(filt_m_all,filt_mic_all)
female = g_ICD(np.add(filt_m_all,filt_f_all),filt_mic_all)
print(np.size(filt_mic_all))
print(np.size(filt_m_all))
print(np.size(female))

# plot
myPlot = plt.subplot(311)
myPlot = plt.plot(filt_mic_all)
myPlot = plt.subplot(312)
myPlot = plt.plot(filt_f_all)
myPlot = plt.subplot(313)
myPlot = plt.plot(female)
plt.show(myPlot)


'''
# load mic (s_clean) and clean female and clean male
S_clean_mic = np.load('S_clean_mic.npy')
S_clean_f = np.load('S_clean_female.npy')
S_clean_m = np.load('S_clean_male.npy')

# print shape of S_clean_i
print(np.shape(S_clean_mic))
print(np.shape(S_clean_f))
print(np.shape(S_clean_m))

# plot imported data
myPlot = plt.subplot(311)
myPlot = plt.plot(filt_mic_all)
myPlot = plt.subplot(312)
#myPlot = plt.plot(filt_f)
myPlot = plt.subplot(313)
#myPlot = plt.plot(filt_m)
plt.show(myPlot)

print('list loaded')

# define simple function g(acc,mic) = mic - sum(acc_{j!=i})
def g_simple(accj,mic):
    mici = mic - accj

# test function g
mic_f = g_simple(S_clean_m,S_clean_mic)
plt.show(plt.plot(mic_f))
'''