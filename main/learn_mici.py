# by josua graf

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from config import output
from sklearn.decomposition import FastICA
import pickle
from adapted_classifier_visualized2 import classify as classify_vis

'''
# creat path to numpy arras created by run
temp,dummy = os.path.split(output)
path = os.path.join(temp,'vectorData')

# load numpy arrays created by run
file = os.path.join(path,'clean_m_x')
with open(file,'rb') as clean_m_x:
    clean_m_x_flat = pickle.load('clean_m_x',encoding='bytes')
'''

'''
#import the filtered data list csv
filt_data_files = pd.read_csv(os.path.join(output,'filt_data_files_MinAmp0.05_PadSec0.50.csv'), index_col='rec_id')
# slice to sdr and DAQmx(to use in future perhaps) by getting rid of the third file, the .wav audio
filt_data_files = filt_data_files.drop('filt_DAQmxAudio', axis=1)


#Put the S_trivial_m, S_trivial_f and S_clean together across recordings
for i, (rec_id, rec_id_files) in enumerate(filt_data_files.iterrows()):
    print(f'\nProcessing recording {rec_id} ({i+1}/{filt_data_files.shape[0]})...')
    daq_file, sdr_file = rec_id_files.values
    if not np.load(daq_file).any().any():
        print('Empty.')
        continue
    male_x, male_y, female_x, female_y, clean_m, clean_f, clean_y_ = classify_vis(sdr_file,
                daq_file, 0, -1,
                show_energy_plot=False, show_framesizes=False, rec_id=rec_id,
                show_vocalization=False)
    print('Done.\n')
    myplot = plt.plot(male_x)
    plt.show(myplot)
    print('OK')
'''


# load filtered file list
filtered_files = pd.read_csv(os.path.join(output,'filt_data_files_MinAmp0.05_PadSec0.50.csv'), index_col='rec_id')

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

# define window size and calculate number of frames
window_size = 10000
no_frames = np.int(np.floor(filt_mic_all.size / window_size))
print('Frames = ', no_frames)

# split long array into frames of approximately window size
mic_frames = np.array_split(filt_mic_all, no_frames)
f_frames = np.array_split(filt_f_all, no_frames)
m_frames = np.array_split(filt_m_all, no_frames)

# fourier transform frames
mic_frames_freq = []
f_frames_freq = []
m_frames_freq = []
for i in range(0, no_frames-1):
    mic_frames_freq.append(np.fft.fft(mic_frames[i]))
    f_frames_freq.append(np.fft.fft(f_frames[i]))
    m_frames_freq.append(np.fft.fft(m_frames[i]))

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

# simple function
reconstruction_freq = g_simple(m_frames_freq,mic_frames_freq)

# retransform
for i in range(0,no_frames):
    reconstruction = np.fft.fft(reconstruction_freq)

# plot
#myPlot = plt.subplot(311)
for i in range(0, no_frames-1):
    myPlot = plt.plot(reconstruction[i])
#myPlot = plt.subplot(312)
#myPlot = plt.plot(filt_f_all)
#myPlot = plt.subplot(313)
#myPlot = plt.plot(female)
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