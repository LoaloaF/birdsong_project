import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from scipy import fft
from scipy.fftpack import dct
import math
import librosa, librosa.display
import scipy.io.wavfile



from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.manifold import TSNE



def frequency_sepectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = np.fft.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)


"""
#matplotlib inline
rcParams('figure.figsize') = 5, 4
sb.set_style 'whitegrid'
"""

"""
Obtaining Paths
"""
path_to_data = 'F:/Neural Systems Project/Birdstuff/b8p2male-b10o15female_aligned'
days = [14, 15, 16, 18, 19]

#Choose which file type you want to extract
wanted_file = 'DAQmxChannels'

#Make list of all the file paths for later use
file_paths = []
for d in days:
    path_to_data = 'F:/Neural Systems Project/Birdstuff/b8p2male-b10o15female_aligned'
    path_to_data = path_to_data + '/2018-08-{}'.format(d)
    for f in os.listdir(path_to_data):
        if f.rfind(wanted_file) != -1:
            file_paths.append(path_to_data + '/' + f)

#print(np.asarray(file_paths))

"""
Extract Audio and Sampling Rate
"""
#start_point = 16
#end_point = 19
start_point = 980
end_point = 1010
audio, sr = sf.read('F:/Neural Systems Project/Birdstuff/b8p2male-b10o15female_aligned/2018-08-14/b8p2male-b10o15female_9_DAQmxChannels.w64')
#audio, sr = sf.read('F:/Neural Systems Project/Birdstuff/b8p2male-b10o15female_aligned/2018-08-14/b8p2male-b10o15female_9_SdrChannels.w64')
audio = audio[int(start_point*sr):int(end_point * sr)]

print('Sampling Rate: ', sr)
print('Audio shape: ', audio.shape)

"""
Simple Audio and Frequency Plot
"""
t = np.arange(len(audio)) / float(sr)
plt.subplot(2, 1, 1)
plt.plot(t, audio)
plt.xlabel('t')
plt.ylabel('y')
"""
frq, X = frequency_sepectrum(audio, sr)

plt.subplot(2, 1, 2)
plt.plot(frq, X, 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')
plt.tight_layout()
"""
plt.show()


"""
MFCC and Filter Bank Features
"""
"""
#pre emphasis audio signal
pre_emphasis = 0.97
emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
print(emphasized_audio.shape)
#produce frames
frame_size = 0.025
frame_stride = 0.01
frame_length, frame_step = frame_size * sr, frame_stride * sr  # Convert from seconds to samples
signal_length = len(emphasized_audio)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
num_frames=num_frames+3
print(num_frames)
pad_signal_length = num_frames * frame_step + frame_length
print(pad_signal_length)
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_audio, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]
#Apply window
frames *= np.hamming(frame_length)
#Fourier Transform
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
#Filter Bank
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sr)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB
#MFCC
num_ceps = 20
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

#sinusoidal filtering

#(nframes, ncoeff) = mfcc.shape
#n = np.arange(ncoeff)
#lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
#mfcc *= lift  #*

#Mean normalisation
filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

print("MFCC shape = ",mfcc.shape)
print("Filter_banks = ",filter_banks.shape)

plt.imshow(mfcc.T, aspect='auto', origin='lower', interpolation='none', cmap=plt.cm.jet)
plt.title("MFCCs in frames")
plt.show()
plt.imshow(filter_banks.T, aspect='auto', origin='lower', interpolation='none', cmap=plt.cm.jet)
plt.title("Filter_banks in frames")
plt.show()
"""
"""
Librosa
"""

"""
Load Audio
"""
audio_librosa, sr_librosa = librosa.load('F:/Neural Systems Project/Birdstuff/b8p2male-b10o15female_aligned/2018-08-14/b8p2male-b10o15female_9_DAQmxChannels.w64', sr = None)
#audio_librosa, sr_librosa = librosa.load('F:/Neural Systems Project/Birdstuff/b8p2male-b10o15female_aligned/2018-08-14/b8p2male-b10o15female_9_SdrChannels.w64', sr = None)
audio_librosa = audio_librosa[int(start_point*sr_librosa):int(end_point*sr_librosa)]
print(sr_librosa)
print(audio_librosa.shape)

"""
Sopectrum
"""
spectrum = librosa.core.stft(audio_librosa, n_fft=int(0.025*sr_librosa), hop_length=int(0.010*sr_librosa))
absolute_value = np.abs(spectrum)
print(absolute_value.shape)
"""
MFCC
"""
#"""
mfcc_librosa = librosa.feature.mfcc(S=absolute_value, sr=sr_librosa)
print("Librosa_mfcc = ",mfcc_librosa.shape)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_librosa, x_axis='time', cmap=plt.cm.jet)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
#"""
"""
Chroma Feature
"""
chromagram = librosa.feature.chroma_stft(S=absolute_value, sr=sr_librosa)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')
plt.show()
print("Chromogram = ",chromagram.shape)
"""
Roll-off
"""
roll_off=librosa.feature.spectral_rolloff(S=absolute_value, sr=sr_librosa)
print("Roll_off = ", roll_off.shape)
"""
Centroid
"""
centroid=librosa.feature.spectral_centroid(S=absolute_value, sr=sr_librosa)
print("Centroid = ", centroid.shape)
"""
RMS Energy
"""
if(sr_librosa==32000):
    frame_len = 800
elif(sr_librosa==24000):
    frame_len = 600
rms = librosa.feature.rms(S=absolute_value, frame_length=frame_len)
print("RMSE = ", rms.shape)
librosa.display.specshow(rms, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')
plt.show()
"""
Normalize features
"""
mfcc = StandardScaler().fit_transform(mfcc_librosa)
print(np.mean(mfcc), np.std(mfcc))
chromagram = StandardScaler().fit_transform(chromagram)
print(np.mean(chromagram), np.std(chromagram))
"""
Concatinate features
"""
#total_features = np.concatenate((mfcc.T, chromagram, roll_off, centroid), axis=0)
total_features = np.concatenate((mfcc.T, chromagram.T, roll_off.T, centroid.T, rms.T), axis=1)
print(total_features.shape)
"""
"""
#PCA for 2D Cluster Plot
"""
pca_features_db = PCA(n_components=20)
principalcomp_db = pca_features_db.fit_transform(total_features)
print(principalcomp_db.shape)
"""
"""
Finding eps
"""
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(total_features)
distances, indices = nbrs.kneighbors(total_features)
mean_dist = distances.mean()
print(distances)
print(mean_dist)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()

"""
DBScan
"""
model = DBSCAN(eps=81., min_samples=33)
model.fit(total_features)

print(model.labels_)
print(model.labels_.shape)
dbscan_dict = Counter(model.labels_)

"""
Expand Labels to Audio Size
"""
expanded_db_coeff = np.empty(len(audio))
index = 0
for i in range(len(model.labels_)):
    for x in range(int(len(audio)/len(mfcc_librosa.T))-1):
        expanded_db_coeff[index]=model.labels_[i]
        index+=1
print(expanded_db_coeff.shape)

"""
PCA for 2D Cluster Plot
"""
pca_features = PCA(n_components=2)
principalcomp = pca_features.fit_transform(total_features)
print(principalcomp.shape)
"""
"""
#2D Cluster Plot PCA
"""
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis",fontsize=20)
for index in range(len(model.labels_)):
    if(model.labels_[index]==-1):
        plt.scatter(principalcomp[index, 0], principalcomp[index, 1], color='red', s=50)
    else:
        plt.scatter(principalcomp[index, 0], principalcomp[index, 1], color='blue', s=50)
plt.show()
"""
"""
TSNE for 2D Cluster Plot
"""
tsne_features = TSNE(n_components=2)
tsnefit = tsne_features.fit_transform(total_features)
print(tsnefit.shape)
"""
"""
#2D Cluster Plot TSNE
"""
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('TSNE Component - 1',fontsize=20)
plt.ylabel('TSNE Component - 2',fontsize=20)
plt.title("TSNE Component Analysis",fontsize=20)
for index in range(len(model.labels_)):
    if(model.labels_[index]==-1):
        plt.scatter(tsnefit[index, 0], tsnefit[index, 1], color='red', s=50)
    else:
        plt.scatter(tsnefit[index, 0], tsnefit[index, 1], color='blue', s=50)
plt.show()
"""
plt.scatter(principalcomp[:,0], principalcomp[:,1], c=model.labels_, cmap=plt.cm.jet)
plt.show()
plt.scatter(tsnefit[:,0], tsnefit[:,1], c=model.labels_, cmap=plt.cm.jet)
plt.show()

"""
Labeled Audio
"""
plt.scatter(t,audio,c=expanded_db_coeff, linestyle='-', linewidths=0.5)
plt.show()
print(dbscan_dict)
dbscan_keys = list(dbscan_dict.keys())
print(dbscan_keys)

"""
#Audio Snippets for check
"""
for i in range(len(dbscan_keys)):
    value = dbscan_keys[i]
    audio_title = 'audio_snippet_' + str(value) +'.wav'
    print(audio_title)
    audio_snippet = np.empty(len(audio))
    for x in range(len(expanded_db_coeff)):
        if(expanded_db_coeff[x]==value):
            audio_snippet[x] = audio_librosa[x]
        else:
            audio_snippet[x] = 0
    audio_path = 'F:/Neural Systems Project/Code/DBScan/Output/' + audio_title
    librosa.output.write_wav(audio_path, audio_snippet, sr_librosa)


