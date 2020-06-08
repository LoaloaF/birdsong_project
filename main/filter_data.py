import pandas as pd
import numpy as np
import soundfile as sf
import sounddevice as sd

from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path


def filter_audio(threshold, pad, chunk, save_wav_file=False):
    start, stop = chunk
    pad *= sr
    print(f'Processing recordings {start} to {stop}...')

    # iterate over the recordings in chunk
    audio_data = []     # mic data
    sdr_data = []       # sdr data
    for _, data in data_files.iloc[start:stop].loc(1)['DAQmx', 'SdrChannels'].iterrows():
        audio = np.array(sf.read(data['DAQmx'])[0])
        sdr = np.array(sf.read(data['SdrChannels'])[0])
    
        # super tedious to fix... recordings 20, 22, 23, 24, 25, 28 differ in the length of mic and sdr data...
        mic_length = len(audio)/32000
        sdr_length = len(sdr)/24000
        if mic_length != sdr_length:
            print('Mic and Sdr data differ in length. Dropping last 17 values from mic data\n')
            audio = audio[:-int(13*(4/3))]      # 4/3 because of the samplerate difference, 13 is consistent somehow
        audio_data.append(audio)
        sdr_data.append(sdr)

    audio_data = np.concatenate(audio_data)
    sdr_data = np.concatenate(sdr_data)

    # filter audio data for high amplitudes
    highs_idx = np.argwhere(np.logical_or(audio_data >=threshold, audio_data <= -threshold))[:,0]

    # unreadable, but fast: iterate all highs, make a list of `padded` arrays each corresponding to a specifc high
    index_ext = [np.arange(highs_idx[i]-pad,      # lower bound, pad in negative direction
                           # full-length pad in positve direction if this doesn't reach into the next high, else up to next high
                           highs_idx[i]+pad if highs_idx[i+1]-pad > highs_idx[i]+pad else highs_idx[i+1]-pad, dtype=int) 
                 for i in np.arange(len(highs_idx)-1)]
    # the last high was excluded in the iteration, add here
    index_ext.append(np.arange(highs_idx[-1], highs_idx[-1]+pad, dtype=int))
    # glue together, drop duplicates, shift one to left for avoiding outofbounds indexing below
    index_ext = np.unique(np.concatenate(index_ext))
    # the positive padding may overshoot the last audio data -> remove here
    index_ext = index_ext[index_ext<len(audio_data)]

    # slice the audio to the the indices from above
    audio_mask = np.zeros(audio_data.shape, dtype=bool)
    audio_mask[index_ext] = True

    # because the sdr file has a sample rate of 24000, drop every 4 element from the audio mask (sr=32000)
    sdr_mask = np.delete(audio_mask, np.arange(0, audio_mask.size, 4))
    # slice the sdr data to the the indices from above
    sdr_sl = sdr_data[sdr_mask]
    audio_sl = audio_data[audio_mask]

    Path("../data/filtered").mkdir(exist_ok=True)
    np.save(f'../data/filtered/DAQmx_filtered_{start}_{stop}', audio_sl)
    np.save(f'../data/filtered/Sdr_filtered_{start}_{stop}', sdr_sl)
    if save_wav_file:
        sf.write(f'../data/filtered/audio_filtered_{start}_{stop}.wav', audio_sl, sr)

    return audio_sl, sdr_sl

# load dataframe with file locations
data_files = pd.read_csv('../data_files.csv', index_col='rec_id')
sr = 32000      # mic data sample rate

# the recordings interval to glue together into one filtered file
#chunks = [(0, 6), (6, 13), (13, 16), (16, 19), (19, 22), (22, 25), (25, 29), (29, 31)]
chunks = [(0, 6)]

# minimum amplitude
thr = .05
# seconds to include before and after an above-threshold amplitude
pad = .5

# this will tike quite some time and storage. You can also just check out chunk 1, 0-6, which is used in the gui.
for i, chunk in enumerate(chunks):
    print(f'{i+1}/{len(chunks)}:  ', end='')
    audio_sl, sdr_sl = filter_audio(thr, pad, chunk, save_wav_file=True)