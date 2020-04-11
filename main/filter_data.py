"""
2. Generate filtered data by mergeing chunks of recordings and filtering them by amplitude.
"""
import pandas as pd
import numpy as np
import soundfile as sf
import sounddevice as sd

from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import os
from pathlib import Path
from glob import glob


def filter_audio(threshold, pad, chunk, save_wav_file=False):
    """
    Merge the recordings passed in chunks to one large recording. Filter the
    the DAQmx data according to the defined parameters `thr, and `pad`. `thr` 
    defines the minimally required amplitude, `pad` the seconds to include after 
    and before a sufficiently high audio frame. This makes the audio file still 
    interpretable (listening to it) by providing context. The audio frames that 
    pass the threshold form as mask that is used to filter the Sdr data.
    Save the mic and sdr data in as .npy files in data/filtered/. Optionally, 
    also save a .wav file of the DAQmx data. 
    """
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
    np.save(f'../data/filtered/DAQmx_{start:0>2d}-{stop:0>2d}_{this_thr}', audio_sl)
    np.save(f'../data/filtered/Sdr_{start:0>2d}-{stop:0>2d}_{this_thr}', sdr_sl)
    if save_wav_file:
        sf.write(f'../data/filtered/audio_{start:0>2d}-{stop:0>2d}_{this_thr}.wav', audio_sl, sr)

    return audio_sl, sdr_sl

def make_filtered_data_files_csv():
    """
    Save a file-lookup .csv equivalent to `data_files.csv` as 
    `./filt_data_files_min.amp.*_pad.sec.*.csv. Asterisks are 
    replaced by thresholds and pads used.
    """
    files = glob(f'../data/filtered/*{this_thr}*')
    index = [f[f.find('_')+1 :f.find('_', f.find('_')+1)] for f in files]
    index = pd.Index(index, name='rec_id').drop_duplicates()
    
    cols = ['DAQmx', 'SdrChannels', 'DAQmxAudio']
    data = dict()
    data[cols[0]] = [f for f in files if 'DAQmx' in f] # mic data
    data[cols[1]] = [f for f in files if 'Sdr' in f] # Sdr data
    data[cols[2]] = [f for f in files if 'audio' in f] # audio

    data = pd.DataFrame(data, index=index)
    fname = f'../filt_data_files_{this_thr}.csv'
    data.to_csv(fname)
    print(f'{fname} - created and saved successfully.')

# load dataframe with file locations
data_files = pd.read_csv('../data_files.csv', index_col='rec_id')
sr = 32000      # mic data sample rate

# the recordings interval to glue together into one filtered file
chunks = [(0, 6), (6,13), (13, 16), (16, 19), (19, 22), (22, 25), (25, 29), (29, 31)]

thr = .05 # minimum amplitude
pad = .5 # seconds to include before and after an above-threshold amplitude
this_thr = f'min.amp.{thr:.2f}_pad.sec.{(pad):.2f}'

# this will tike quite some time and storage. Adjust chunks if necessary
print(f'Used threshold paramers: {this_thr}')
for i, chunk in enumerate(chunks):
    print(f'{i+1}/{len(chunks)}:  ', end='')
    audio_sl, sdr_sl = filter_audio(thr, pad, chunk, save_wav_file=True)

make_filtered_data_files_csv()