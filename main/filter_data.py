"""
2. Generate filtered versions of each recording by filtering by amplitude.
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

spacer = '------------------------------------'
output = '/media/loaloa/Samsung_T5/filt_birdsong_data'

def filter_audio(threshold, pad, chunk, save_wav_file=False):
    """
    Iterate through recordings and filter the DAQmx data according to the 
    passed hyperparameters `thr, and `pad`. `thr` defines the minimally 
    required amplitude, `pad` the seconds to include after and before a 
    sufficiently high audio frame. This makes the audio file still interpretable 
    (listening to it) by providing context. The audio frames that pass the 
    threshold form as mask that is used to filter the Sdr data. Save the mic and 
    sdr data in as .npy files in `output`/. Optionally, also save a .wav file 
    of the DAQmx data.
    """
    # load in the recording (daq and sdr) using soundfile package
    print(f'Processing recording {rec_id}: ', end='')
    data = data_files.loc[rec_id, ['DAQmx', 'SdrChannels']]
    audio = np.array(sf.read(data['DAQmx'])[0])
    sdr = np.array(sf.read(data['SdrChannels'])[0])
    print(f'Length: {audio.shape[0]/(32000*60):.2f} min')

    # tedious to fix... recordings 20, 22, 23, 24, 25, 28 differ by a consistent 
    # number of sample points (in the length) of mic and sdr data:
    mic_length = len(audio)/32000
    sdr_length = len(sdr)/24000
    if mic_length != sdr_length:
        print('Mic and Sdr data differ in length. Dropping last 17 values from '
              'mic data to fit Sdr shape.')
        # 4/3 because of the samplerate difference, 13 is consistent somehow, 
        # seems recording software related
        audio = audio[:-int(13*(4/3))]

    # filter audio data for high amplitudes     
    print('Filtering... ', end='')
    highs_idx = np.argwhere(np.logical_or(audio >=threshold, audio <= -threshold))[:,0]
    if highs_idx.any():
        pad *= 32000
        
        # unreadable, but fast: iterate all highs, make a list of `padded` 
        # arrays each corresponding to a specifc high
                               # lower bound, pad in negative direction
        index_ext = [np.arange(start=highs_idx[i]-pad,      
                               # full-length pad in positive direction if this doesn't reach into the next high, else up to next high
                               stop=highs_idx[i]+pad if highs_idx[i+1]-pad > highs_idx[i]+pad else highs_idx[i+1]-pad, dtype=int) 
                     for i in np.arange(len(highs_idx)-1)]
        # the last high was excluded in the iteration, add here
        index_ext.append(np.arange(highs_idx[-1], highs_idx[-1]+pad, dtype=int))
        # glue together, drop overlapping padding using np.unique
        index_ext = np.unique(np.concatenate(index_ext))
        # the positive padding may overshoot the last audio data -> remove here
        index_ext = index_ext[index_ext<len(audio)]

        # slice the audio to the the indices from above
        audio_mask = np.zeros(audio.shape, dtype=bool)
        audio_mask[index_ext] = True
        audio_sl = audio[audio_mask]
        # sdr data has sample rate of 24000, drop every 4. element from the
        # audio mask (sr=32000) to convert to sr=24000
        sdr_mask = np.delete(audio_mask, np.arange(0, audio_mask.size, 4))
        sdr_sl = sdr[sdr_mask]
        print(f'Done:\t\t Length: {audio_sl.shape[0]/(32000*60):.2f} min')

    else:
        sdr_sl = audio_sl = np.array([])
        print(f'Done:\t\t Length: 00.00 min (empty)')

    np.save(f'{output}/DAQmx_{rec_id:0>2d}_{this_thr}', audio_sl)
    np.save(f'{output}/Sdr_{rec_id:0>2d}_{this_thr}', sdr_sl)
    if save_wav_file:
        sf.write(f'{output}/audio_{rec_id:0>2d}_{this_thr}.wav', audio_sl, 32000)
    return audio_sl, sdr_sl

def make_filtered_data_files_csv():
    """
    Save a file-lookup .csv equivalent to `data_files.csv` as 
    `./filt_data_files_min.amp.*_pad.sec.*.csv. Asterisks are 
    replaced by thresholds and pads used.
    """
    # get all the files in the output dir that match the current hyperparamters
    files = glob(f'{output}/*{this_thr}*')
    index = [f[-28:-26] for f in files if 'audio' in f]

    # construct a file location lookup dataframe
    cols = ['filt_DAQmx', 'filt_SdrChannels', 'filt_DAQmxAudio']
    data = dict()
    data[cols[0]] = [f for f in files if 'DAQmx' in f] # mic data
    data[cols[1]] = [f for f in files if 'Sdr' in f] # Sdr data
    data[cols[2]] = [f for f in files if 'audio' in f] # audio

    data = pd.DataFrame(data, index=pd.Index(index, name='rec_id'))
    fname = f'{output}/filt_data_files_{this_thr}.csv'
    data.to_csv(fname)
    print(f'\n{fname} - created and saved successfully.')

# load dataframe with file locations
data_files = pd.read_csv('../data_files.csv', index_col='rec_id')

thr = .05 # minimum amplitude
pad = 0.5 # seconds to include before and after an above-threshold amplitude
this_thr = f'MinAmp{thr:.2f}_PadSec{(pad):.2f}'
print(f'Hyperparamters: {this_thr}')

# iterate all recordings, call filter function to write filtered data file
for i, rec_id in enumerate(data_files.index.values):
    print(f'\n{i+1}/{data_files.shape[0]} ...')
    audio_sl, sdr_sl = filter_audio(thr, pad, rec_id, save_wav_file=True)
# write file lookup csv
make_filtered_data_files_csv()