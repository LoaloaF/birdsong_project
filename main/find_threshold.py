import pandas as pd
import numpy as np
import soundfile as sf
import sounddevice as sd

from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path

# load dataframe with file locations
data_files = pd.read_csv('../data_files.csv', index_col='rec_id')
sr = 32000
threshold = .04     # minimum amplitude
pad = .5    # seconds to include before and after an above-threshold amplitude
pad *= sr

# split in 3 files
start, stop = 0, 10    # which recordings to aggregate
# start, stop = 10, 20    # which recordings to aggregate
# start, stop = 20, 31    # which recordings to aggregate

def filter_audio(save=False):
    audio_files = data_files.loc(1)['DAQmx'].iloc[start:stop]
    audio = np.concatenate([np.array(sf.read(f)[0]) for f in audio_files])

    # filter audio data for high amplitudes
    highs_idx = np.argwhere(np.logical_or(audio >=threshold, audio <= -threshold))[:,0]

    # unreadable, but fast: iterate all highs, make a list of arrays that pad a specifc high
    index_ext = [np.arange(highs_idx[i]-pad,      # lower bound, pad in negative direction
                           # full-length pad in positve direction if this doesn't reach into the next high, else up to next high
                           highs_idx[i]+pad if highs_idx[i+1]-pad > highs_idx[i]+pad else highs_idx[i+1]-pad, dtype=int) 
                 for i in np.arange(len(highs_idx)-1)]
    # the last high was excluded in the iteration, add here
    index_ext.append(np.arange(highs_idx[-1], highs_idx[-1]+pad, dtype=int))
    # glue together, drop duplicates
    index_ext = np.unique(np.concatenate(index_ext))

    # slice the audio to the the indices from above
    audio_mask = np.empty(audio.shape).astype(bool)
    audio_mask[index_ext] = True
    audio_sl = audio[audio_mask]

    if save:
        Path("../data/filtered").mkdir(exist_ok=True)
        np.save(f'../data/filtered/audio_filtered_{start}_{stop}', audio_sl)
        np.save(f'../data/filtered/audio_filtered_mask_{start}_{stop}', audio_mask)
        sf.write(f'../data/filtered/audio_filtered_{start}_{stop}.wav', audio_sl, sr)

    return audio_sl, audio_mask

def load_filtered_audio():
    audio_sl = np.load(f'../data/filtered/audio_filtered_{start}_{stop}.npy')
    audio_mask = np.load(f'../data/filtered/audio_filtered_mask_{start}_{stop}.npy')
    return audio_sl, audio_mask


def amplitudes():
    colors=['#e6194B', '#3cb44b', '#f58231', ]
    fig, ax = plt.subplots(figsize=(12,7), nrows=1, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=.15, right=.96, top=.83, left=.16)

    # the amplidute plots (excluding SdrCarrierFreq & SdrReceiveFreq)
    ax.set_facecolor('#ededed')
    ax.yaxis.grid(color='w', linewidth=2, alpha=.6)

    ax.set_ylabel('DAQmx', rotation='horizontal', size=13, labelpad=65)
    ax.tick_params('y', labelleft=False, left=False, right=True, labelright=True)

    # first
    ax.set_title('Amplitude plot', size=16, pad=15)

    # last
    ax.set_xlabel('time in [seconds]', size=13)
    labels = ['microphone', 'backpack1', 'backpack2']
    patches = [Patch(color=colors[i], label='mic')
            for i in range(3)]
    fig.legend(handles=patches, loc='upper right', ncol=1, fontsize=13)
    ax.plot(y_data, audio_sl, alpha=.7)
    
    sd.play(audio_sl, sr) 


audio_sl, audio_mask = filter_audio(save=True)

# audio_sl, audio_mask = load_filtered_audio()

y_data = np.arange(len(audio_sl)) /sr

amplitudes()
plt.show()