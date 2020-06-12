import pandas as pd
import numpy as np
import soundfile as sf

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# load dataframe with file locations
data_files = pd.read_csv('../data_files.csv', index_col='rec_id')

# check out recording no10 as an example
rec_no = 10
rec_files = data_files.loc[rec_no,:]

# read in all the audio files into `sfs`, save the corresbonding label in `index`
# print a basic summary for each file
index = []
audios = []
samples_rates = []
y_data = []
for i, f in enumerate(rec_files.values):
    if f.endswith('.w64'):
        audio, sr = sf.read(f)
        
        audios.append(audio)
        samples_rates.append(sr)
        # each frame/sample (element in the np.array `audio`) is divided by the sample rate in Hz
        # to norm the y axis to seconds
        y_data.append(np.arange(len(audio))/sr) 
        index.append(rec_files.index[i])
        
        info = '{}:\n-----------------\n{}\nduration: {:.2f} minutes\n\n'.format(
                index[-1], sf.info(f), y_data[0][-1]/60)
        print(info)
# read in the channel list csv 
channel_list = pd.read_csv(rec_files.SdrChannelList, index_col='channel_number')
# Recording/ Data Format:
"""
Number of recordings: 32



Fileformat: 
-------------------------
bird1           bird2           no_rec      filetype
------------------------
b8p2male    -   b10o15female    _10_           *



Filetypes:
-------------------------
filetype                    description                                         sampling rate   #channels
-------------------------
SdrChannels.w64	            The demodulated SDR channels.                       r_lf            3
SdrCarrierFreq.w64          The estimated carrier frequencies f_c for           r_tr            3
                            all SDR channels                            
SdrReceiveFreq.w64	        The mixing frequencies f_r for all SDR channels.    r_tr            3
SdrSignalStrength.w64	    The signal's power spectral density at f_c for      r_tr            3
                            all SDR channels.
DAQmxChannels.w64	        The NI DAQ channels.                                r_daq           1
SdrChannelList.csv          The SDR channel list as configured in the GUI.      -
prefix>_log.txt             Overflow events and runtime errors are logged 
                            in here.
"""

# summary plots
def amplitudes():
    colors=['#e6194B', '#3cb44b', '#f58231', ]
    fig, axes = plt.subplots(figsize=(12,7), nrows=3, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=.15, right=.96, top=.83, left=.16)

    # the amplidute plots (excluding SdrCarrierFreq & SdrReceiveFreq)
    for i, ax in zip((0,2,4), axes):
        ax.set_prop_cycle(color=colors)
        ax.set_facecolor('#ededed')
        ax.yaxis.grid(color='w', linewidth=2, alpha=.6)

        ax.set_ylabel(index[i], rotation='horizontal', size=13, labelpad=65)
        ax.tick_params('y', labelleft=False, left=False, right=True, labelright=True)

        # first
        if not i:
            ax.set_title('Amplitude plot', size=16, pad=15)
        # last
        elif i == 4:
            ax.set_xlabel('time in [seconds]', size=13)
            labels = ['microphone', 'backpack1', 'backpack2']
            patches = [Patch(color=colors[i], label=labels[i])
                    for i in range(3)]
            fig.legend(handles=patches, loc='upper right', ncol=1, fontsize=13)
        ax.plot(y_data[i], audios[i], alpha=.7)
    fig.savefig('../'+str(rec_no)+'_amplitudes.png')


def frequencies():
    # init figure
    colors=['#e6194B', '#3cb44b', '#f58231', ]
    fig, axes = plt.subplots(figsize=(12,5), nrows=2, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=.15, right=.88, top=.79, left=.16)

    # SdrCarrierFreq & SdrReceiveFreq frequencies
    for i, ax in zip((1,3), axes):
        ax.set_prop_cycle(color=colors)
        ax.set_facecolor('#ededed')
        ax.yaxis.grid(color='w', linewidth=2, alpha=.6)

        ax.set_ylabel(index[i], rotation='horizontal', size=13, labelpad=65)
        ax.tick_params('y', labelleft=False, left=False, right=True, labelright=True)

        # first
        if i == 1:
            ax.set_title('Frequency plot', size=16, pad=15)
        # last
        elif i == 3:
            ax.set_xlabel('time in [seconds]', size=13)
            labels = ['microphone', 'backpack1', 'backpack2']
            patches = [Patch(color=colors[i], label=labels[i])
                    for i in range(3)]
            fig.legend(handles=patches, loc='upper right', ncol=1, fontsize=13)

        ax.plot(y_data[i], audios[i])
        lbls = [str(int(lbl/1000))+' kHz' for lbl in ax.get_yticks()]
        ax.set_yticklabels(lbls)
    fig.savefig('../'+str(rec_no)+'_frequencies.png')

def spectrogram():
    # sizes of indiviual plots (in ratios of 1)
    ratio = {'width_ratios': [.8],
            'height_ratios': [.12, .06, .12, .12, .12, .10, .12, .12, .12]}
    # init figure
    fig, axes = plt.subplots(figsize=(12,10), nrows=9, ncols=1, sharex=True, sharey=False, gridspec_kw=ratio)
    fig.subplots_adjust(hspace=0, right=.96, top=.9, left=.16)

    # iterate the 3 audio data files: SdrChannels, SdrSignalStrength, DAQmxChannels
    which_ax = 0
    for i in [0,2,4]:
        # expand array with only one channel (DAQmx) to enable columns iteration 
        if audios[i].ndim == 1:
            audios[i] = np.expand_dims(audios[i], axis=1)

        # iterate channels mic, backpack1 ,backpack2
        for channel_lbl, channel in zip(channel_list.bird_name.values, audios[i].T):
            # set current axis
            ax = axes[which_ax]
            # set axis 1 and 5 as spacers, ie. set invisible 
            if which_ax in [1, 5]:
                ax.set_visible(False)
                which_ax += 1
                ax = axes[which_ax]

            # draw spectrogram, set range for DAQmx & SdrChanels different to the one of SdrSignalStrength
            ampl_range = [-120, 0] if i in (0, 2) else [-25,25]
            spec, freqs, t, im = axes[which_ax].specgram(channel, Fs=samples_rates[i], alpha=.7, cmap='jet',  
                                                    vmin=ampl_range[0], vmax=ampl_range[1])
            # setup y axis labels, tick parameters
            ax.tick_params(labelleft=False, left=False, right=True, labelright=True, labelbottom=False)
            if which_ax in [0, 2, 6]:
                ax.set_title(index[i], loc='left', pad=4)
            ax.set_ylabel(channel_lbl, rotation='horizontal', size=11, ha='right')
            # the SdrSignalStrength is on a very different scale. Maybe map signal strengths from CSV to values here? 
            # for now don't change the y axis labeling for the last 3 plots
            if which_ax < 6:
                yticks = [2000, 4000, 6000, 8000, 10000]
                ytick_lbls = [str(int(yt/1000)) + 'kHz' for yt in yticks]
                ax.set_yticks(yticks)
                ax.set_yticklabels(ytick_lbls, size=8)
            # setup x axis
            ax.set_xlim(0, y_data[i][-1])

            # first plot: draw the colorbar
            if not i:
                fig.suptitle('Spectrogram (amplitude per frequency plot)', size=16)
                at = (0.75, .94, .2, .015)
                cb = ax.figure.colorbar(im, cax=fig.add_axes(at), alpha =.3,
                                orientation='horizontal')
                cb.set_label('DAQmx & SdrChanels Amplitude(?)')
                cb.ax.get_xaxis().set_label_position('top')
            
            # last plot: set xaxis labels and draw seconds colorbar
            elif which_ax == 8:
                ax.set_xlabel('time in [seconds]', size=13)
                ax.tick_params(labelbottom=True)
            
                at = (0.75, .42, .2, .015)
                cb = ax.figure.colorbar(im, cax=fig.add_axes(at), alpha =.3,
                                orientation='horizontal')
                cb.set_label('SdrSignalSrength Amplitude(?)')
                cb.ax.get_xaxis().set_label_position('top')
            which_ax += 1
    fig.savefig('../'+str(rec_no)+'_spectrogram.png')

# save the plots in the current directory
amplitudes()
frequencies()
spectrogram()
