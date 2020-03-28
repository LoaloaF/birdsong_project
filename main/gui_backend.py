import pandas as pd
import numpy as np
import soundfile as sf
import datetime

from matplotlib.dates import date2num
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation

import ipywidgets as widgets
from ipywidgets import interact, interact_manual, interactive

import time

# load dataframe with file locations
data_files = pd.read_csv('../data_files.csv', index_col='rec_id')

# `data` serves as a general purpose container for a specfic recording
# the keys refer to the kind of data
# for each datatype, the elements for this key correspond to the 5 files per recording
data_keys = 'index', 'audios', 'x_data', 'sample_rates'
data = {key: [None, None, None, None, None] for key in data_keys}
# sliced versions of the core original data, set in `pick_plot_limits`
data_sliced = {key: [None, None, None, None, None] for key in ['audios_sl', 'x_data_sl']}
# 5th file per recording `channel_list.csv` saved seperately
channel_list = None


def pick_rec_no(interactive=True, rec_no=3):
    def _pick_rec_no(rec_no):
        print('Reading and formatting data... ', end='')

        rec_files = data_files.loc[rec_no,:]
        audio_rec_files = rec_files.where(rec_files.str.contains('.w64')).dropna()
        for i, f in enumerate(audio_rec_files.values):
            audio, sr = sf.read(f)

            # names of filetype
            data['index'][i] = rec_files.index[i]
            # actual data
            data['audios'][i] = audio
            # norm y axis to seconds by dividing by the samplerate
            data['x_data'][i] = np.arange(len(audio))/sr
            # sample rates (int value in Hz)
            data['sample_rates'][i] = sr
        # read in the channel list csv 
        global channel_list
        channel_list = pd.read_csv(rec_files.SdrChannelList, index_col='channel_number')

        print('Done.')
        print('Length of the selected recording `{}`: {} min'.format(rec_no, int(data['x_data'][0][-1]/60)))

    if interactive:
        interact(_pick_rec_no,
                 rec_no=widgets.Dropdown(
                    options=sorted(data_files.index.unique()),
                    description='Recording you want to explore',
                    style={'description_width': 'initial'})
                )
    else:
        _pick_rec_no(rec_no)
    
def pick_plot_limits(interactive=True, start=0, length=15):
    def _pick_plot_limits(start, length):
        start *= 60
        if start+length > data['x_data'][0][-1]:
            print('Length exceeds file length. Change `start` or `length`.')
            return None if interactive else exit()
            
        # iterate over the 5 filetypes, slice each individually, because sample rates differ
        for i in range(5):
            # x_data depends on the sample rate, therefore the index is always recalculated
            from_idx = np.argmax(data['x_data'][i] > start)
            to_idx = np.argmax(data['x_data'][i] >= start+length)
            # slice all audios and x_data to the time interval
            data_sliced['audios_sl'][i] = data['audios'][i][from_idx:to_idx]
            data_sliced['x_data_sl'][i] = data['x_data'][i][from_idx:to_idx]
        print('Start: {:.2f}s - End: {:.2f}s'.format(data_sliced['x_data_sl'][0][0], 
                                                     data_sliced['x_data_sl'][0][-1]))

    if interactive:
        widgets.interact_manual.opts['manual_name'] = 'Update time interval'
        interact_manual(_pick_plot_limits,
                        start=widgets.FloatSlider(
                            value=start,
                            min=0,
                            max=data['x_data'][0][-1]/60,
                            step=0.01,
                            description='Start in minutes',
                            style={'description_width': 'initial'},
                            continuous_update=False), 
                        length=widgets.FloatSlider(
                            value=length,
                            min=.1,
                            max=data['x_data'][0][-1],
                            step=.1,
                            description='Length in seconds',
                            style={'description_width': 'initial'},
                            continuous_update=False)
                        )
    else:
        _pick_plot_limits(start, length)

def spectrogram(interactive=True, vmin=-150, vmax=-20, xunit='minutes'):
    def _spectrogram(vmin, vmax, xunit):
        plt.ioff()
        # sizes of indiviual plots (in ratios of 1)
        ratio = {'width_ratios': [.8],
                'height_ratios': [.12, .06, .12, .12, .12, .10, .12, .12, .12]}
        # init figure
        fig, axes = plt.subplots(figsize=(14,10), nrows=9, ncols=1, sharex=True, 
                                 sharey=False, gridspec_kw=ratio)
        fig.subplots_adjust(hspace=0, right=.96, top=.9, left=.16)

        # iterate the 3 audio data files: SdrChannels, SdrSignalStrength, DAQmxChannels
        which_ax = 0
        for i in [0,2,4]:
            # get the data for the current audio file 
            audio = data_sliced['audios_sl'][i]
            samplerate = data['sample_rates'][i]
            x_data = data_sliced['x_data_sl'][i]
            index = data['index'][i]
            # expand array with only one channel (DAQmx) to enable column iteration 
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=1)

            # iterate channels mic, backpack1 ,backpack2
            for channel_lbl, channel in zip(channel_list.bird_name.values, audio.T):
                # set current axis
                ax = axes[which_ax]
                # set axis 1 and 5 as spacers, ie. set invisible 
                if which_ax in [1, 5]:
                    ax.set_visible(False)
                    which_ax += 1
                    ax = axes[which_ax]

                # setup y axis labels, tick parameters
                ax.tick_params(labelleft=False, left=False, right=True, labelright=True, labelbottom=False)
                if which_ax in [0, 2, 6]:
                    ax.set_title(index, loc='left', pad=4)
                ax.set_ylabel(channel_lbl, rotation='horizontal', size=11, ha='right')
                # the SdrSignalStrength is on a very different scale. Maybe map signal strengths from CSV to values here? 
                # for now don't change the y axis labeling for the last 3 plots
                if which_ax < 6:
                    yticks = [2000, 4000, 6000, 8000, 10000]
                    ytick_lbls = [str(int(yt/1000)) + 'kHz' for yt in yticks]
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(ytick_lbls, size=8)
                
                # draw spectrogram, set range for DAQmx & SdrChanels different to the one of SdrSignalStrength
                if i not in (0, 2):
                    vmin, vmax = [-25,25]
                spec, freqs, t, im = axes[which_ax].specgram(channel, Fs=samplerate, alpha=.9, cmap='jet', scale='dB',
                                                             vmin=vmin, vmax=vmax)

                # first plot: draw the colorbar
                if not i:
                    fig.suptitle('Spectrogram (amplitude per frequency plot)', size=16)
                    at = (0.75, .94, .2, .015)
                    cb = ax.figure.colorbar(im, cax=fig.add_axes(at), alpha =1,
                                    orientation='horizontal')
                    cb.set_label('DAQmx & SdrChanels Amplitude(?)')
                    cb.ax.get_xaxis().set_label_position('top')
                            
                # last plot: set xaxis labels and draw seconds colorbar
                elif which_ax == 8:
                    at = (0.75, .42, .2, .015)
                    cb = ax.figure.colorbar(im, cax=fig.add_axes(at), alpha =.3,
                                    orientation='horizontal')
                    cb.set_label('SdrSignalSrength Amplitude(?)')
                    cb.ax.get_xaxis().set_label_position('top')
                
                    # adjust xaxis
                    ax.set_xlabel('time in [{}]'.format(xunit), size=13)
                    ax.tick_params(labelbottom=True)
                    # get the DAQmx x_data because it has the highest sample rate/ resolution 
                    x_data = data_sliced['x_data_sl'][0]
                    # the spectrogram always starts at 0, therefore the x_data is adjusted to also start at 0
                    x_axis = np.round(x_data - x_data[0], 3)
                    ax.set_xlim(0, x_axis[-1])

                    # if the presented data didn't start at 0, annotate the true start
                    if round(x_data[0], 3) != x_axis[0]:
                        start_lbl = 'True start:\n{:0>2.0f}:{:0>2.0f} min'.format(*divmod(x_data[0], 60))
                        ax.annotate(start_lbl, (0.13,0.05), xycoords='figure fraction', size=12)
                        
                    # convert                    
                    if xunit == 'minutes':
                        lbls = ['{:0>2.0f}:{:0>2.0f}'.format(*divmod(t, 60)) for t in ax.get_xticks()]
                        ax.set_xticklabels(lbls)
                
                which_ax += 1
        print('New spectrogram generated. Run cell below to show!')

    if interactive:
        widgets.interact_manual.opts['manual_name'] = 'Make spectrogram'
        interact_manual(_spectrogram,
                        vmin=widgets.IntSlider(
                            value=vmin,
                            min=-300,
                            max=300,
                            description='Amplitude range: Min',
                            style={'description_width': 'initial'},
                            continuous_update=False), 
                        vmax=widgets.IntSlider(
                            value=vmax,
                            min=-300,
                            max=300,
                            description='Amplitude range: Max',
                            style={'description_width': 'initial'},
                            continuous_update=False),
                        xunit=widgets.RadioButtons(
                              options=['seconds', 'minutes'],
                              value=xunit,
                              style={'description_width': 'initial'},
                              description='X axis unit',
                              disabled=False)
                        )
    else:
        _spectrogram(vmin=vmin, vmax=vmax, xunit=xunit)

def animate_spectrogram(resolution):
    fig = plt.gcf()
    axes = fig.axes
    if not axes:
        raise Exception('No spectrogram generated. Press button above.')

    # animation
    lines = [axes[i].axvline(0, -.1, 1.1, color='#c20000', clip_on=False) for i in (0,2,3,4)]
    # empirical rate that kinda works: comes somewhat close to real time
    x_dwnsampled = data_sliced['x_data_sl'][0][::resolution]
    x_dwnsampled -= x_dwnsampled[0] # adjusting to x axis that was normed to start at 0
    x_dwnsampled = x_dwnsampled[1:] # skip the first frame (0)

    animation_frame = lambda i: [line.set_xdata(i) for line in lines]
    return fig, x_dwnsampled, animation_frame

if __name__ == "__main__":
    pick_rec_no(interactive=False)
    pick_plot_limits(interactive=False)
    spectrogram(interactive=False,xunit='minutes')

    fig, x_dwnsampled, animation_frame = animate_spectrogram(resolution=8000)
    an = FuncAnimation(fig, func=animation_frame, frames=x_dwnsampled, interval=1, repeat=False)
    plt.show()
