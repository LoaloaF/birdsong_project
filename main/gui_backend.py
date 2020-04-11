import pandas as pd
import numpy as np
import soundfile as sf
import sounddevice as sd
import glob

from matplotlib.dates import date2num
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation

import ipywidgets as widgets
from ipywidgets import interact, interact_manual, interactive

# set the filtered data to fetch
thr = .05
pad = .5
this_thr = f'min.amp.{thr:.2f}_pad.sec.{(pad):.2f}'

# load dataframe with file locations, check if filtered data exists
data_files = pd.read_csv('../data_files.csv', index_col='rec_id')
if glob.glob('../filt_data_files_*'):
    filtered_data = pd.read_csv(f'../filt_data_files_{this_thr}.csv', 
                                index_col='rec_id')
    data_files = pd.concat([data_files, filtered_data]) # merge with standard data

# init empty data containers for all the different audio-types
data_keys = 'data', 'data_sl', 'x_ticks', 'x_ticks_sl', 'sample_rate', 'name'
empty_data = dict().fromkeys(data_keys, None)

daq, sdr = empty_data.copy(), empty_data.copy()
sdr_carrier_freq, sdr_receiver_freq = empty_data.copy(), empty_data.copy()
sdr_signal_strength = empty_data.copy()
sdr_channellist = None

def pick_rec_no(interactive=True, rec_no=3):
    """
    Fill the empty data containers defined above with one specific recording
    or a filtered recording chunk generated in `filter_data.py`. As all the 
    other functions in this script, one may call `pick_rec_no` with the 
    `interactive`=False keyword to use it in a script, or as True in a jupyter 
    notebook. 
    """
    def _pick_rec_no(rec_no):
        print('Reading and formatting data... ', end='')

        rec_files = data_files.loc[rec_no,:]
        audio_types = daq, sdr_carrier_freq, None, sdr, sdr_receiver_freq, sdr_signal_strength
        for audio_type, (name, file) in zip(audio_types, rec_files.iteritems()):
            # unfiltered recording selected 
            if isinstance(rec_no, int):
                if name not in ('SdrChannelList', 'DAQmxAudio', 'log'):
                    ampl, sr = sf.read(file)
                    audio_type['data'] = ampl
                    audio_type['sample_rate'] = sr
                    audio_type['x_ticks'] = np.arange(len(ampl))/sr
                    audio_type['name'] = name

                elif name == 'SdrChannelList':
                    global sdr_channellist
                    sdr_channellist = pd.read_csv(file, index_col='channel_number')
            
            # filtered data only populates daq, sdr and sdr_channellist 
            elif name in ('DAQmx', 'SdrChannels'):
                ampl, sr = np.load(file), 32000 if name == 'DAQmx' else 24000
                audio_type['data'] = ampl
                audio_type['sample_rate'] = sr
                audio_type['x_ticks'] = np.arange(len(ampl))/sr
                audio_type['name'] = name

                # simply the first element of the unfiltered recordings
                sdr_channellist = pd.read_csv(data_files['SdrChannelList'].iloc[0], 
                                            index_col='channel_number')
        
        print('Done.\nLength of the selected recording `{}`: {} min'.format(rec_no, int(daq['x_ticks'][-1]/60)))
        if isinstance(rec_no, str):
            print(f'\nFiltered data selected with parameters min_amplitude: {thr}; pad_seconds: {pad}.')
            print('To get differently filtered data edit the gui_backend.py file.')
        
        
    if interactive:
        interact(_pick_rec_no,
                 rec_no=widgets.Dropdown(
                    options=data_files.index.unique(),
                    description='Recording you want to explore',
                    style={'description_width': 'initial'}),
                )
    else:
        _pick_rec_no(rec_no)

def pick_plot_limits(interactive=True, start=0, length=15):
    """
    Slice the recording to a user-selected time interval. Sliced versions
    of `data` (`data_sl`) and `x_ticks`(`x_ticks_sl`) are saved in the 
    all the dictionaries  daq, sdr, sdr_carrier_freq...
    """
    def _pick_plot_limits(start, length):
        start *= 60
        if start+length > daq['x_ticks'][-1]:
            print('Length exceeds file length. Change `start` or `length`.')
            return None if interactive else exit()
            
        for audio_type in (daq, sdr, sdr_carrier_freq, sdr_receiver_freq, sdr_signal_strength):
            if audio_type['data'] is not None:
                # x_ticks depends on the specifc sample rate of the data, thus the index is always recalculated
                from_idx = np.argmax(audio_type['x_ticks'] > start)
                to_idx = np.argmax(audio_type['x_ticks'] >= start+length)
                # slice all audios and x_data to the time interval
                audio_type['data_sl'] = audio_type['data'][from_idx:to_idx]
                audio_type['x_ticks_sl'] = audio_type['x_ticks'][from_idx:to_idx]

        print('Start: {:.2f}s - End: {:.2f}s'.format(daq['x_ticks_sl'][0], 
                                                     daq['x_ticks_sl'][-1]))

    if interactive:
        widgets.interact_manual.opts['manual_name'] = 'Update time interval'
        interact_manual(_pick_plot_limits,
                        start=widgets.FloatSlider(
                            value=start,
                            min=0,
                            max=daq['x_ticks'][-1]/60,
                            step=0.01,
                            description='Start in minutes',
                            style={'description_width': 'initial'},
                            continuous_update=False), 
                        length=widgets.FloatSlider(
                            value=length,
                            min=.1,
                            max=daq['x_ticks'][-1],
                            step=.1,
                            description='Length in seconds',
                            style={'description_width': 'initial'},
                            continuous_update=False)
                        )
    else:
        _pick_plot_limits(start, length)

def spectrogram(interactive=True, vmin=-150, vmax=-20, xunit='minutes'):
    """
    Draw a spectrogram for the selected recording. The produced figure 
    is split into 3 subplots, the amplitude of DAQmx, the spectrogram of
    DAQmx and the spectrogram of the Sdr data (split into 3 channels mic, 
    bird1, bird2) (top to bottom). 
    """
    def _spectrogram(vmin, vmax, xunit):
        plt.ioff()
        # sizes of indiviual plots (in ratios of 1)
        ratio = {'width_ratios': [.8],
                'height_ratios': [.24, .08, .24, .08, .12, .12, .12]}
        # init figure
        fig, axes = plt.subplots(figsize=(14,10), nrows=7, ncols=1, sharex=True, 
                                 sharey=False, gridspec_kw=ratio)
        fig.subplots_adjust(hspace=0, right=.96, top=.9, left=.16)

        for which_ax, ax in enumerate(axes.flatten()):
            if which_ax in [1, 3]:
                ax.set_visible(False)
                continue

            # setup y axis labels, tick parameters
            ax.tick_params(labelleft=False, left=False, right=True, labelright=True, labelbottom=False)
            if which_ax in [0, 2, 4]:
                if which_ax == 0:
                    title = 'DAQmx Amplitude'
                elif which_ax == 2:
                    title = 'DAQmx spectrogram'
                elif which_ax == 4:
                    title = 'Sdr Spectrogram'
                ax.set_title(title, loc='left', pad=4)
            if which_ax in (4,5,6):
                ax.set_ylabel(sdr_channellist.bird_name[which_ax-4], 
                              rotation='horizontal', size=11, ha='right')

            # amplitude plot
            if which_ax == 0:
                ax.set_facecolor('#ededed')
                ax.yaxis.grid(color='w', linewidth=2, alpha=.6)
                x = daq['x_ticks_sl'] -daq['x_ticks_sl'][0] # norm to start at 0
                ax.plot(x, daq['data_sl'], alpha=.7)
                ax.tick_params(left=True, labelleft=True, right=True, labelright=True)
                ax.set_ylabel('[dB]')
            # two spectrograms
            else:
                yticks = [2000, 4000, 6000, 8000, 10000]
                ytick_lbls = [str(int(yt/1000)) + 'kHz' for yt in yticks]
                ax.set_yticks(yticks)
                ax.set_yticklabels(ytick_lbls, size=8)
                # draw spectrogram
                if which_ax == 2:
                    channel, sr = daq['data_sl'], daq['sample_rate']
                else:
                    channel, sr = sdr['data_sl'][:, which_ax-4], sdr['sample_rate']
                _, _, _, im = ax.specgram(channel, Fs=sr, alpha=.9, cmap='jet', scale='dB',
                                          vmin=vmin, vmax=vmax)

            # first plot: draw the colorbar
            if which_ax == 2:
                fig.suptitle('Spectrogram (frequency-densities)', size=16)
                at = (0.75, .94, .2, .015)
                cb = ax.figure.colorbar(im, cax=fig.add_axes(at), alpha =1,
                                orientation='horizontal')
                cb.set_label('Amplitude')
                cb.ax.get_xaxis().set_label_position('top')
                        
            # last plot: set xaxis labels and draw seconds colorbar
            elif which_ax == 6:
                ax.set_xlabel('time in [{}]'.format(xunit), size=13)
                ax.tick_params(labelbottom=True)
                # get the DAQmx x_data because it has the highest sample rate/ resolution 
                # the spectrogram always starts at 0, therefore the x_data is adjusted to also start at 0
                x_axis = np.round(daq['x_ticks_sl'] - daq['x_ticks_sl'][0], 3)
                ax.set_xlim(0, x_axis[-1])

                # if the presented data didn't start at 0, annotate the true start
                if round(daq['x_ticks_sl'][0], 3) != daq['x_ticks_sl'][0]:
                    start_lbl = 'True start:\n{:0>2.0f}:{:0>2.0f} min'.format(*divmod(daq['x_ticks_sl'][0], 60))
                    ax.annotate(start_lbl, (0.13,0.05), xycoords='figure fraction', size=12)
                    
                # convert                    
                if xunit == 'minutes':
                    lbls = ['{:0>2.0f}:{:0>2.0f}'.format(*divmod(t, 60)) for t in ax.get_xticks()]
                    ax.set_xticklabels(lbls)
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
    """
    This function draws 3 vertical red lines onto the plot and returns a.o. 
    a function to update the x position of those lines.  
    """
    fig = plt.gcf()
    axes = fig.axes
    if not axes:
        raise Exception('No spectrogram generated. Press button above.')
    
    # animation
    lines = [axes[ax].axvline(0, -.1, 1.1, color='#c20000', clip_on=False) for ax in [0,2,4,5,6]]
    # empirical rate that kinda works: comes somewhat close to real time
    x_dwnsampled = daq['x_ticks_sl'][::resolution]
    x_dwnsampled -= x_dwnsampled[0] # adjusting to x axis that was normed to start at 0
    x_dwnsampled = x_dwnsampled[1:] # skip the first frame (0)

    animation_frame = lambda i: [line.set_xdata(i) for line in lines]
    return fig, x_dwnsampled, animation_frame

if __name__ == "__main__":
    pick_rec_no(interactive=False, rec_no='00-06')  # select recording
    pick_plot_limits(interactive=False, start=1.15, length=23)  # slice the data
    spectrogram(interactive=False,xunit='minutes')  # draw the spectrogram
    
    # adjust the resolution value if the red line and audio are out of sync
    # increase if the red line is lacking behind the sound/ too slow
    # decrease if the red line is before the sound/ too fast
    fig, x_dwnsampled, animation_frame = animate_spectrogram(resolution=6500)   # draw the lines  
    an = FuncAnimation(fig, func=animation_frame, frames=x_dwnsampled, interval=1, repeat=False) # animate the lines
    sd.play(daq['data_sl'], daq['sample_rate']) # play the sound corresponding to the spectrogram
    
    plt.show()
