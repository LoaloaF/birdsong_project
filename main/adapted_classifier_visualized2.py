import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import soundfile as sf
import os

import librosa
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import resample
import cv2
import IPython.display as ipd
import ipywidgets as widgets

def classify(sdr_file, daq_file, start, stop, window_length=10240, hop_length=2560, 
             show_energy_plot=False, show_framesizes=False, show_vocalization=False, rec_id=''):
    """
    Returns frames classified into S_trivial_m, S_trivial_f and S_clean

    Parameter:
    ----------
    :param filename: file URL (xx_SdrChannels)
    :param start: start of audio (in seconds; 0 for minimum)
    :param stop: stop of audio (in seconds; -1 for maximum)
    :param window_length: frame length
    :param hop_length: overlap of frames (ideally: window_length / int)
    :return: (S_trivial_m, S_trivial_f, S_clean), with each of the form: [[y, y_f, y_m],...,[y', y_f', y_m']],
        where y = mic, y_f = female acc, y_m = male_acc are amplitude values of one vocalization frame
    """

    os.makedirs('../audios', exist_ok=True)

    # Import audio -----------------------------------------------------------------------------------------------------
    sampling_rate = 24000

    start = start * sampling_rate
    stop = stop * sampling_rate if stop > 0 else -1
    

    
    
    #audio
    # audio, sampling_rate = sf.read(filename, start=start, stop=stop)
    sdr = np.load(sdr_file)
    daq_raw = np.load(daq_file)
    daq = resample(daq_raw, sdr.shape[0])   # get the DAQmx data from 32k to 24k

    lim = math.floor(len(sdr) / window_length) * window_length    # ensures full "framization" of sequence
    channels = {
        # "mic": sdr[:lim, 0],
        "mic": daq[:lim],
        "female": sdr[:lim, 1],
        "male": sdr[:lim, 2]
    }

    mic = channels["mic"]
    female = channels["female"]
    male = channels["male"]
    time = np.arange(len(mic)) / sampling_rate    # in seconds
    print(f'\tLength of the recording: {int(time[-1])} sec, {len(mic):,} samples')

    # Local Regression (eliminate offset) ------------------------------------------------------------------------------
    signal = []
    length = 10000  # interpolation frame length
    model = LinearRegression()
    transformer = PolynomialFeatures(3)     # polynomial interpolation of order 3
    for i in range(0, len(mic), length):
        t = time[i:i + length]
        X = transformer.fit_transform(t.reshape(-1, 1))
        y = mic[i:i + length]
        model.fit(X, y)
        signal.append(model.predict(X))
    mic = mic - [item for items in signal for item in items]

    # Features ---------------------------------------------------------------------------------------------------------
    energy = 200 * librosa.feature.rms(np.asfortranarray(mic), center=False,
                                       frame_length=window_length, hop_length=hop_length)[0] ** 2

    # Signal detection  ------------------------------------------------------------------------------------------------
    frames = [[i, i + window_length] for i in range(0, len(mic) - window_length + 1, hop_length)]
    labels = ["" for i in range(len(energy))]   # class of each frame
    print(f'\tSplit into {len(frames)} (partially overlapping) frames, window size {window_length/sampling_rate:.2f} sec')

    def plot_energy():
        plt.figure(figsize=(14,6))
        plt.plot([frame[0]/sampling_rate for frame in frames], energy, label='energy')
        plt.plot(time, mic, label='microphone')
        plt.ylabel('Amplitude/ Energy (peaks up to 25)')
        plt.xlabel('seconds')
        plt.ylim((-4,8))
        plt.legend(loc='upper right')
        plt.show()
    if show_energy_plot:
        plot_energy()


    threshold_silence = 0.005
    threshold_signal_lower = 0.07
    threshold_signal_upper = 0.3
    
    n_silence = 0
    n_signal = 0
    n_notclsf = 0
    for i, value in enumerate(energy):
        if value < threshold_silence:
            labels[i] = "silence"
            n_silence += 1
        elif threshold_signal_lower < value < threshold_signal_upper:
            labels[i] = "signal"
            n_signal += 1
        else:
            labels[i] = "not classified"
            n_notclsf += 1

    # proportions 
    n_silence /= len(energy)
    n_signal /= len(energy)
    n_notclsf /= len(energy)
    print(f'\tOf {len(energy)} windows, {n_signal*100:.2f}% ({n_signal*len(energy)} total) classified as `signal`, '
          f'{n_silence*100:.2f}% as `silence` and {n_notclsf*100:.2f}% `not classified`.' )

    samples = {}    # samples[index of audio file] = corresponding class
    for label in ["not classified", "silence", "signal"]:
        for i, frame in enumerate(frames):
            if labels[i] == label:
                for t in range(frame[0], frame[1]):
                    samples[t] = label

    # Reorder frames (connect overlapping frames) ----------------------------------------------------------------------
    # list of indices from t
    # 2 sec = 20k indices. in the array the indices of the samples which are labeled as either silence or signal (200:500)
    # i get start and end values of these samples. 
    
    signal = []
    frame = []
    # iterate from sample 0 to 3.2 million
    for t in sorted(samples.keys()):
        # check if that sample was classified as a `signal`
        label = samples[t]
        if label == "signal":
            # this collects CONSECUTIVE signals into frame
            frame.append(t)
        else:
            # after a sequence of `signal` samples, when smth else is encountered as the next elements,
            # append the built up frame to signal, which is the final container
            signal.append(frame) if len(frame) != 0 else None
            frame = []
    
    print(f'\tMerged overlapping `signal` windows (fixed number of samples (window_'
          f'length), total {int(n_signal*len(energy))}) to {len(signal)} windows '
          '(differing in number of samples)')
    
    def plot_framesize():
        plt.figure()
        plt.hist([len(sig)/24000 for sig in signal], bins=20)
        plt.xlabel('Length of frame in seconds')
        plt.ylabel('Number of frames')
        plt.show()
    if show_framesizes:
        plot_framesize()


    # Clustering -------------------------------------------------------------------------------------------------------
    sr = sampling_rate
    
    def radio(m, f): 
        spec_flatness_f = 100 * np.average(librosa.feature.spectral_flatness(np.asfortranarray(f))) # percentage
        spec_flatness_m = 100 * np.average(librosa.feature.spectral_flatness(np.asfortranarray(m))) # percentage
        
        return True if (spec_flatness_m > 30 or spec_flatness_f > 30) else False # only filters out radio noise
    
    def vocalizer(m, f):
        # mel
        S_m = librosa.feature.melspectrogram(y=np.asfortranarray(m), sr=sr, n_mels=64, fmin=300, fmax=2000)
        m = librosa.power_to_db(S_m, ref=np.max)

        S_f = librosa.feature.melspectrogram(y=np.asfortranarray(f), sr=sr, n_mels=64, fmin=300, fmax=2000)
        f = librosa.power_to_db(S_f, ref=np.max)

        # image analysis
        img_m = np.asarray(m, dtype=np.uint8)
        img_f = np.asarray(f, dtype=np.uint8)

        # yellow analysis (spectral energy in freq interval)
        yellow = np.average([np.average(list) for list in img_m])
        voc_m = 210 < np.average([np.average(list) for list in img_m]) < 250
        voc_f = 210 < np.average([np.average(list) for list in img_f]) < 250

        # spectral complexity analysis
        a, b = np.shape(img_m) # same for img_f
        imgs = [img_m, img_f]
        perimeters = [0, 0]

        for j, img in enumerate(imgs):
            # accentuate complexity
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            ret, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
            img = cv2.GaussianBlur(img, (7,7), 0)
            ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
            img = cv2.Canny(img, 100, 200)

            perimeter = np.count_nonzero(img != 0)
            perimeters[j] = 100 * perimeter / (2 * (a + b))

        voc_m_bool = (perimeters[0] > 90) and voc_m
        voc_f_bool = (perimeters[1] > 90) and voc_f
    
        return voc_m_bool, voc_f_bool, S_m, S_f 
    
    S_trivial_m = []    # only male vocalizes
    S_trivial_f = []    # only female vocalizes
    S_clean = []        # both vocalize
    print('\t----Classifying----')
    
    male_x =[]
    male_y = []
    female_x = []
    female_y = []
    clean_m_x = []
    clean_f_x = []
    clean_y = []

    n_male_voc = 0
    n_fem_voc = 0
    n_both_voc = 0
    for frame in signal:
        t = np.asarray(frame) #y-mic
            
        y = [mic[x] for x in t] #take  the timepoints from mic that were labeled as signal
        y_f = [female[x] for x in t] #take the timepoints from female that were labeled as signal in mic
        y_m = [male[x] for x in t] #take the timepoints from male that were labeled as signal in mic
        
        if not radio(y_m, y_f): # make sure has no noise
            m_bool, f_bool, m_, f_ = vocalizer(y_m, y_f) #m_ = melspectrogram of y_m; f_ = melspectrogram of y_f
            
            y_ = librosa.feature.melspectrogram(y=np.asfortranarray(y), sr=sr, 
                                                n_mels=64, fmin=300, fmax=2000)
            
            if m_bool and f_bool:
                S_clean.append([y_, m_, f_, t]) # both 
                n_both_voc += m_bool and f_bool
                
                t_both = [samples_t for pos_frame in S_clean for samples_t in pos_frame[3]]
                clean_m, _, _, _ = plt.specgram(male[t_both], Fs=24000)
                clean_f, _, _, _ = plt.specgram(female[t_both], Fs=24000)
                clean_y_, _, _, _ = plt.specgram(mic[t_both], Fs=24000)
                plt.close()
                
                clean_m_x.append(clean_m)
                clean_f_x.append(clean_f)
                clean_y.append(clean_y_)
                
            elif m_bool:
                S_trivial_m.append([y_, m_, t]) # only male
                n_male_voc += m_bool
                
                t_male = [samples_t for pos_frame in S_trivial_m for samples_t in pos_frame[2]]
                male_x_, _, _, _ = plt.specgram(male[t_male], Fs=24000)
                male_y_, _, _, _ = plt.specgram(mic[t_male], Fs=24000)
                plt.close()
                
                male_x.append(male_x_)
                male_y.append(male_y_)

                
            elif f_bool:
                S_trivial_f.append([y_, f_, t]) # only female
                n_fem_voc += f_bool
                
                t_fem = [samples_t for pos_frame in S_trivial_f for samples_t in pos_frame[2]]
                female_x_, _, _, _ = plt.specgram(female[t_fem], Fs=24000)
                female_y_, _, _, _ = plt.specgram(mic[t_fem], Fs=24000)
                plt.close()
                
                female_x.append(female_x_)
                female_y.append(female_y_)
                        
    
    print(f'\tOut of {len(signal)} frames, {n_male_voc} male alone vocalizations, '
          f'{n_fem_voc} female alone, {n_both_voc} both.')


#### NEW #####



    return male_x, male_y, female_x, female_y, clean_m_x, clean_f_x, clean_y
#S_trivial_m, S_trivial_f, S_clean






#if any([n_male_voc, n_fem_voc, n_both_voc]):
#    # merge all the sample timepoints that were positively classified 
#    t_fem = [samples_t for pos_frame in S_trivial_f for samples_t in pos_frame[2]]
#    t_male = [samples_t for pos_frame in S_trivial_m for samples_t in pos_frame[2]]
#    t_both = [samples_t for pos_frame in S_clean for samples_t in pos_frame[3]]
#    t_all = np.unique(np.concatenate((t_fem, t_male, t_both)))
#
#    # this slices the daq and sdr data to all the frames/ windows identified
#    mic_energy_filt = mic[t_all]
#    fem_energy_filt = female[t_all]
#    male_energy_filt = male[t_all]
#    energy_filt = np.stack([mic_energy_filt, fem_energy_filt, male_energy_filt])
#    sf.write(f'../audios/{rec_id}_vocalizaiton.wav', mic_energy_filt, 24000)
#
#    # This is for drawing the boundaries around identified vocalization
#    # get the first and last sample index of the classified vocalization, unpack list
#    t_fem_edges = [(pos_frame[2][0], pos_frame[2][-1]) for pos_frame in S_trivial_f]
#    #unpack pairs of start and stop sample time points
#    t_fem_edges = [smpl_idx for edges in t_fem_edges for smpl_idx in edges]         
#    # when we slice the original mic data to t_all, the indexes change as 
#    # well. Here, the new index of each edge is found 
#    t_fem_edges = np.concatenate([np.where(t_all==edge)[0] for edge in t_fem_edges]) /24000
#
#    # repeat for male and clean
#    t_male_edges = [(pos_frame[2][0], pos_frame[2][-1]) for pos_frame in S_trivial_m]
#    t_male_edges = [smpl_idx for edges in t_male_edges for smpl_idx in edges]       
#    t_male_edges = np.concatenate([np.where(t_all==edge)[0] for edge in t_male_edges]) /24000
#
#    t_both_edges = [(pos_frame[3][0], pos_frame[3][-1]) for pos_frame in S_clean]
#    t_both_edges = [smpl_idx for edges in t_both_edges for smpl_idx in edges]
#    t_both_edges = np.concatenate([np.where(t_all==edge)[0] for edge in t_both_edges]) /24000
#    
#    #using matplotlib spectrograms for learning ? Uncomment and return items below
#    male_x, _, _, _ = plt.specgram(male[t_male], Fs=24000)
#    female_x, _, _, _ = plt.specgram(female[t_fem], Fs=24000)
#    male_y, _, _, _ = plt.specgram(mic[t_male], Fs=24000)
#    female_y, _, _, _ = plt.specgram(mic[t_fem], Fs=24000)
#    clean_m_x, _, _, _ = plt.specgram(male[t_both], Fs=24000)
#    clean_f_x, _, _, _ = plt.specgram(female[t_both], Fs=24000)
#    clean_y, _, _, _ = plt.specgram(mic[t_both], Fs=24000)
#    plt.close()
#    
#    if show_vocalization:
#        # draw spectrogram
#        spectrogram(mic, energy_filt, fem_edges=t_fem_edges, male_edges=t_male_edges, 
#                    both_edges=t_both_edges)
#


# copied from gui backend

def spectrogram(mic, energy_filt, fem_edges, male_edges, both_edges, 
                vmin=-150, vmax=-20, xunit='minutes'):
    
    # sizes of indiviual plots (in ratios of 1)
    ratio = {'width_ratios': [.8],
            'height_ratios': [.24, .08, .12, .12, .12, .08, .12, .12, .12]}
    # init figure
    fig, axes = plt.subplots(figsize=(14,10), nrows=9, ncols=1, gridspec_kw=ratio)
    fig.subplots_adjust(hspace=0, right=.96, top=.9, left=.16)

    # iterate axes 0-8
    for which_ax, ax in enumerate(axes.flatten()):
        if which_ax in [1, 5]:
            # negative space between plots
            ax.set_visible(False)
            continue

        # setup y axis labels, tick parameters
        ax.tick_params(labelleft=False, left=False, right=True, labelright=True, labelbottom=False)
        if which_ax in [0, 2, 6]:
            if which_ax == 0:
                title = 'DAQmx original(filtered) specgram=matplotlib)'
            elif which_ax == 2:
                title = 'Sdr+DAQmx merged energy-filtered activity windows -- (matplotlib)'
            elif which_ax == 6:
                title = 'Sdr+DAQmx merged energy-filtered activity windows -- librosa mel-spectrogram'
            ax.set_title(title, loc='left', pad=4)

        # label left y axis
        if which_ax in (2,3,4,  6,7,8):
            if which_ax in (2,6):
                lbl = 'DAQmx'
            elif which_ax in (3,7):
                lbl = 'female_acc'
            elif which_ax in (4,8):
                lbl = 'male_acc'
            ax.set_ylabel(lbl, rotation='horizontal', size=11, ha='right')
        
        #  first 2 spectrograms
        if which_ax in (0, 2,3,4):
            # y ticks 
            yticks = [2000, 4000, 6000, 8000, 10000]
            ytick_lbls = [str(int(yt/1000)) + 'kHz' for yt in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_lbls, size=8)
            
            # first two spectrograms
            if which_ax == 0:
                channel = mic
            elif which_ax in (2,3,4):
                channel = energy_filt[which_ax-2,:]
            _, _, _, im = ax.specgram(channel, Fs=24000, alpha=.9, cmap='jet',
                                        vmin=vmin, vmax=vmax)
        # last spectrogram (librosa)
        elif which_ax in (6,7,8):
            ax.tick_params(labelright=False)
            channel = energy_filt[which_ax-6,:]
            spec = librosa.feature.melspectrogram(y=np.asfortranarray(channel), 
                                    sr=24000, n_mels=64, fmin=300, fmax=2000)
            ax.imshow(spec[::-1], cmap='jet', aspect='auto', vmin=0, vmax=.01)

        # draw boxes around vocalization for bottom 2spectrograms for 2 three stack spectograms
        if which_ax in (3,4,  7,8):
            # assign the specifc edges to draw according to current axes (male/ female)
            if which_ax in (3,7):
                edges = fem_edges
            elif which_ax in (4,8):
                edges = male_edges

            # upper spectrogram stack (matplotlib)
            if which_ax in (3,4):
                height = 12000
                alpha = .4
            # lower spectrogram stack (librosa)
            elif which_ax in (7,8):
                height = 64
                alpha = .15
                # adjust to weird x axis ticks of librosa spectrogram
                len_ratio = spec.shape[1] / (energy_filt.shape[1]/24000)
                edges = edges*len_ratio
                if which_ax == 7:
                    both_edges = both_edges*len_ratio   # only once
            
            # draw boxes for male/female
            boxes = [Rectangle((edges[i], 0), edges[i+1]-edges[i], height, facecolor='k', alpha=alpha, linewidth=3, edgecolor='k', clip_on=False)
                    for i in range(0, len(edges), 2)]
            [ax.add_patch(box) for box in boxes]

            # draw boxes for both (S_clean)
            boxes = [Rectangle((both_edges[i], 0), both_edges[i+1]-both_edges[i], height, facecolor='k', alpha=alpha, linewidth=3, edgecolor='k', clip_on=False)
                    for i in range(0, len(both_edges), 2)]
            [ax.add_patch(box) for box in boxes]
            ax.set_axisbelow(True)

        fig.suptitle('Spectrogram vocalization classification.', size=16)
        if which_ax in (0, 4):
            ax.tick_params(labelbottom=True)
        if which_ax == 8:
            ax.set_xlabel('time in seconds'.format(xunit), size=12)

        if which_ax == 0: 
            x_axis = np.arange(len(mic)) /24000
            ax.set_xlim(0, x_axis[-1])
    plt.show()