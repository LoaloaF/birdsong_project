import numpy as np
import pandas as pd
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

####################################################
# this code is the same than simons code but it only uses recording 22 due to laptop performance limit

def classify(sdr_file, daq_file, start, stop, window_length=10240, hop_length=2560,
             show_energy_plot=False, show_framesizes=False, plot_classification=False, rec_id=''):
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

    sampling_rate = 24000

    start = start * sampling_rate
    stop = stop * sampling_rate if stop > 0 else -1

    #######
    start = 0 * sampling_rate
    stop = -1 * sampling_rate if -1 > 0 else -1

    # audio
    sdr = np.load(sdr_file)
    daq_raw = np.load(daq_file)
    daq = resample(daq_raw, sdr.shape[0])  # get the DAQmx data from 32k to 24k

    lim = math.floor(len(sdr) / window_length) * window_length  # ensures full "framization" of sequence
    channels = {
        # "mic": sdr[:lim, 0],
        "mic": daq[:lim],
        "female": sdr[:lim, 1],
        "male": sdr[:lim, 2]
    }

    mic = channels["mic"]
    female = channels["female"]
    male = channels["male"]
    time = np.arange(len(mic)) / sampling_rate  # in seconds
    print(f'\tLength of the recording: {int(time[-1])} sec, {len(mic):,} samples')

    # Local Regression (eliminate offset) ------------------------------------------------------------------------------
    signal = []
    length = 10000  # interpolation frame length
    model = LinearRegression()
    transformer = PolynomialFeatures(3)  # polynomial interpolation of order 3
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
    labels = ["" for i in range(len(energy))]  # class of each frame
    print(
        f'\tSplit into {len(frames)} (partially overlapping) frames, window size {window_length/sampling_rate:.2f} sec')

    def plot_energy():
        plt.figure(figsize=(14, 6))
        plt.plot([frame[0] / sampling_rate for frame in frames], energy, label='energy')
        plt.plot(time, mic, label='microphone')
        plt.ylabel('Amplitude/ Energy (peaks up to 25)')
        plt.xlabel('seconds')
        plt.ylim((-4, 8))
        plt.legend(loc='upper right')
        plt.show()

    if show_energy_plot:
        plot_energy()

    threshold_silence = 0.005
    threshold_signal_lower = 0.07
    threshold_signal_upper = 0.3

    # label the frames and count for each category
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

    # percentage of label categories
    n_silence /= len(energy)
    n_signal /= len(energy)
    n_notclsf /= len(energy)
    print(f'\tOf {len(energy)} windows, {n_signal*100:.2f}% ({n_signal*len(energy)} total) classified as `signal`, '
          f'{n_silence*100:.2f}% as `silence` and {n_notclsf*100:.2f}% `not classified`.')

    samples = {}  # samples[index of audio file] = corresponding class
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
        plt.hist([len(sig) / 24000 for sig in signal], bins=20)
        plt.xlabel('Length of frame in seconds')
        plt.ylabel('Number of frames')
        plt.show()

    if show_framesizes:
        plot_framesize()

    # Clustering -------------------------------------------------------------------------------------------------------
    sr = sampling_rate

    def radio(m, f):
        spec_flatness_f = 100 * np.average(librosa.feature.spectral_flatness(np.asfortranarray(f)))  # percentage
        spec_flatness_m = 100 * np.average(librosa.feature.spectral_flatness(np.asfortranarray(m)))  # percentage

        return True if (spec_flatness_m > 30 or spec_flatness_f > 30) else False  # only filters out radio noise

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
        a, b = np.shape(img_m)  # same for img_f
        imgs = [img_m, img_f]
        perimeters = [0, 0]

        for j, img in enumerate(imgs):
            # accentuate complexity
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            ret, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
            img = cv2.GaussianBlur(img, (7, 7), 0)
            ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
            img = cv2.Canny(img, 100, 200)

            perimeter = np.count_nonzero(img != 0)
            perimeters[j] = 100 * perimeter / (2 * (a + b))

        voc_m_bool = (perimeters[0] > 90) and voc_m
        voc_f_bool = (perimeters[1] > 90) and voc_f

        return voc_m_bool, voc_f_bool, S_m, S_f

    tstamps_fem = []  # only male vocalizes
    trivial_fem = []
    tstamps_male = []  # only female vocalizes
    trivial_male = []
    tstamps_clean = []  # any vocalize
    clean = []

    # booleanlistmaker
    def boolListMaker(n,bool):
        if bool == True or bool == 1:
            listofbool = [1] * n
        elif bool == False or bool == 0:
            listofbool = [0] * n
        return listofbool

    print('\t----Classifying----')
    eng_filt_all = []
    for frame in signal:
        t = np.asarray(frame)  # y-mic
        eng_filt_all.append(t)

        y = [mic[x] for x in t]  # take  the timepoints from mic that were labeled as signal
        y_f = [female[x] for x in t]  # take the timepoints from female that were labeled as signal in mic
        y_m = [male[x] for x in t]  # take the timepoints from male that were labeled as signal in mic

        if not radio(y_m, y_f):  # make sure has no noise
            m_bool, f_bool, m_, f_ = vocalizer(y_m, y_f)  # m_ = melspectrogram of y_m; f_ = melspectrogram of y_f

            if m_bool or f_bool:       # changed and to or for clean => any bird vocalizes
                tstamps_clean.append(t)  # any
                if m_bool and not f_bool:
                    clean.append({'mic': y, 'acc_m': y_m, 'acc_f': y_f, 'bool_m': boolListMaker(len(y), True), 'bool_f': boolListMaker(len(y), False)})
                elif m_bool and f_bool:
                    clean.append({'mic': y, 'acc_m': y_m, 'acc_f': y_f, 'bool_m': boolListMaker(len(y), True), 'bool_f': boolListMaker(len(y), True)})
                elif f_bool and not m_bool:
                    clean.append({'mic': y, 'acc_m': y_m, 'acc_f': y_f, 'bool_m': boolListMaker(len(y), False), 'bool_f': boolListMaker(len(y), True)})
            if m_bool and not f_bool:
                tstamps_male.append(t)  # only male
                trivial_male.append({'mic': y, 'acc_m': y_m})
            if f_bool and not m_bool:
                tstamps_fem.append(t)  # only female
                trivial_fem.append({'mic': y, 'acc_f': y_f})
    print(f'\tOut of {len(signal)} frames, {len(trivial_male)} male alone vocalizations, '
          f'{len(trivial_fem)} female alone, {len(clean)} both.')

    # draw only if all lists are not empty
    if all([len(trivial_male), len(trivial_fem), len(clean)]):
        # merge all the sample timepoints that were positively classified
        t_all = np.unique(np.concatenate((*tstamps_fem, *tstamps_male, *tstamps_clean)))

        # this slices the daq and sdr data to all the frames/ windows identified
        mic_energy_filt = mic[t_all]
        fem_energy_filt = female[t_all]
        male_energy_filt = male[t_all]
        energy_filt = np.stack([mic_energy_filt, fem_energy_filt, male_energy_filt])
        sf.write(f'../audios/{rec_id}_vocalizaiton.wav', mic_energy_filt, 24000)

        # This is for drawing the boundaries around identified vocalization
        # get the first and last sample index of the classified vocalization, unpack list
        t_fem_edges = [(pos_frame[0], pos_frame[-1]) for pos_frame in tstamps_fem]
        # unpack pairs of start and stop sample time points
        t_fem_edges = [smpl_idx for edges in t_fem_edges for smpl_idx in edges]
        # when we slice the original mic data to t_all, the indexes change as
        # well. Here, the new index of each edge is found
        all_fem_edges = np.concatenate([np.where(t_all == edge)[0] for edge in t_fem_edges]) / 24000

        # repeat for male and clean
        t_male_edges = [(pos_frame[0], pos_frame[-1]) for pos_frame in tstamps_male]
        t_male_edges = [smpl_idx for edges in t_male_edges for smpl_idx in edges]
        all_male_edges = np.concatenate([np.where(t_all == edge)[0] for edge in t_male_edges]) / 24000

        t_both_edges = [(pos_frame[0], pos_frame[-1]) for pos_frame in tstamps_clean]
        t_both_edges = [smpl_idx for edges in t_both_edges for smpl_idx in edges]
        all_both_edges = np.concatenate([np.where(t_all == edge)[0] for edge in t_both_edges]) / 24000

        # same for top plot (edges need different index...)
        t_all = np.concatenate(eng_filt_all)
        sf.write(f'../audios/{rec_id}_vocalizaiton_all.wav', mic[t_all], 24000)
        t_fem_edges = np.concatenate([np.where(t_all == edge)[0] for edge in t_fem_edges]) / 24000
        t_male_edges = np.concatenate([np.where(t_all == edge)[0] for edge in t_male_edges]) / 24000
        t_both_edges = np.concatenate([np.where(t_all == edge)[0] for edge in t_both_edges]) / 24000
        all_all_edges = np.unique(np.concatenate((t_fem_edges, t_male_edges, t_both_edges)))

        if plot_classification:
            # draw spectrogram
            spectrogram(mic[t_all], energy_filt, fem_edges=all_fem_edges, male_edges=all_male_edges,
                        both_edges=all_both_edges, top_plot_edges=all_all_edges, rec_id=rec_id)

    # return male_x, male_y, female_x, female_y, clean_m_x, clean_f_x, clean_y
    return trivial_male, trivial_fem, clean, len(signal)


# copied from gui backend
def spectrogram(mic, energy_filt, fem_edges, male_edges, both_edges, top_plot_edges,
                rec_id, vmin=-150, vmax=-20, xunit='minutes', ):
    # sizes of indiviual plots (in ratios of 1)
    ratio = {'width_ratios': [.8],
             'height_ratios': [.24, .08, .12, .12, .12, .08, .12, .12, .12]}
    # init figure
    fig, axes = plt.subplots(figsize=(14, 10), nrows=9, ncols=1, gridspec_kw=ratio)
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
        if which_ax in (2, 3, 4, 6, 7, 8):
            if which_ax in (2, 6):
                lbl = 'DAQmx'
            elif which_ax in (3, 7):
                lbl = 'female_acc'
            elif which_ax in (4, 8):
                lbl = 'male_acc'
            ax.set_ylabel(lbl, rotation='horizontal', size=11, ha='right')

        #  first 2 spectrograms
        if which_ax in (0, 2, 3, 4):
            # y ticks
            yticks = [2000, 4000, 6000, 8000, 10000]
            ytick_lbls = [str(int(yt / 1000)) + 'kHz' for yt in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_lbls, size=8)

            # first two spectrograms
            if which_ax == 0:
                channel = mic
            elif which_ax in (2, 3, 4):
                channel = energy_filt[which_ax - 2, :]
            _, _, _, im = ax.specgram(channel, Fs=24000, alpha=.9, cmap='jet',
                                      vmin=vmin, vmax=vmax)
        # last spectrogram (librosa)
        elif which_ax in (6, 7, 8):
            ax.tick_params(labelright=False)
            channel = energy_filt[which_ax - 6, :]
            spec = librosa.feature.melspectrogram(y=np.asfortranarray(channel),
                                                  sr=24000, n_mels=64, fmin=300, fmax=2000)
            ax.imshow(spec[::-1], cmap='jet', aspect='auto', vmin=0, vmax=.01)

        # draw boxes around vocalization for bottom 2spectrograms for 2 three stack spectograms
        if which_ax in (0, 3, 4, 7, 8):
            # assign the specifc edges to draw according to current axes (male/ female)
            if which_ax == 0:
                edges = top_plot_edges
            elif which_ax in (3, 7):
                edges = fem_edges
            elif which_ax in (4, 8):
                edges = male_edges

            # upper spectrogram stack (matplotlib)
            if which_ax in (0, 3, 4):
                height = 12000
                alpha = .4
            # lower spectrogram stack (librosa)
            elif which_ax in (7, 8):
                height = 64
                alpha = .15
                # adjust to weird x axis ticks of librosa spectrogram
                len_ratio = spec.shape[1] / (energy_filt.shape[1] / 24000)
                edges = edges * len_ratio
                if which_ax == 7:
                    both_edges = both_edges * len_ratio  # only once

            # draw boxes for male/female
            boxes = [Rectangle((edges[i], 0), edges[i + 1] - edges[i], height, facecolor='k', alpha=alpha, linewidth=3,
                               edgecolor='k', clip_on=False)
                     for i in range(0, len(edges), 2)]
            [ax.add_patch(box) for box in boxes]

            # draw boxes for both (S_clean)
            boxes = [
                Rectangle((both_edges[i], 0), both_edges[i + 1] - both_edges[i], height, facecolor='k', alpha=alpha,
                          linewidth=3, edgecolor='k', clip_on=False)
                for i in range(0, len(both_edges), 2)]
            [ax.add_patch(box) for box in boxes]
            ax.set_axisbelow(True)

        fig.suptitle(f'Rec {rec_id}: Vocalization classification.', size=16)
        if which_ax in (0, 4):
            ax.tick_params(labelbottom=True)
        if which_ax == 8:
            ax.set_xlabel('time in seconds'.format(xunit), size=12)

        if which_ax == 0:
            x_axis = np.arange(len(mic)) / 24000
            ax.set_xlim(0, x_axis[-1])

    fig.savefig(f'{rec_id}_vocalizaiton.png')


#####################################START######################################
# import the filtered data list csv
# filt_data_files = pd.read_csv('/media/loaloa/Samsung_T5/filt_birdsong_data/filt_data_files_MinAmp0.05_PadSec0.50.csv', index_col='rec_id')
filt_data_files = pd.read_csv('C:/Users/Josua Graf/polybox/Neural Systems/b8p2male-b10o15female_aligned(1)/filtered_20200615/filt_data_files_MinAmp0.05_PadSec0.50_rec22.csv',
    index_col='rec_id')
# slice to sdr and DAQmx(to use in future perhaps) by getting rid of the third file, the .wav audio
filt_data_files = filt_data_files.drop('filt_DAQmxAudio', axis=1)

S_trivial_m_all = []
S_trivial_f_all = []
S_clean_all = []
n_vocs_log = dict()

# these are handlabeled signals that are actually clean, eg. no wingflaps
# none were found for female and `both` also don't sound to promising
# male has 45 sec of solid vocalizations
no_flapping_vocs = {
    2: {'male': [3], 'female': [0, 1], 'both': [0]},
    4: {'male': [0], 'female': [], 'both': []},
    6: {'male': [0, 1], 'female': [], 'both': [0]},
    13: {'male': [0, 1, 2, 3], 'female': [], 'both': [0]},
    21: {'male': [32], 'female': [], 'both': []},
    22: {'male': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50], 'female': [],
         'both': [2]},
    24: {'male': [0, 1], 'female': [], 'both': []},
}

train_dat_path = '../train_data/'
os.makedirs(train_dat_path, exist_ok=True)
os.makedirs('../audios/', exist_ok=True)

# Put the S_trivial_m, S_trivial_f and S_clean together across recordings
for i, (rec_id, rec_id_files) in enumerate(filt_data_files.iterrows()):
    print(f'\nProcessing recording {rec_id} ({i+1}/{filt_data_files.shape[0]})...')
    daq_file, sdr_file = rec_id_files.values
    if not np.load(daq_file).any().any():
        print('Empty.')
        continue
    # elif rec_id not in no_flapping_vocs.keys():
    #     print('No no-flapping vocalizations in this rec id.')
    #     continue

    S_trivial_m, S_trivial_f, S_clean, n_frames = classify(sdr_file,
                                                           daq_file,
                                                           0, -1,
                                                           show_energy_plot=False,
                                                           show_framesizes=False,
                                                           rec_id=rec_id,
                                                           plot_classification=True)
    print('Done.\n')
    n_vocs_log[rec_id] = [len(S_trivial_m), len(S_trivial_f), len(S_clean), n_frames]
    '''
    if rec_id not in no_flapping_vocs:
        continue

    if S_trivial_m:  # if S_trivial_m not empty, append it to the list
        S_trivial_m_all.extend(
            [vocs_in_rec for i, vocs_in_rec in enumerate(S_trivial_m) if i in no_flapping_vocs[rec_id]['male']])
    if S_trivial_f:  # if S_trivial_f not empty, append it to the list
        S_trivial_f_all.extend(
            [vocs_in_rec for i, vocs_in_rec in enumerate(S_trivial_f) if i in no_flapping_vocs[rec_id]['female']])
    if S_clean:  # if S_clean not empty, append it to the list
        S_clean_all.extend(
            [vocs_in_rec for i, vocs_in_rec in enumerate(S_clean) if i in no_flapping_vocs[rec_id]['both']])
    '''
    if S_trivial_m:  # if S_trivial_m not empty, append it to the list
        S_trivial_m_all.extend(S_trivial_m)
    if S_trivial_f:  # if S_trivial_f not empty, append it to the list
        S_trivial_f_all.extend(S_trivial_f)
    if S_clean:  # if S_clean not empty, append it to the list
        S_clean_all.extend(S_clean)

# log
df = pd.DataFrame(n_vocs_log, index=['male vocal', 'fem vocal.', 'both vocal.', 'of total frames']).T
df.columns.name = 'rec_id'
df = df.append(pd.Series(df.sum(), name='sum'))
df.to_csv('../n_vocal.csv')
print('\n', df.to_string())

if S_trivial_m_all:
    S_trivial_m_mic = np.concatenate([np.array(vocs_in_rec['mic']) for vocs_in_rec in S_trivial_m_all])
    S_trivial_m_acc = np.concatenate([np.array(vocs_in_rec['acc_m']) for vocs_in_rec in S_trivial_m_all])

    np.save(train_dat_path + 'S_trivial_m_mic.npy', S_trivial_m_mic)
    np.save(train_dat_path + 'S_trivial_m_acc.npy', S_trivial_m_acc)
    sf.write(f'../audios/triv_male_merged.wav', S_trivial_m_mic, 24000)

# important! the frames added here are not valid (no good vocalization in them)
# `no_flapping_vocs` should actually be empty for female because I couldn't find
# any good ones. This is just here so there is at least something to read in
# But its probably useless for learning.
if S_trivial_f_all:
    S_trivial_f_mic = np.concatenate([np.array(vocs_in_rec['mic']) for vocs_in_rec in S_trivial_f_all])
    S_trivial_f_acc = np.concatenate([np.array(vocs_in_rec['acc_f']) for vocs_in_rec in S_trivial_f_all])

    np.save(train_dat_path + 'S_trivial_f_mic.npy', S_trivial_f_mic)
    np.save(train_dat_path + 'S_trivial_f_acc.npy', S_trivial_f_acc)
    sf.write(f'../audios/triv_fem_merged.wav', S_trivial_f_mic, 24000)

if S_clean_all:
    S_clean_mic = np.concatenate([np.array(vocs_in_rec['mic']) for vocs_in_rec in S_clean_all])
    S_clean_m_acc = np.concatenate([np.array(vocs_in_rec['acc_m']) for vocs_in_rec in S_clean_all])
    S_clean_f_acc = np.concatenate([np.array(vocs_in_rec['acc_f']) for vocs_in_rec in S_clean_all])
    S_clean_m_bool = np.concatenate([np.array(vocs_in_rec['bool_m']) for vocs_in_rec in S_clean_all])
    S_clean_f_bool = np.concatenate([np.array(vocs_in_rec['bool_f']) for vocs_in_rec in S_clean_all])

    np.save(train_dat_path + 'S_clean_mic.npy', S_clean_mic)
    np.save(train_dat_path + 'S_clean_m_acc.npy', S_clean_m_acc)
    np.save(train_dat_path + 'S_clean_f_acc.npy', S_clean_f_acc)
    np.save(train_dat_path + 'S_clean_m_bool.npy', S_clean_m_bool)
    np.save(train_dat_path + 'S_clean_f_bool.npy', S_clean_f_bool)
    sf.write(f'../audios/triv_claen_merged.wav', S_clean_mic, 24000)

# read like this:
# S_clean_m_acc = np.load(train_dat_path+'S_clean_m_acc.npy)
# 2 male, 2 female, 3 clean
# ...
# spectrograms like this:
# male_x, _, _, _ = plt.specgram(S_trivial_m_mic, Fs=24000)
# male_y, _, _, _ = plt.specgram(S_clean_m_acc, Fs=24000)
# ....
