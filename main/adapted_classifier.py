import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import cv2
import IPython.display as ipd
import ipywidgets as widgets

def classify(filename, start, stop, window_length=10240, hop_length=2560):
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

    # Import audio -----------------------------------------------------------------------------------------------------
    sampling_rate = 24000

    start = start * sampling_rate
    stop = stop * sampling_rate if stop > 0 else -1
    
    #audio
    audio, sampling_rate = sf.read(filename, start=start, stop=stop)

    lim = math.floor(len(audio) / window_length) * window_length    # ensures full "framization" of sequence
    channels = {
        "mic": audio[:lim, 0],
        "female": audio[:lim, 1],
        "male": audio[:lim, 2]
    }

    mic = channels["mic"]
    female = channels["female"]
    male = channels["male"]
    time = np.arange(len(mic)) / sampling_rate    # in seconds

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

    threshold_silence = 0.005
    threshold_signal_lower = 0.07
    threshold_signal_upper = 0.3
    for i, value in enumerate(energy):
        if value < threshold_silence:
            labels[i] = "silence"
        elif threshold_signal_lower < value < threshold_signal_upper:
            labels[i] = "signal"
        else:
            labels[i] = "not classified"

    samples = {}    # samples[index of audio file] = corresponding class
    for label in ["not classified", "silence", "signal"]:
        for i, frame in enumerate(frames):
            if labels[i] == label:
                for t in range(frame[0], frame[1]):
                    samples[t] = label

    # Reorder frames (connect overlapping frames) ----------------------------------------------------------------------
    #list of indices from t
    #2 sec = 20k indices. in the array the indices of the samples which are labeled as either silence or signal (200:500)
    #i get start and end values of these samples. 
    
    signal = []
    
    frame = []
    for t in sorted(samples.keys()):
        label = samples[t]
        if label == "signal":
            frame.append(t)
        else:
            signal.append(frame) if len(frame) != 0 else None
            frame = []


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

        S_f = librosa.feature.melspectrogram(y=np.asfortranarray(f), sr=sr, n_mels=64, fmin=0, fmax=3000)
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
    
    for frame in signal:
        t = np.asarray(frame) #y-mic
            
        y = [mic[x] for x in t] #take  the timepoints from mic that were labeled as signal
        y_f = [female[x] for x in t] #take the timepoints from female that were labeled as signal in mic
        y_m = [male[x] for x in t] #take the timepoints from male that were labeled as signal in mic
        
        if not radio(y_m, y_f): # make sure has no noise
            
            m_bool, f_bool, m_, f_ = vocalizer(y_m, y_f) #m_ = melspectrogram of y_m; f_ = melspectrogram of y_f
            
            #create spectogram from the microphone data
            y_ = librosa.feature.melspectrogram(y=np.asfortranarray(y), sr=sr, n_mels=64, fmin=300, fmax=2000)
            
            if m_bool and f_bool:
                S_clean.append([y_, m_, f_]) # both 
            elif m_bool:
                S_trivial_m.append([y_, m_]) # only male
            elif f_bool:
                S_trivial_f.append([y_, f_]) # only female
            else:
                None    # none

    return S_trivial_m, S_trivial_f, S_clean
#male vocalizes and female is silent = s_trivial m. 
#female vocalizes and male is silent = s_trivial f
#both vocalize at he same time = s_clean

