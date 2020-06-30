# by josua graf

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from config import output
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
import pickle
from adapted_classifier_visualized2 import classify as classify_vis


# creat path to numpy arrays
path = 'C:/Users/Josua Graf/PycharmProjects/train_data/'

# load numpy arrays created by voc_classific_vis_josua
file = os.path.join(path, 'S_clean_mic.npy')
clean_mic = np.load(file)
print(file, 'loaded')
file = os.path.join(path, 'S_clean_f_acc.npy')
clean_f = np.load(file)
print(file, ' loaded')
file = os.path.join(path, 'S_clean_m_acc.npy')
clean_m = np.load(file)
print(file, ' loaded')
file = os.path.join(path, 'S_clean_f_bool.npy')
clean_f_bool = np.load(file)
print(file, ' loaded')
file = os.path.join(path, 'S_clean_m_bool.npy')
clean_m_bool = np.load(file)
print(file, ' loaded')

# define sampling rate
sampling_rate = 24000
# define time axis
time = np.arange(len(clean_mic)) / sampling_rate  # in seconds

# plot clean imported signal
def plotImportData(i):
    if i == True:
        clean_plot = plt.subplot(511)
        clean_plot = plt.plot(time, clean_mic,c='k')
        clean_plot = plt.title('Clean Mic')
        clean_plot = plt.ylabel('energy')
        clean_plot = plt.subplot(512)
        clean_plot = plt.plot(time, clean_f)
        clean_plot = plt.title('Clean Female Acc')
        clean_plot = plt.ylabel('energy')
        clean_plot = plt.subplot(513)
        clean_plot = plt.plot(time, clean_f_bool)
        clean_plot = plt.title('Boolean Female Voc')
        clean_plot = plt.subplot(514)
        clean_plot = plt.plot(time, clean_m,c='r')
        clean_plot = plt.title('Clean Male Acc')
        clean_plot = plt.ylabel('energy')
        clean_plot = plt.subplot(515)
        clean_plot = plt.plot(time, clean_m_bool,c='r')
        clean_plot = plt.title('Boolean Male Voc')
        clean_plot = plt.xlabel('time in s')
        clean_plot = plt.subplots_adjust(hspace=1.5)
        plt.show(clean_plot)
    return

plotImportData(True)

# extract trivial_m for linear regression
trivial_m_mic = []
trivial_m_acc = []
for i in range(len(clean_mic)):
    if clean_m_bool[i] == 1:
        trivial_m_mic.append(clean_mic[i])
        trivial_m_acc.append(clean_m[i])
    else:
        continue
trivial_m_mic = np.transpose(np.array(trivial_m_mic))
trivial_m_acc = np.transpose(np.array(trivial_m_acc))

# general trivial extraction defind as a function
def extractTrivial(c_mic, c_birdi, c_birdj, c_bool):
    trivial_mic = []
    trivial_birdi = []
    trivial_birdj = []
    for i in range(len(c_mic)):
        if c_bool[i] == 1:
            trivial_mic.append(c_mic[i])
            trivial_birdi.append(c_birdi[i])
            trivial_birdj.append(c_birdj[i])
        else:
            continue
    trivial_mic = np.transpose(np.array(trivial_mic))
    trivial_birdi = np.transpose(np.array(trivial_birdi))
    trivial_birdj = np.transpose(np.array(trivial_birdj))
    return trivial_mic, trivial_birdi, trivial_birdj

# split trivial male into frames
window_size = 2400
no_frames = np.int(np.floor(len(trivial_m_mic)/window_size))
trivial_m_mic_frames = np.array(np.array_split(trivial_m_mic, no_frames))
trivial_m_acc_frames = np.array(np.array_split(trivial_m_acc, no_frames))

# transform frames into spectrograms
trivial_m_mic_frames_freq = []
trivial_m_acc_frames_freq = []
for i in range(no_frames):
    trivial_m_mic_frames_freq.append(np.fft.fft(trivial_m_mic_frames[i]))
    trivial_m_acc_frames_freq.append(np.fft.fft(trivial_m_acc_frames[i]))
trivial_m_mic_frames_freq = np.array(trivial_m_mic_frames_freq)
trivial_m_acc_frames_freq = np.array(trivial_m_acc_frames_freq)

# concatinate
trivial_m_mic_freq = np.transpose([np.real(np.concatenate(trivial_m_mic_frames_freq))])
trivial_m_acc_freq = np.transpose([np.real(np.concatenate(trivial_m_acc_frames_freq))])

# fit linear model
reg = LinearRegression()
reg.fit(trivial_m_acc_freq, trivial_m_mic_freq)
print('reg score', reg.score(trivial_m_acc_freq, trivial_m_mic_freq))
trivial_line_freq = reg.predict(trivial_m_acc_freq)

# plot scatter
def plotLinearRegressionData(x,y,l,x_label, y_label, i):
    if i == True:
        p = plt.scatter(x, y, c='r',marker='x')
        p = plt.plot(x, l, c='b')
        p = plt.xlabel(x_label)
        p = plt.ylabel(y_label)
        p = plt.title('Scatter Plot of Accelerometer male vs. Microphone male trivial')
        plt.show(p)
    return

plotLinearRegressionData(trivial_m_acc_freq, trivial_m_mic_freq, trivial_line_freq, 'Male Acc','Microphone', False)

# prediction of male bird
trivial_m_mic_pre = np.fft.ifft(trivial_line_freq)

def plotPred(mic,acc,pred,i):
    if i == True:
        trivial_time = np.arange(len(mic)) / sampling_rate
        fig = plt.figure()
        fig.suptitle('Trivial Male Microphone Signal Reconstruction')
        p1 = fig.add_subplot(311)
        p1 = plt.plot(trivial_time, mic, c='k')
        p1 = plt.title('Microphone')
        p1 = plt.xlabel('time in s')
        p1 = plt.ylabel('energy')
        p2 = fig.add_subplot(312)
        p2 = plt.plot(trivial_time, acc, c='b')
        p2 = plt.title('Male Acc')
        p2 = plt.xlabel('time in s')
        p2 = plt.ylabel('energy')
        p3 = fig.add_subplot(313)
        p3 = plt.plot(trivial_time, pred, c='r')
        p3 = plt.title('Male Mic Prediction f(male_acc)')
        p3 = plt.xlabel('time in s')
        p3 = plt.ylabel('energy')
        fig.subplots_adjust(hspace=1)
        plt.show(fig)
    return

plotPred(trivial_m_mic, trivial_m_acc, trivial_m_mic_pre, False)

# general cut into frames of approximatly window size and concatinate the fourier transform
def cutFramesFft(mic):
    no_frames = np.int(np.floor(len(mic)/window_size))
    mic_frame =  np.array(np.array_split(mic, no_frames))
    mic_frame_freq = []
    for i in range(no_frames):
        mic_frame_freq.append(np.fft.fft(mic_frame[i]))
    mic_frame_freq = np.array(mic_frame_freq)
    mic_frame_freq = np.transpose([np.real(np.concatenate(mic_frame_freq))])
    return mic_frame_freq

# applay Linear Regression on clean values
clean_m_line_freq = reg.predict(cutFramesFft(clean_m))
clean_m_mic_pre = np.fft.ifft(clean_m_line_freq)
clean_f_line_freq = reg.predict(cutFramesFft(clean_f))
clean_f_mic_pre = np.fft.ifft(clean_f_line_freq)
# do same as a function Function mic_pre = f(acc)
def predictMicrophone(acc):
    line_freq = reg.predict(cutFramesFft(acc))
    mic_pre = np.fft.ifft(line_freq)
    return mic_pre

# define function g simple
def functionGSimple(mic,accj):
    pre = predictMicrophone(accj)
    mici = np.concatenate(mic - np.transpose(pre))
    return mici

#results plots
# a) trivial vs g
def plotTrivialAndG(mic,accj,title1,title2): # accj is the other bird not the one of interest
    time_axis = np.arange(len(mic)) / sampling_rate
    fig = plt.figure()
    p1 = fig.add_subplot(211)
    p1 = plt.plot(time_axis, mic)
    p1 = plt.title(title1)
    p1 = plt.xlabel('time in s')
    p1 = plt.ylabel('energy')
    p2 = fig.add_subplot(212)
    p2 = plt.plot(time_axis, functionGSimple(mic, accj))
    p2 = plt.title(title2)
    p2 = plt.xlabel('time in s')
    p2 = plt.ylabel('energy')
    fig.subplots_adjust(hspace=1)
    plt.show(fig)
    return
def plotTrivialAndGOnePlot(mic,accj,title,label1,label2):
    time_axis = np.arange(len(mic)) / sampling_rate
    fig = plt.figure()
    fig = plt.plot(time_axis, mic,c='k',label=label1)
    fig = plt.title(title)
    fig = plt.xlabel('time in s')
    fig = plt.ylabel('energy')
    fig = plt.plot(time_axis, functionGSimple(mic, accj),c='r',label=label2)
    fig = plt.legend()
    plt.show(fig)
    return

#plot trivial male and  g prediction male
trivial_mic, trivial_m, trivial_m_f = extractTrivial(clean_mic,clean_m,clean_f,clean_m_bool)
#plotTrivialAndG(trivial_mic,trivial_m_f,'Trivial Male Microphone','Male Reconstruction = g(trivial_mic, female for trivial_m)')
#plotTrivialAndGOnePlot(trivial_mic,trivial_m_f,'Trivial Male','Microphone','Reconstructed Male')

#plot trivial female and  g prediction female
trivial_mic, trivial_f, trivial_f_m = extractTrivial(clean_mic,clean_f,clean_m,clean_f_bool)
#plotTrivialAndG(trivial_mic,trivial_f_m,'Trivial Female Microphone','Female Reconstruction = g(trivial_mic, male for trivial_f)')
#plotTrivialAndGOnePlot(trivial_mic,trivial_f_m,'Trivial Female','Microphone','Reconstructed Female')

# b) plot trivial
# for male is silent (trivial_f)
trivial_mic, trivial_f, trivial_f_m = extractTrivial(clean_mic,clean_m,clean_f,clean_f_bool)
#plotTrivialAndG(trivial_mic, trivial_f,'Male Silent Microphone','Male Reconstruction')
#plotTrivialAndGOnePlot(trivial_mic,trivial_f,'Trivial Female / Male Silent','Microphone','Reconstructed Male')

# for female is silent (trivial_m)
trivial_mic, trivial_m, trivial_m_f = extractTrivial(clean_mic,clean_f,clean_m,clean_m_bool)
#plotTrivialAndGOnePlot(trivial_mic,trivial_m,'Trivial Male / Female Silent','Microphone','Reconstructed Female')