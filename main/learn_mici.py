# by josua graf

import numpy as np
import matplotlib.pyplot as plt
from config import *

# load mic (s_clean) and clean female and clean male
S_clean_mic = np.load('S_clean_mic.npy')
S_clean_f = np.load('S_clean_female.npy')
S_clean_m = np.load('S_clean_male.npy')

# print shape of S_clean_i
print(np.shape(S_clean_mic))
print(np.shape(S_clean_f))
print(np.shape(S_clean_m))

# plot imported data
myPlot = plt.subplot(311)
myPlot = plt.plot(np.transpose(S_clean_mic))
myPlot = plt.subplot(312)
myPlot = plt.plot(np.transpose(S_clean_f))
myPlot = plt.subplot(313)
myPlot = plt.plot(np.transpose(S_clean_m))
plt.show(myPlot)

# define simple function g(acc,mic) = mic - sum(acc_{j!=i})
def g_simple(accj,mic):
    mici = mic - accj

# test function g
mic_f = g_simple(S_clean_m,S_clean_mic)
plt.show(plt.plot(mic_f))
