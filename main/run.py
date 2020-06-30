import pandas as pd
import numpy as np
from adapted_classifier import *
import random
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV,ElasticNet, MultiTaskLasso, MultiTaskElasticNetCV
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
from numpy import save
from numpy import load
import pickle

from adapted_classifier_visualized2 import classify as classify_vis

#import the filtered data list csv
filt_data_files = pd.read_csv('/Volumes/Drive/ETH/Neural_Systems/b8p2male-b10o15female_aligned/filtered/filt_data_files_MinAmp0.05_PadSec0.50.csv', index_col='rec_id')
# slice to sdr and DAQmx(to use in future perhaps) by getting rid of the third file, the .wav audio
filt_data_files = filt_data_files.drop('filt_DAQmxAudio', axis=1)


trivial_m_x = [] #
trivial_m_y = [] #
trivial_f_x = [] #
trivial_f_y = [] #
clean_m_x = [] #
clean_f_x = [] #
clean_y = [] #

#Put the S_trivial_m, S_trivial_f and S_clean together across recordings
for i, (rec_id, rec_id_files) in enumerate(filt_data_files.iterrows()):
    print(f'\nProcessing recording {rec_id} ({i+1}/{filt_data_files.shape[0]})...')
    daq_file, sdr_file = rec_id_files.values
    if not np.load(daq_file).any().any():
        print('Empty.')
        continue
    male_x, male_y, female_x, female_y, clean_m, clean_f, clean_y_ = classify_vis(sdr_file,
                daq_file, 0, -1, 
                show_energy_plot=False, show_framesizes=False, rec_id=rec_id,
                show_vocalization=False)
    print('Done.\n')
    
    
    if male_x: #if S_trivial_m not empty, append it to the list
        trivial_m_x.append(male_x)
        trivial_m_y.append(male_y)
    if female_x: #if S_trivial_f not empty, append it to the list
        trivial_f_x.append(female_x)
        trivial_f_y.append(female_y)
    if clean_m: #if S_clean not empty, append it to the list
        clean_m_x.append(clean_m)
        clean_f_x.append(clean_f)
        clean_y.append(clean_y_)


with open('trivial_m_x', 'wb') as trivial_m_x:
  pickle.dump(trivial_m_x, trivial_m_x)

with open('trivial_m_y', 'wb') as trivial_m_y:
  pickle.dump(trivial_m_y, trivial_m_y)
  
with open('trivial_f_x', 'wb') as trivial_f_x:
  pickle.dump(trivial_f_x, trivial_f_x)

with open('trivial_f_y', 'wb') as trivial_f_y:
  pickle.dump(trivial_f_y, trivial_f_y)

with open('clean_m_x', 'wb') as clean_m_x:
  pickle.dump(clean_m_x, clean_m_x)  

with open('clean_f_x', 'wb') as clean_f_x:
  pickle.dump(clean_f_x, clean_f_x)  
  
with open('clean_y', 'wb') as clean_y:
  pickle.dump(clean_y, clean_y)  






