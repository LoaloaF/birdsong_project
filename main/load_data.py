import pandas as pd
import numpy as np
from adapted_classifier import *
import random
from sklearn.model_selection import train_test_split
from numpy import savetxt
from numpy import savez_compressed
from numpy import load

#import the file list csv that Simon created
data_files = pd.read_csv('/Volumes/Drive/ETH/Neural_Systems/b8p2male-b10o15female_aligned/data_files.csv', index_col='rec_id')


###############################################################################
#############################LOAD AND ORGANIZE THE DATA########################
###############################################################################
#get the S_trivial_m, S_trivial_f and S_clean subsets from all recordings
#In Janosch's code, each recording has a list of S_trivial_m, S_trivial_f and S_clean.
#If any S_trivial_m, S_trivial_f and S_clean signals were detected, they get appeneded to
#the lists, if not- empty lists are appended
S_trivial_m_all = [] #
S_trivial_f_all = [] #
S_clean_all = [] #
#Put the S_trivial_m, S_trivial_f and S_clean together across recordings
counter=1
for filename in data_files["SdrChannels"]:
    S_trivial_m, S_trivial_f, S_clean = classify(filename, 0, -1)
    print(f'file {counter} finished')
    
    if S_trivial_m: #if S_trivial_m not empty, append it to the list
        S_trivial_m_all.append(S_trivial_m)
    if S_trivial_f: #if S_trivial_f not empty, append it to the list
        S_trivial_f_all.append(S_trivial_f)
    if S_clean: #if S_clean not empty, append it to the list
        S_clean_all.append(S_clean)

    counter +=1


##### S_trivial ###############################################################
#Should concatenate S_trivial_m and S_trivial_f together across all days, but
#S_trivial_f is empty. 
    
#Create a list of signals across all days from male (get rid of the "day" dimension)    
S_trivial_m_all_flat = [item for sublist in S_trivial_m_all for item in sublist]

#Take the first array from evey sublist (mic channel)
mic = [item[0] for item in S_trivial_m_all_flat]
#Take the second array from every sublist (male channel)
male = [item[1] for item in S_trivial_m_all_flat]

#Concatenate the arrays along the frames
y = np.concatenate(mic, axis=1)
x = np.concatenate(male, axis=1)


##### S_clean #################################################################

#Create a list of signals across all days from clean (get rid of the "day" dimension)    
S_clean_all_flat = [item for sublist in S_clean_all for item in sublist]

#Take the first array from evey sublist (mic channel)
mic_clean = [item[0] for item in S_clean_all_flat]
#Take the second array from every sublist (male channel)
male_clean = [item[1] for item in S_clean_all_flat]
#Take the second array from every sublist (male channel)
female_clean = [item[2] for item in S_clean_all_flat]

#Concatenate the arrays along the frames
mic_clean = np.concatenate(mic_clean, axis=1)
male_clean = np.concatenate(male_clean, axis=1)
female_clean = np.concatenate(female_clean, axis=1)
