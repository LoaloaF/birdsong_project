import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adapted_classifier_visualized import classify as classify_vis
import random
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV,ElasticNet, MultiTaskLasso, MultiTaskElasticNetCV
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
from numpy import save
from numpy import load
import pickle

"""FILTERED DATA MATPLOTLIB SPECTROGRAMS"""

#import the filtered data list csv
filt_data_files = pd.read_csv('/Volumes/Drive/ETH/Neural_Systems/b8p2male-b10o15female_aligned/filtered/filt_data_files_MinAmp0.05_PadSec0.50.csv', index_col='rec_id')
# slice to sdr and DAQmx(to use in future perhaps) by getting rid of the third file, the .wav audio
filt_data_files = filt_data_files.drop('filt_DAQmxAudio', axis=1)

###############################################################################
#############################LOAD AND ORGANIZE THE DATA########################
###############################################################################
#get the S_trivial_m, S_trivial_f and S_clean subsets from all recordings

trivial_m_x = [] #
trivial_m_y = [] #
trivial_f_x = [] #
trivial_f_y = [] #
clean_m_x = [] #
clean_f_x = [] #
clean_y = [] #

#Put the S_trivial_m, S_trivial_f and S_clean together across recordings
for i, (rec_id, rec_id_files) in enumerate(filt_data_files.iterrows()):
    if i == 0:
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
    else:
        continue


names= ['trivial_m_x', 'trivial_m_y', 'trivial_f_x', 'trivial_f_y', 'clean_m_x',
        'clean_f_x','clean_y']

#save the data
for name in names:
    with open(name, 'wb') as name+'_file':
        pickle.dump(name, name+'_file') 
  
#open the data
for name in names:
    with open(name, 'rb') as vars()[name]:
        vars()[name] = pickle.load(vars()[name])
  


#Create a list of signals across all days (get rid of the "day" dimension) and
#concatenate along the frames

for name in names:
    vars()[name] = [item for sublist in vars()[name] for item in sublist]
    vars()[name] = np.transpose(np.concatenate(vars()[name], axis=1))

#Concatenate male and female together 
trivial_m_x = np.concatenate((trivial_m_x, trivial_f_x), axis=0)
trivial_m_y = np.concatenate((trivial_m_y, trivial_f_y), axis=0)

########Ordinary least squares Linear Regression########

####Train test split####
x_train, x_test, y_train, y_test = train_test_split(trivial_m_x, trivial_m_y, test_size=0.25, random_state=42)

####Fit the model####
reg = LinearRegression().fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
reg.score(x_train, y_train) #0.005217602296311823
coefficients= reg.coef_

####Make prediction####
y_pred = reg.predict(x_test)
r2_score(y_test, y_pred) #0.005358670806285639


y_pred = cross_val_predict(reg, trivial_m_x, trivial_m_y, cv=3)
r2_score(trivial_m_y, y_pred) # -0.011907515433415573
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred) #2.451648619442972e-12
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred) #8.192148785845101e-08



########Ridge#########################################

ridge = Ridge(alpha=0.5).fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
ridge.score(x_train, y_train) #0.00027214616623364384
ridge_coefficients= ridge.coef_
####Make prediction####
y_pred_ridge = ridge.predict(x_test)

####Prediction metrics- evaluate the quality of prediction####
r2_score_ridge = r2_score(y_test, y_pred_ridge) #0.0005083528151936361
#the best value is 0.0
mean_squared_error(y_test, y_pred_ridge) #2.356249559499751e-12
#the best value is 0.0
mean_absolute_error(y_test, y_pred_ridge) #7.924473545416611e-08


####Cross Validation of alpha####
####Fit the model####
clf = RidgeCV(alphas=np.logspace(-5, 5, 10)).fit(trivial_m_x, trivial_m_y) 
clf.alpha_ #1e-05
#Return the coefficient of determination R^2 of the prediction.
clf.score(trivial_m_x, trivial_m_y) #0.003191030285531968
####Make prediction####
y_pred_clf = clf.predict(trivial_m_x)

####Prediction metrics- evaluate the quality of prediction####
r2_score(trivial_m_y, y_pred_clf) #0.0017832681127065415
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred_clf) #2.4140146843112224e-12
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred_clf) #7.893972783438647e-08

#Try with cross validation prediction
y_pred_clf = cross_val_predict(clf, trivial_m_x, trivial_m_y, cv=3)
r2_score(trivial_m_y, y_pred_clf) #-0.011979565565654608
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred_clf) #2.4331698366120048e-12
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred_clf) #8.077433261076299e-08



########Lasso#########################################

####Fit the model####
lasso = Lasso(alpha=0.1).fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
lasso.score(x_train, y_train) #2.0785721978772887e-16
lasso_coefficients= reg.coef_
####Make prediction####
y_pred_lasso = lasso.predict(x_test)

####Prediction metrics- evaluate the quality of prediction####
r2_score_lasso = r2_score(y_test, y_pred_lasso) #-4.69331263393043e-06
#the best value is 0.0
mean_squared_error(y_test, y_pred_lasso) #2.3569360597541025e-12
#the best value is 0.0
mean_absolute_error(y_test, y_pred_lasso) #7.933480602705968e-08

#Try with cross validation prediction
y_pred_lasso = cross_val_predict(lasso, trivial_m_x, trivial_m_y, cv=3)
r2_score(trivial_m_y, y_pred_lasso) #-0.008139696923718123
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred_lasso) #2.4340297493805133e-12
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred_lasso) #8.057737236219667e-08




########Elstic Net###################################

####Fit the model####
ElNet = ElasticNet(alpha= 0.5, random_state=0).fit(trivial_m_x, trivial_m_y)
ElNet.score(trivial_m_x, trivial_m_y) #-1.545373138656844e-13



#Try with cross validation prediction
y_pred_ElNet = cross_val_predict(ElNet, trivial_m_x, trivial_m_y, cv=3) 
r2_score(trivial_m_y, y_pred_ElNet) #-0.008139696923718123
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred_ElNet) #2.4340297493805133e-12
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred_ElNet) #8.057737236219667e-08

#Multi Task Elstic Net with CV
ElNetCV = MultiTaskElasticNetCV(random_state=0, verbose=1).fit(trivial_m_x, trivial_m_y)
##UserWarning: Objective did not
#converge. You might want to increase the number of iterations


start = 100000
plt.figure()
plt.pcolormesh(np.log(y_test[start:start+100,:]))
plt.ylabel('time')
plt.xlabel('freq')
plt.figure()
plt.pcolormesh(np.log(y_pred[start:start+100,:]))
plt.ylabel('time')
plt.xlabel('freq')


start = 100000
plt.figure()
plt.pcolormesh(np.log(x_test[start:start+100,:]))
plt.ylabel('time')
plt.xlabel('freq')
plt.figure()
plt.pcolormesh(np.log(y_test[start:start+100,:]))
plt.ylabel('time')
plt.xlabel('freq')
plt.figure()
plt.pcolormesh(np.log(y_pred[start:start+100,:]))
plt.ylabel('time')
plt.xlabel('freq')


##### S_clean #################################################################
#Apply the best performing model on S_clean - OLS with R2 = 0.0054


reg = LinearRegression().fit(trivial_m_x, trivial_m_y)
#Return the coefficient of determination R^2 of the prediction.
reg.score(trivial_m_x, trivial_m_y) #0.005163539159205976


####Make prediction####
y_pred_male = reg.predict(clean_m_x)
r2_score(clean_y, y_pred_male) #-0.02428561641697913

y_pred_female = reg.predict(clean_f_x)
r2_score(clean_y, y_pred_female) #-0.016166908712702682

#plot
oo = np.transpose(clean_y)
aa = np.transpose(y_pred_female)
bb = np.transpose(y_pred_male)

start = 1000
plt.figure()
plt.pcolormesh(np.log(oo[:,start:start+1000]))
plt.ylabel('freq')
plt.xlabel('time')
plt.figure()
plt.pcolormesh(np.log(bb[:,start:start+1000]))
plt.ylabel('freq')
plt.xlabel('time')
plt.figure()
plt.pcolormesh(np.log(aa[:,start:start+1000]))
plt.ylabel('freq')
plt.xlabel('time')

  