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

"""CLEAN VOCALIZATIONS- MANUALLY LABELLED"""


###############################################################################
#############################LOAD AND ORGANIZE THE DATA########################
###############################################################################
#get the S_trivial_m, S_trivial_f and S_clean subsets from all recordings




names= ['trivial_m_x', 'trivial_m_y', 'trivial_f_x', 'trivial_f_y', 'clean_m_x',
        'clean_f_x','clean_y']

  
#open the data
for name in names:
    vars()[name] = np.load(name+".npy")
  
# spectrograms
trivial_m_x, _, _, _ = plt.specgram(trivial_m_x, Fs=24000)
trivial_m_y, _, _, _ = plt.specgram(trivial_m_y, Fs=24000)
trivial_f_x, _, _, _ = plt.specgram(trivial_f_x, Fs=24000)
trivial_f_y, _, _, _ = plt.specgram(trivial_f_y, Fs=24000)

clean_m_x, _, _, _ = plt.specgram(clean_m_x, Fs=24000)
clean_f_x, _, _, _ = plt.specgram(clean_f_x, Fs=24000)
clean_y, _, _, _ = plt.specgram(clean_y, Fs=24000)

#concatenate male and female together
trivial_m_x = np.concatenate((trivial_m_x, trivial_f_x), axis=1)
trivial_m_y = np.concatenate((trivial_m_y, trivial_f_y), axis=1)

trivial_m_x = np.transpose(trivial_m_x)
trivial_m_y = np.transpose(trivial_m_y)



########Ordinary least squares Linear Regression########

####Train test split####
x_train, x_test, y_train, y_test = train_test_split(trivial_m_x, trivial_m_y, test_size=0.25, random_state=42)


####Fit the model####
reg = LinearRegression().fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
reg.score(x_train, y_train) # 0.041450756727440946
coefficients= reg.coef_

####Make prediction####
y_pred = reg.predict(x_test)
r2_score(y_test, y_pred) #-0.07967621661226337


y_pred = cross_val_predict(reg, trivial_m_x, trivial_m_y, cv=3)
r2_score(trivial_m_y, y_pred) #-16.839971955296793
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred) #7.322796230589489e-12
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred) #1.448974257571389e-07



########Ridge#########################################

ridge = Ridge(alpha=0.5).fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
ridge.score(x_train, y_train) #4.790482721090037e-06
ridge_coefficients= ridge.coef_
####Make prediction####
y_pred_ridge = ridge.predict(x_test)

####Prediction metrics- evaluate the quality of prediction####
r2_score_ridge = r2_score(y_test, y_pred_ridge) #-0.0003566749488951416
#the best value is 0.0
mean_squared_error(y_test, y_pred_ridge) #1.2188816408994208e-12
#the best value is 0.0
mean_absolute_error(y_test, y_pred_ridge) #6.180966062728223e-08



####Cross Validation of alpha####
####Fit the model####
trivial_m_x = np.transpose(trivial_m_x)
trivial_m_y = np.transpose(trivial_m_y)

clf = RidgeCV(alphas=np.logspace(-10, 10, 100)).fit(trivial_m_x, trivial_m_y) 
clf.alpha_ #1e-10
#Return the coefficient of determination R^2 of the prediction.
clf.score(trivial_m_x, trivial_m_y) #0.3691341646626037
####Make prediction####
y_pred_clf = clf.predict(trivial_m_x)

####Prediction metrics- evaluate the quality of prediction####
r2_score(trivial_m_y, y_pred_clf) #0.31342039751909695
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred_clf) #5.186757352245299e-13
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred_clf) #5.594165675510405e-08

#Try with cross validation prediction
y_pred_clf = cross_val_predict(clf, trivial_m_x, trivial_m_y, cv=3)
r2_score(trivial_m_y, y_pred_clf) #
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred_clf) #1.3291711614621479e-09
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred_clf) #8.840482902004718e-07



########Lasso#########################################

####Fit the model####
lasso = Lasso(alpha=0.5).fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
lasso.score(x_train, y_train) #3.8283841090210575e-18
lasso_coefficients= reg.coef_
####Make prediction####
y_pred_lasso = lasso.predict(x_test)

####Prediction metrics- evaluate the quality of prediction####
r2_score_lasso = r2_score(y_test, y_pred_lasso) #-5.956285044019928
#the best value is 0.0
mean_squared_error(y_test, y_pred_lasso) #1.0998191777313142e-12
#the best value is 0.0
mean_absolute_error(y_test, y_pred_lasso) #7.246915935655628e-08

#Try with cross validation prediction
y_pred_lasso = cross_val_predict(lasso, trivial_m_x, trivial_m_y, cv=3)
r2_score(trivial_m_y, y_pred_lasso) #-0.11900273269595178
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred_lasso) #8.605883654486582e-13
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred_lasso) #6.712707047038071e-08




########Elstic Net###################################

####Fit the model####
ElNet = ElasticNet(alpha= 0.5, random_state=0).fit(trivial_m_x, trivial_m_y)
ElNet.score(trivial_m_x, trivial_m_y) #-7.988766865769053e-17

#Try with cross validation prediction
y_pred_ElNet = cross_val_predict(ElNet, trivial_m_x, trivial_m_y, cv=3) 
r2_score(trivial_m_y, y_pred_ElNet) #-0.11900273269595178
#the best value is 0.0
mean_squared_error(trivial_m_y, y_pred_ElNet) #8.605883654486582e-13
#the best value is 0.0
mean_absolute_error(trivial_m_y, y_pred_ElNet) #6.712707047038071e-08

#Multi Task Elstic Net with CV
ElNetCV = MultiTaskElasticNetCV(random_state=0, verbose=1).fit(trivial_m_x, trivial_m_y)
##UserWarning: Objective did not
#converge. You might want to increase the number of iterations



start = 6000
plt.figure()
plt.pcolormesh(np.log(trivial_m_x[:,start:start+1000]))
plt.ylabel('freq')
plt.xlabel('time')
plt.figure()
plt.pcolormesh(np.log(trivial_m_y[:,start:start+1000]))
plt.ylabel('freq')
plt.xlabel('time')
plt.figure()
plt.pcolormesh(np.log(y_pred_clf[:,start:start+1000]))
plt.ylabel('freq')
plt.xlabel('time')


##### S_clean #################################################################
#Apply the best performing model on S_clean - Ridge with R2 = 0.31342039751909695

for name in names:
    with open(name, 'rb') as vars()[name]:
        vars()[name] = pickle.load(vars()[name])
        
clean_m_x = np.transpose(clean_m_x)
clean_f_x = np.transpose(clean_f_x)
clean_y = np.transpose(clean_y)

####Make prediction####
y_pred_male = clf.predict(clean_m_x)
r2_score(clean_y, y_pred_male) #-3360.1980472935597

y_pred_female = clf.predict(clean_f_x)
r2_score(clean_y, y_pred_female) #-3827.755082195441

#plot
oo = np.transpose(clean_y)
aa = np.transpose(y_pred_female)
bb = np.transpose(y_pred_male)

start = 0
plt.figure()
plt.pcolormesh(np.log(oo[:,start:start+619]))
plt.ylabel('freq')
plt.xlabel('time')
plt.figure()
plt.pcolormesh(np.log(bb[:,start:start+619]))
plt.ylabel('freq')
plt.xlabel('time')
plt.figure()
plt.pcolormesh(np.log(aa[:,start:start+619]))
plt.ylabel('freq')
plt.xlabel('time')






  