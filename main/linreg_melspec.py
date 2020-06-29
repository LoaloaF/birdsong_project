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

"""MELSPECTROGRAMS"""

###############################################################################
#############################LOAD AND ORGANIZE THE DATA########################
###############################################################################
# get the S_trivial_m, S_trivial_f and S_clean subsets from all recordings
# In Janosch's code, each recording has a list of S_trivial_m, S_trivial_f and S_clean.
# If any S_trivial_m, S_trivial_f and S_clean signals were detected, they get appeneded to
# the lists, if not- empty lists are appended

# import the file list csv that Simon created
data_files = pd.read_csv('../data_files.csv', index_col='rec_id')

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
    counter += 1





#################################S_trivial ####################################
#Should concatenate S_trivial_m and S_trivial_f together across all days, but
#S_trivial_f is empty. 
    
#Create a list of signals across all days from male (get rid of the "day" dimension)    
S_trivial_m_all_flat = [item for sublist in S_trivial_m_all for item in sublist]

#save the data
with open('S_trivial_m_all', 'wb') as S_trivial_m_all:
  pickle.dump(S_trivial_m_all_flat, S_trivial_m_all)
#open the data
with open('S_trivial_m_all', 'rb') as S_trivial_m_all:
    S_trivial_m_all_flat = pickle.load(S_trivial_m_all)
exit()
    
#Take the first array from evey sublist (mic channel)
mic = [item[0] for item in S_trivial_m_all_flat]
#Take the second array from every sublist (male channel)
male = [item[1] for item in S_trivial_m_all_flat]

#Concatenate the arrays along the frames
y = np.transpose(np.concatenate(mic, axis=1))
x = np.transpose(np.concatenate(male, axis=1))



###############################################################################
#############################LINEAR REGRESSION#################################
###############################################################################


#################################S_trivial ####################################


#MELSPECTROGRAM DATA
########Ordinary least squares Linear Regression########

####Train test split####
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

####Fit the model####
reg = LinearRegression().fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
reg.score(x_train, y_train) #0.022011202797197614
coefficients= reg.coef_

####Make prediction####
y_pred = reg.predict(x_test)


####Prediction metrics- evaluate the quality of prediction####
#R2_score= It represents the proportion of variance (of y) that has been explained by the
#independent variables in the model. It provides an indication of goodness of fit
#and therefore a measure of how well unseen samples are likely to be predicted
#by the model, through the proportion of explained variance. Best possible score is 1.0 
r2_score(y_test, y_pred) # -0.015451101294948174
#the best value is 0.0
mean_squared_error(y_test, y_pred) #0.00013726673
#the best value is 0.0
mean_absolute_error(y_test, y_pred) #0.00051730196

#Try with cross validation prediction
y_pred = cross_val_predict(reg, x, y, cv=3)
r2_score(y, y_pred) # -0.8381593206228277
#the best value is 0.0
mean_squared_error(y, y_pred) #0.00020003068
#the best value is 0.0
mean_absolute_error(y, y_pred) #0.0005999213

#cross val 0.8227082193278795 lower


########Ridge#########################################

ridge = Ridge(alpha=0.5).fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
ridge.score(x_train, y_train) #0.019570364860566434
ridge_coefficients= ridge.coef_
####Make prediction####
y_pred_ridge = ridge.predict(x_test)

####Prediction metrics- evaluate the quality of prediction####
r2_score_ridge = r2_score(y_test, y_pred_ridge) #0.007960149368177155
#the best value is 0.0
mean_squared_error(y_test, y_pred_ridge) #0.0001342104
#the best value is 0.0
mean_absolute_error(y_test, y_pred_ridge) #0.00052425894


####Cross Validation of alpha####
####Fit the model####
clf = RidgeCV(alphas=np.logspace(-6, 6, 100)).fit(x, y) #70,100,0.1
clf.alpha_ #1.1497569953977356
#Return the coefficient of determination R^2 of the prediction.
clf.score(x, y) #0.014369855695492047
####Make prediction####
y_pred_clf = clf.predict(x)

####Prediction metrics- evaluate the quality of prediction####
r2_score_clf = r2_score(y, y_pred_clf) #0.01348769442626145
#the best value is 0.0
mean_squared_error(y, y_pred_clf) #7.747000550505998e-05
#the best value is 0.0
mean_absolute_error(y, y_pred_clf) #0.0005041117994546001

#Try with cross validation prediction
y_pred_clf = cross_val_predict(clf, x, y, cv=3)
r2_score(y, y_pred_clf) # -0.3775439656891002
#the best value is 0.0
mean_squared_error(y, y_pred_clf) #0.0001218898182731349
#the best value is 0.0
mean_absolute_error(y, y_pred_clf) #0.0005801960904272817

#cross val 0.38550411505727733 lower 

########Lasso#########################################

####Fit the model####
lasso = Lasso(alpha=0.1).fit(x_train, y_train)
#Return the coefficient of determination R^2 of the prediction.
lasso.score(x_train, y_train) #-1.6807322419138004e-10
lasso_coefficients= reg.coef_
####Make prediction####
y_pred_lasso = lasso.predict(x_test)

####Prediction metrics- evaluate the quality of prediction####
r2_score_lasso = r2_score(y_test, y_pred_lasso) #-4.333005852998409e-05 == -0.00004333005
#the best value is 0.0
mean_squared_error(y_test, y_pred_lasso) #0.00013500836
#the best value is 0.0
mean_absolute_error(y_test, y_pred_lasso) #0.0006196304

#Try with cross validation prediction
y_pred_lasso = cross_val_predict(lasso, x, y, cv=3)
r2_score(y, y_pred_lasso) #-0.0002686650433182912
#the best value is 0.0
mean_squared_error(y, y_pred_lasso) #7.85883e-05
#the best value is 0.0
mean_absolute_error(y, y_pred_lasso) #0.0005987262

#cross val 0.0002253349847883071 smaller

####Multi task lasso Cross Validation#### 
####Fit the model####
MultiTaskLassoCV = MultiTaskLassoCV(random_state=0, verbose=1).fit(x, y)
#UserWarning: Objective did not
#converge. You might want to increase the number of iterations



########Elstic Net###################################

####Fit the model####
ElNet = ElasticNet(alpha= 0.5, random_state=0).fit(x,y)
ElNet.score(x, y) #-1.1142739679728243e-16


#Try with cross validation prediction
y_pred_ElNet = cross_val_predict(ElNet, x, y, cv=3) 
r2_score(y, y_pred_ElNet) #-0.0002686650433182912
#the best value is 0.0
mean_squared_error(y, y_pred_ElNet) #7.85883e-05
#the best value is 0.0
mean_absolute_error(y, y_pred_ElNet) #0.0005987262

#Multi Task Elstic Net with CV
ElNetCV = MultiTaskElasticNetCV(random_state=0, verbose=1).fit(x, y)
##UserWarning: Objective did not
#converge. You might want to increase the number of iterations


#Plot 
start = 10000
plt.figure()
plt.pcolormesh(np.log(x[start:start+1000,:]))
plt.ylabel('time')
plt.xlabel('freq')
plt.figure()
plt.pcolormesh(np.log(y[start:start+1000,:]))
plt.ylabel('time')
plt.xlabel('freq')







##### S_clean #################################################################

#Create a list of signals across all days from clean (get rid of the "day" dimension)    
S_clean_all_flat = [item for sublist in S_clean_all for item in sublist]

#save the data
with open('S_clean_all', 'wb') as S_clean_all:
  pickle.dump(S_clean_all_flat, S_clean_all)
#open the data
with open('S_clean_all', 'rb') as S_clean_all:
    S_clean_all_flat = pickle.load(S_clean_all)
    
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


#apply the same model to predict the mic data from male_clean





""" 
FROM AMPLITUDE DATA 
"""
#below this point not relevant 

#########From amplitude data#############

#The structure of all the subsets is [y, y_f, y_m, t]   
#Get the actual amplitude data from lists that are not empty
S_trivial_f_data = []
for file in S_trivial_f_all:
    if file:
        #access the level of frames
        for frames in file:
            data = frames[0:2]
            S_trivial_f_data.append(data)
S_trivial_f_array = np.asarray(S_trivial_f_data)


S_trivial_m_data = []
for file in S_trivial_m_all:
    if file:
        #access the level of frames
        for frames in file:
            indx = [0,2]
            data = [frames[i] for i in indx]
            S_trivial_m_data.append(data)
S_trivial_m_array = np.asarray(S_trivial_m_data)

S_clean_data = []
for file in S_clean_all:
    if file:
        #access the level of frames
        for frames in file:
            data = frames[0:3]
            S_trivial_m_data.append(data)
S_clean_array = np.asarray(S_clean_data)

###############################################################################
#############################GET THE DATA IN THE RIGHT SHAPE###################
###############################################################################

#Turn the sublists into arrays, as currently they are lists
S_f = []
for frame in S_trivial_f_array:
    #for i in range(batch_size):
    jj = np.vstack((x, y)).T
    S_f.append(jj)

S_m = []
for frame in S_trivial_m_array:
    #for i in range(batch_size):
    jj = np.vstack((x, y)).T
    S_m.append(jj)

S_cl = []
for frame in S_clean_array:
    #for i in range(batch_size):
    jj = np.vstack((x, y)).T
    S_cl.append(jj)
    print(len(y),y.shape, len(x),x.shape)


#the arrays in S_f, S_m and S_clean are of different lengths
#Just keep the ones that have the majority of lengths within S_f, S_m and S_clean


#try to do this for all at once
f_m_cl = [S_f, S_m, S_cl]

f_m_cl_arrays = []
for list_ in f_m_cl:
    
    shapes = []
    for array in S_m:
        shapes.append(array.shape[0])
        
    #get unique shapes
    uniqe_shapes = np.unique(shapes, return_counts=True)
    #counts of the unique shapes
    counts = uniqe_shapes[1]
    #find the index of the highest occuring shape
    max_idx = [i for i, j in enumerate(counts) if j == max(counts)][0]
    #get the most often occuring shape
    highest_occuring_length = uniqe_shapes[0][max_idx]
    
    #take only the arrays that have the same shape (as otherwise cannot concatenate the
    #arrays with different shapes into one multidimensional array)
    same_length_shapes = [array for array in list_ if array.shape[0] == highest_occuring_length]
    
    array_ = np.asarray(same_length_shapes)
    f_m_cl_arrays.append(array_)


v=np.asarray(S_m)
np.save('S_trivial_male.npy', v)

y = v[:,:,0]
x = v[:,:,1]


y_flat = np.concatenate(y).ravel()
x_flat = np.concatenate(x).ravel()

xtrain, xtest, ytrain, ytest = train_test_split(x_flat, y_flat, test_size = 0.25, shuffle = False)


xtrain = xtrain.reshape(-1, 1)
xtest = xtest.reshape(-1, 1)


m = LinearRegression()
m.fit(xtrain, ytrain)
m.score(xtrain, ytrain) 
m.score(xtest, ytest)
m.score(x, y)
ypred_train = m.predict(xtrain)
ypred_test = m.predict(xtest)
ypred_all = m.predict(x)
r2_score(ytest, ypred_test) #0.362332




def perform_linear_regression_sklearn(x_train,y_train,time,x_test):
    num_samples = len(x_train)
    num_test_samples = len(x_test)
    inputs,labels = get_data(x_train,y_train,time,batch_size=num_samples)
    print(inputs.shape,labels.shape)
    reg=[]
    for i in range(labels.shape[1]):
        reg.append(LinearRegression().fit(inputs, labels[:,i]))

    test_input,_ = get_data(x_test,y_train,time,batch_size=num_test_samples)
    y_test_predicted = []
    for i in range(labels.shape[1]):
        y_test_predicted.append(reg[i].predict(test_input))
    y_test_predicted = np.asarray(y_test_predicted)
    return y_test_predicted



def get_data(x_train,y_train,shuffle=False,batch_size=32):
    """Builds a batch i.e. (x, y) pair."""
    if shuffle:
        condns = random.sample(range(1, 98), batch_size)
    else:
        condns = range(batch_size)
    x = []
    y = []
    for i in range(batch_size):
        x_file_name = x_train[condns[i]]
        y_file_name = y_train[condns[i]]
        x.append(x_file_name)
        y.append(y_file_name)
    #x = np.asarray(x)
    #y = np.asarray(y)
    return x, y



for frame in v:
    #for i in range(batch_size):
    y = frame[0]
    x = frame[0]
    print(len(y), len(x))

y = v[:,0]
x = v[:,1]

x.shape #x=numpy array that consists of lists
Out[176]: (12,)

type(x[0]) #the elements of array x are lists
Out[174]: list

len(x[0]) #the len of these lists varies. I want to turn them into arrays too
Out[179]: 17920

#male
xi= []
yi= []
for frame in v: #(one of 12)
    for subframe in frame:
        y = np.asarray(subframe[0])
        x = np.asarray(subframe[1])
        xi.append(x)
        yi.append(y)
        print(len(y),y.shape, len(x),x.shape)
    
    
for frame in v:
    print(frame)
    
    
out = np.concatenate(xi).ravel()


#female 
k=np.array(S_trivial_f_data)
y = v[:,0]
x = v[:,1]

xifem= []
yifem= []
for frame in k:
    #for i in range(batch_size):
    y = np.asarray(frame[0])
    x = np.asarray(frame[1])
    xifem.append(x)
    yifem.append(y)
    print(len(y),y.shape, len(x),x.shape)

xi= []
yi= []
v_arrays = []
for frame in v:
    #for i in range(batch_size):
    y = np.asarray(frame[0])
    x = np.asarray(frame[1])
    xi.append(x)
    yi.append(y)
    jj = np.vstack((x, y)).T
    v_arrays.append(jj)
    print(len(y),y.shape, len(x),x.shape)
    
#the arrays in v_arrays are of different lengths
#so just remove the ones that are of different lengths
shapes = []
for list_ in v_arrays:
    shapes.append(list_.shape[0])

uniqe_shapes = np.unique(shapes, return_counts=True)
#counts of the unique shapes
counts = uniqe_shapes[1]
#find the index of the highest occuring shape
max_idx = [i for i, j in enumerate(counts) if j == max(counts)][0]

#get the most often occuring shape
highest_occuring_length = uniqe_shapes[0][max_idx]

#take only the arrays that have the same shape (as otherwise cannot concatenate the
#arrays with different shapes into one multidimensional array)
same_length_shapes = [array for array in v_arrays if array.shape[0] == highest_occuring_length]

aa = np.asarray(same_length_shapes)


np.save('S_trivial_male_first_5.npy', aa)










