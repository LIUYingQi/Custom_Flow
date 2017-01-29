# this file is to verify submission.csv
import numpy as np
import sklearn
import pandas as pd

# load submission matrix
with open('testset.csv','rb') as test_file:
    testset =  np.loadtxt(test_file,dtype=int)
    testset.tolist()
submission = np.empty((0,14),dtype=float)
for i in testset:
    with open('flow_per_shop/'+str(i)+'.csv', 'rb') as test_file:
        info = pd.read_csv(test_file)
        info = info.loc[480:493,['count']].values
        info = info.reshape(14)
        submission = np.vstack((submission,info))
print submission

# load label matrix
with open('testset.csv','rb') as test_file:
    testset =  np.loadtxt(test_file,dtype=int)
    testset.tolist()
label = np.empty((0,14),dtype=float)
for i in testset:
    with open('flow_per_shop/'+str(i)+'.csv', 'rb') as test_file:
        info = pd.read_csv(test_file)
        info = info.loc[481:494,['count']].values
        info = info.reshape(14)
        label = np.vstack((label,info))
print label

# score judge
# first verify is same shape
print submission.shape
print label.shape
sum = 0.
for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        sum += np.absolute((submission[i,j]-label[i,j])/(submission[i,j]+label[i,j]+0.000000001))
nt = float((label.shape[0]*label.shape[1]))
score = sum/nt
print score