# this file is to define train set and test set
# do labeling for cluster first before this

import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn
import statsmodels.api as sm

# split train set and test set
trainset = np.arange(1,2001)
trainset = np.random.choice(trainset,1500,False)
trainset = np.sort(trainset)
trainset = trainset.tolist()
testset = []
for i in range(1,2001):
    if i not in trainset:
        testset.append(i)
print trainset
print testset
trainset = np.array(trainset,dtype=np.int32)
testset = np.array(testset,dtype=np.int32)

with open('trainset.csv','wb') as trainset_file:
    np.savetxt(trainset_file,trainset,fmt='%d')

with open('testset.csv','wb') as test_file:
    np.savetxt(test_file,testset,fmt='%d')

# load labels
labels = np.loadtxt('labels.csv',dtype=int)
print labels
cluster_num = np.max(labels)+1

# generating trainset and testset for each clustring
for i in range(cluster_num):
    print i
    indice = np.where(labels == i)[0]+1
    cluster = indice
    cluster_trainset = np.intersect1d(indice,trainset)
    cluster_testset = np.intersect1d(indice,testset)
    print cluster.shape
    print cluster_trainset.shape
    print cluster_testset.shape
    print ''
    np.savetxt('classification/cluster_'+str(i)+'.csv',cluster,fmt='%d')
    np.savetxt('classification/cluster_'+str(i)+'_trainset.csv',cluster_trainset,fmt='%d')
    np.savetxt('classification/cluster_'+str(i)+'_testset.csv',cluster_testset,fmt='%d')

# change here to see result
info = np.loadtxt('classification/cluster_2.csv',dtype=int)
for i in info:
    print i

    file = 'flow_per_shop/' + str(i) + '.csv'
    info = pd.read_csv(file)
    ts = info['count'].values
    ts = ts[242:]

    file = 'flow_per_shop/' + str(i) + '_fluent.csv'
    info_fluent = pd.read_csv(file)
    ts_fluent = info_fluent['count'].values
    ts_fluent = ts_fluent[242:]

    file = 'flow_per_shop/' + str(i) + '_rare.csv'
    info_rare = pd.read_csv(file)
    ts_rare = info_rare['count'].values
    ts_rare = ts_rare[242:]

    pyplot.figure(figsize=(10, 8))
    pyplot.plot(ts)
    pyplot.plot(ts_fluent)
    pyplot.plot(ts_rare)
    # res = sm.tsa.seasonal_decompose(ts_fluent,freq=7)
    # fig = res.plot()
    # fig.show()

    pyplot.show()