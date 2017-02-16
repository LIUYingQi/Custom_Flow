# this file is to view test set result

import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn

cluster = 0
baseline = 5
testset = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)
print testset

testset_predict = np.loadtxt('test_set/baseline_'+str(baseline)+'_clus_'+str(cluster)+'_predict.csv',dtype=int)

counter = 0
for item in testset:
    data = pd.read_csv('flow_per_shop/' + str(item) + '.csv')
    data = data['count'].values
    pyplot.figure()
    pyplot.plot(np.arange(0,495),data)
    pyplot.plot(np.arange(481,495),testset_predict[counter])
    pyplot.show()
    counter+=1