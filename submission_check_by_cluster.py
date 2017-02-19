import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot

baseline = 7
cluster = 2

# load train set and test set
cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)
trainset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
testset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)
result_reversefit = np.loadtxt('submission/baseline_'+str(baseline)+'_clus_'+str(cluster)+'_predict.csv')

# visualizing check
counter = 0
for i in cluster_file:
    print counter
    data = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
    data = data['count'].values
    pyplot.figure()
    pyplot.plot(np.arange(0,495),data)
    pyplot.plot(np.arange(495,509),result_reversefit[counter])
    pyplot.show()
    counter+=1