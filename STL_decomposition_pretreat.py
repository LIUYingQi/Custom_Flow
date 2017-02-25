# this file try to show how decomposite ts
import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn
import statsmodels.api as sm

##################################################################################
###   pre-treat
##################################################################################

# # load train set and test set
# cluster = 2
# cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)
#
# truncted_array = np.empty((0,217))
# counter = 0
# for example in cluster_file:
#     file = 'flow_per_shop/'+str(cluster_file[counter])+'_fluent.csv'
#     info = pd.read_csv(file)
#     info.drop(info.index[[348,349,350,351,352,353,354,446,447,448,449,450,451,452,
#                           411,412,413,414,415,416,417,
#                           460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
#     print info
#     date = info['target_dates'].values
#     ts = info['count'].values
#     ts = ts[243:]
#     ts = ts.astype(int)
#     counter+=1
#     truncted_array = np.vstack((truncted_array,ts))
# print truncted_array.shape
# np.savetxt('input/cluster_'+str(cluster)+'_fluent.csv',truncted_array,fmt='%d',delimiter=',')
#
# counter = 0
# truncted_array = np.empty((0,217))
#
# for example in cluster_file:
#     file = 'flow_per_shop/'+str(cluster_file[counter])+'_rare.csv'
#     info = pd.read_csv(file)
#     info.drop(info.index[[348,349,350,351,352,353,354,446,447,448,449,450,451,452,
#                           411,412,413,414,415,416,417,
#                           460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
#     date = info['target_dates'].values
#     ts = info['count'].values
#     ts = ts[243:]
#     ts = ts.astype(int)
#     counter+=1
#     truncted_array = np.vstack((truncted_array,ts))
# print truncted_array.shape
# np.savetxt('input/cluster_'+str(cluster)+'_rare.csv',truncted_array,fmt='%d',delimiter=',')
#
# cluster = 3
# cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)
#
# truncted_array = np.empty((0,217))
# counter = 0
# for example in cluster_file:
#     file = 'flow_per_shop/'+str(cluster_file[counter])+'_fluent.csv'
#     info = pd.read_csv(file)
#     info.drop(info.index[[348,349,350,351,352,353,354,446,447,448,449,450,451,452,
#                           411,412,413,414,415,416,417,
#                           460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
#     date = info['target_dates'].values
#     ts = info['count'].values
#     ts = ts[243:]
#     ts = ts.astype(int)
#     counter+=1
#     truncted_array = np.vstack((truncted_array,ts))
# print truncted_array.shape
# np.savetxt('input/cluster_'+str(cluster)+'_fluent.csv',truncted_array,fmt='%d',delimiter=',')
#
# counter = 0
# truncted_array = np.empty((0,217))
#
# for example in cluster_file:
#     file = 'flow_per_shop/'+str(cluster_file[counter])+'_rare.csv'
#     info = pd.read_csv(file)
#     info.drop(info.index[[348,349,350,351,352,353,354,446,447,448,449,450,451,452,
#                           411,412,413,414,415,416,417,
#                           460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
#     date = info['target_dates'].values
#     ts = info['count'].values
#     ts = ts[243:]
#     ts = ts.astype(int)
#     counter+=1
#     truncted_array = np.vstack((truncted_array,ts))
# print truncted_array.shape
# np.savetxt('input/cluster_'+str(cluster)+'_rare.csv',truncted_array,fmt='%d',delimiter=',')


cluster = 1
cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)

truncted_array = np.empty((0,70))
counter = 0
for example in cluster_file:
    file = 'flow_per_shop/'+str(cluster_file[counter])+'_fluent.csv'
    info = pd.read_csv(file)
    info.drop(info.index[[348,349,350,351,352,353,354,446,447,448,449,450,451,452,
                          411,412,413,414,415,416,417,
                          460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
    print info
    date = info['target_dates'].values
    ts = info['count'].values
    ts = ts[390:]
    ts = ts.astype(int)
    print ts.shape
    counter+=1
    truncted_array = np.vstack((truncted_array,ts))
print truncted_array.shape
np.savetxt('input/cluster_'+str(cluster)+'_fluent.csv',truncted_array,fmt='%d',delimiter=',')

counter = 0
truncted_array = np.empty((0,70))

for example in cluster_file:
    file = 'flow_per_shop/'+str(cluster_file[counter])+'_rare.csv'
    info = pd.read_csv(file)
    info.drop(info.index[[348,349,350,351,352,353,354,446,447,448,449,450,451,452,
                          411,412,413,414,415,416,417,
                          460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
    date = info['target_dates'].values
    ts = info['count'].values
    ts = ts[390:]
    ts = ts.astype(int)
    counter+=1
    truncted_array = np.vstack((truncted_array,ts))
print truncted_array.shape
np.savetxt('input/cluster_'+str(cluster)+'_rare.csv',truncted_array,fmt='%d',delimiter=',')

##################################################################################
###   decomposition
##################################################################################

# # decomposition
# cluster = 2
# cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)
# cluster_2_fluent = np.loadtxt('input/cluster_2_fluent.csv',dtype=int,delimiter=',')
#
# ts = cluster_2_fluent[4,:]
# res = sm.tsa.seasonal_decompose(ts,freq=7)
# print res.trend
# pyplot.figure()
# pyplot.plot(ts)
# pyplot.show()