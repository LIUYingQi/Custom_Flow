# this file is to ensemble

import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn
# load results

baselines = [3,7]
num_clusters = 3
weights = {}
total_num = 0
cluster_weight = []
final_score = 0

for cluster in range(num_clusters):
    total_num += len(np.loadtxt('classification/cluster_'+str(cluster)+'.csv'))

for cluster in range(num_clusters):
    cluster_weight.append(float(len(np.loadtxt('classification/cluster_'+str(cluster)+'.csv')))/total_num)
    print cluster_weight[cluster]

# weighting
for cluster in range(num_clusters):
    for baseline in baselines:
        testset_y = np.loadtxt('test_set/baseline_'+str(baseline)+'_clus_' + str(cluster) + '_label.csv')
        prediction = np.loadtxt('test_set/baseline_'+str(baseline)+'_clus_' + str(cluster) + '_predict.csv')
        prediction[prediction<0]=0

        # scoring
        sum = 0.
        for i in range(testset_y.shape[0]):
            for j in range(testset_y.shape[1]):
                sum += np.absolute(
                    (prediction[i, j] - testset_y[i, j]) / (prediction[i, j] + testset_y[i, j] + 0.000000001))
        nt = float((testset_y.shape[0] * testset_y.shape[1]))
        score = sum / nt
        print 'baseline: ' + str(baseline) + '  cluster: ' + str(cluster) + '  score: '+ str(score)
        # weight = np.exp(1/score)
        # weight = 1/score
        # weight = np.power(1/score,1/score)
        # weight = np.exp2(1/score)
        # weight = 1/score**6
        # weight = -np.log(score)
        weight = (-np.log(score-0.06)) ** 4

        weights[str(baseline)+'_'+str(cluster)] = weight
print weights

# ensembling
for cluster in range(num_clusters):
    total_weight =0
    prediction_ensembled = np.zeros(np.loadtxt('test_set/baseline_1_clus_' + str(cluster) + '_label.csv').shape)
    for baseline in baselines:
        total_weight += weights[str(baseline)+'_'+str(cluster)]
    for baseline in baselines:
        weight = weights[str(baseline)+'_'+str(cluster)]/total_weight
        print 'weight: cluster_'+str(cluster)+'_ baseline _ '+str(baseline)+' _ '+str(weight)
        prediction = np.loadtxt('test_set/baseline_' + str(baseline) + '_clus_' + str(cluster) + '_predict.csv')
        prediction[prediction<0]=0
        prediction_ensembled += prediction * weight
    prediction_ensembled = np.round(prediction_ensembled)
    np.savetxt('test_set/clus_'+str(cluster)+'_result.csv',prediction_ensembled,fmt='%d')
    # scoring
    testset_y = np.loadtxt('test_set/baseline_1_clus_' + str(cluster) + '_label.csv')
    # scoring
    sum = 0.
    for i in range(testset_y.shape[0]):
        for j in range(testset_y.shape[1]):
            sum += np.abs((prediction_ensembled[i, j] - testset_y[i, j])) / np.abs((prediction_ensembled[i, j] + testset_y[i, j] + 0.000000001))
    nt = float((testset_y.shape[0] * testset_y.shape[1]))
    score = sum / nt
    print score
    final_score += cluster_weight[cluster]* score

print 'final :' + str(final_score)

# visualizing

prediction_ensembled = np.loadtxt('test_set/clus_'+str(2)+'_result.csv',dtype=int)
counter = 0
test_set = np.loadtxt('classification/cluster_'+str(2)+'_testset.csv',dtype=int)
for item in test_set:
    file = 'flow_per_shop/' + str(item) + '.csv'
    info = pd.read_csv(file)
    ts = info['count'].values
    prediction = prediction_ensembled[counter]
    pyplot.figure(figsize=(18, 8))
    pyplot.plot(np.arange(0,495),ts)
    pyplot.plot(np.arange(481,495),prediction)
    pyplot.axvline(98, color='brown')
    pyplot.axvline(170, color='red')
    pyplot.axvline(227, color='yellow')
    pyplot.axvline(311, color='green')
    pyplot.axvline(448, color='green')
    pyplot.axvline(464, color='brown')
    pyplot.axvspan(129, 142, facecolor='0.5', alpha=0.5)
    pyplot.legend(['ts', '10.1', '1212', 'spring D', '5.1', 'Moon', '10.1', '1111'])
    pyplot.show()
    counter+=1