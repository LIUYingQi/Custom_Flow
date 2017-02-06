# this file is to ensemble

import numpy as np

# load results

baselines = [1,3,4]
num_clusters = 3
weights = {}

# weighting
for baseline in baselines:
    for cluster in range(num_clusters):
        testset_y = np.loadtxt('result/baseline_'+str(baseline)+'_clus_' + str(cluster) + '_label.csv')
        prediction = np.loadtxt('result/baseline_'+str(baseline)+'_clus_' + str(cluster) + '_predict.csv')

        # scoring
        sum = 0.
        for i in range(testset_y.shape[0]):
            for j in range(testset_y.shape[1]):
                sum += np.absolute(
                    (prediction[i, j] - testset_y[i, j]) / (prediction[i, j] + testset_y[i, j] + 0.000000001))
        nt = float((testset_y.shape[0] * testset_y.shape[1]))
        score = sum / nt
        print 'baseline: ' + str(baseline) + '  cluster: ' + str(cluster) + '  score: '+ str(score)
        weight = 1/score

        weights[str(baseline)+'_'+str(cluster)] = weight
print weights

# ensembling
for cluster in range(num_clusters):
    total_weight =0
    prediction_ensembled = np.zeros(np.loadtxt('result/baseline_1_clus_' + str(cluster) + '_label.csv').shape)
    for baseline in baselines:
        total_weight += weights[str(baseline)+'_'+str(cluster)]
    for baseline in baselines:
        weight = weights[str(baseline)+'_'+str(cluster)]/total_weight
        prediction_ensembled += np.loadtxt('result/baseline_'+str(baseline)+'_clus_' + str(cluster) + '_predict.csv') * weight
    prediction_ensembled = np.round(prediction_ensembled)
    # scoring
    testset_y = np.loadtxt('result/baseline_1_clus_' + str(cluster) + '_label.csv')
    # scoring
    sum = 0.
    for i in range(testset_y.shape[0]):
        for j in range(testset_y.shape[1]):
            sum += np.absolute(
                (prediction_ensembled[i, j] - testset_y[i, j]) / (prediction_ensembled[i, j] + testset_y[i, j] + 0.000000001))
    nt = float((testset_y.shape[0] * testset_y.shape[1]))
    score = sum / nt
    print score