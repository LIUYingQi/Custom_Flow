# this file is to ensemble

import numpy as np

# load results

baselines = [1,3,4,5]
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