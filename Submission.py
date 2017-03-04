# this file is to ensemble

import numpy as np

# load results

### attention for baseline or baseline_1
baselines = [2,3,4,7]
num_clusters = 4
weights = {}

# weighting
for baseline in baselines:
    for cluster in range(num_clusters):

        if baseline==1 and cluster==0:
            weights[str(baseline) + '_' + str(cluster)] = 0.000000001
            continue


        if baseline == 3 and cluster!=0:
            weights[str(baseline) + '_' + str(cluster)] = 10
            continue

        if baseline == 3 and cluster==0:
            weights[str(baseline) + '_' + str(cluster)] = 0.0001
            continue


        if baseline == 4 and cluster!=0:
            weights[str(baseline) + '_' + str(cluster)] = 10
            continue

        if baseline == 4 and cluster==0:
            weights[str(baseline) + '_' + str(cluster)] = 0.0001
            continue

        testset_y = np.loadtxt('test_set/baseline_'+str(baseline)+'_clus_' + str(cluster) + '_label.csv',delimiter=',',dtype=int)
        prediction = np.loadtxt('test_set/baseline_'+str(baseline)+'_clus_' + str(cluster) + '_predict.csv',delimiter=',',dtype=int)
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
        weight = (-np.log(score)) ** 3

        weights[str(baseline)+'_'+str(cluster)] = weight
print weights

# ensemble to submit
result_per_cluster =[]
for cluster in range(num_clusters):
    total_weight =0
    prediction_ensembled = np.zeros(np.loadtxt('submission/baseline_2_clus_' + str(cluster) + '_predict.csv',delimiter=',').shape)
    for baseline in baselines:
        total_weight += weights[str(baseline)+'_'+str(cluster)]
    for baseline in baselines:
        weight = weights[str(baseline)+'_'+str(cluster)]/total_weight
        prediction = np.loadtxt('submission/baseline_' + str(baseline) + '_clus_' + str(cluster) + '_predict.csv',delimiter=',',dtype=int)
        prediction[prediction < 0] = 0
        prediction_ensembled += prediction * weight
    prediction_ensembled = np.round(prediction_ensembled).astype(int)
    print prediction_ensembled
    np.savetxt('submission/clus_' + str(cluster) + '_predict.csv',prediction_ensembled,fmt='%d')
    result_per_cluster.append(prediction_ensembled.tolist())
print np.array(result_per_cluster).shape

# generating submission file
labels = np.loadtxt('labels.csv',dtype=int)
submission = np.empty((0,14),dtype=int)
for i in labels:
    submission = np.vstack((submission,np.array(result_per_cluster[i].pop(0))))
print submission.shape
print np.arange(1,2001).shape
submission = np.concatenate((np.arange(1,2001).reshape((2000,1)),submission),axis=1)
print submission.shape
print submission
np.savetxt('submmission.csv',submission,fmt='%d',delimiter=',')
