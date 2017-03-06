# baseline 3 using just last 2 week avrange

import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib import pyplot
import seaborn
num_clus = 4

for cluster in range(num_clus):
# load train set and test set

    ####################################################################################
    ### fluent part
    #################################################################################

    # fluent
    cluster_file = np.loadtxt('classification/cluster_' + str(cluster) + '.csv', dtype=int)
    trainset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
    testset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)
    trainset_x = np.empty((0,140), dtype = int)
    trainset_y = np.empty((0,14), dtype = int)
    testset_x = np.empty((0,140), dtype = int)
    testset_y = np.empty((0,14), dtype = int)

    for i in trainset_file:
        data = pd.read_csv('flow_per_shop/'+str(i)+'_fluent.csv')
        data = data['count'].values
        len_data = len(data)
        trainset_y = np.vstack((trainset_y,data[len_data-14:]))
        trainset_x = np.vstack((trainset_x,data[len_data-154:len_data-14]))

    for i in testset_file:
        data = pd.read_csv('flow_per_shop/'+str(i)+'_fluent.csv')
        data = data['count'].values
        len_data = len(data)
        testset_y = np.vstack((testset_y,data[len_data-14:]))
        testset_x = np.vstack((testset_x,data[len_data-154:len_data-14]))

    testset_x = testset_x[:,-14:]
    testset_x_holiday = np.round(np.mean(testset_x[:,-3:-2],axis=1)).astype(int)
    testset_x_workday = np.round(np.mean(testset_x[:,-8:-3],axis=1)).astype(int)

    prediction = np.empty((0,14))

    for i in range(len(testset_x)):
        array = []
        array = [testset_x_workday[i]]*4
        array.extend([testset_x_holiday[i]]*2)
        array.extend([testset_x_workday[i]]*5)
        array.extend([testset_x_holiday[i]]*2)
        array.extend([testset_x_workday[i]])
        prediction = np.vstack((prediction,np.array(array)))
    prediction = np.round(prediction)

    label_fluent_testset = testset_y
    predict_fluent_testset = prediction

    # predicting to submission
    submission = np.empty((0,21))

    for i in cluster_file:
        data = pd.read_csv('flow_per_shop/' + str(i) + '_fluent.csv')
        data = data['count'].values
        data = data[-21:]
        submission = np.vstack((submission,data))

    submission_x_1 = np.round(np.mean(submission[:, [ 6, 13,  20]], axis=1)).astype(int)
    submission_x_234 = np.round(np.mean(submission[:, [0, 1, 2, 7, 8, 9, 13, 14, 15, 16]], axis=1)).astype(int)
    submission_x_5 = np.round(np.mean(submission[:, [3, 10, 17]], axis=1)).astype(int)
    submission_x_67 = np.round(np.mean(submission[:, [4,5,11,12,18,19]], axis=1)).astype(int)

    submission = np.empty((0,14))
    for i in range(len(cluster_file)):
        array = [submission_x_234[i]]*3
        array.extend([submission_x_5[i]])
        array.extend([submission_x_67[i]]*2)
        array.extend([submission_x_1[i]])
        array.extend([submission_x_234[i]]*3)
        array.extend([submission_x_5[i]])
        array.extend([submission_x_67[i]]*2)
        array.extend([submission_x_1[i]])

        submission = np.vstack((submission,np.array(array)))
    submission = np.round(submission)

    predict_fluent_submission = submission

    #########################################################################
    ### rare
    #########################################################################

    cluster_file = np.loadtxt('classification/cluster_' + str(cluster) + '.csv', dtype=int)
    trainset_file = np.loadtxt('classification/cluster_' + str(cluster) + '_trainset.csv', dtype=int)
    testset_file = np.loadtxt('classification/cluster_' + str(cluster) + '_testset.csv', dtype=int)
    trainset_x = np.empty((0, 140), dtype=int)
    trainset_y = np.empty((0, 14), dtype=int)
    testset_x = np.empty((0, 140), dtype=int)
    testset_y = np.empty((0, 14), dtype=int)

    for i in trainset_file:
        data = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
        data = data['count'].values
        len_data = len(data)
        trainset_y = np.vstack((trainset_y, data[len_data - 14:]))
        trainset_x = np.vstack((trainset_x, data[len_data - 154:len_data - 14]))

    for i in testset_file:
        data = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
        data = data['count'].values
        len_data = len(data)
        testset_y = np.vstack((testset_y, data[len_data - 14:]))
        testset_x = np.vstack((testset_x, data[len_data - 154:len_data - 14]))

    testset_x = testset_x[:, -14:]
    testset_x_holiday = np.round(np.mean(testset_x[:, -3:-2], axis=1)).astype(int)
    testset_x_workday = np.round(np.mean(testset_x[:, -8:-3], axis=1)).astype(int)

    prediction = np.empty((0, 14))

    for i in range(len(testset_x)):
        array = [testset_x_workday[i]] * 4
        array.extend([testset_x_holiday[i]] * 2)
        array.extend([testset_x_workday[i]] * 5)
        array.extend([testset_x_holiday[i]] * 2)
        array.extend([testset_x_workday[i]])
        prediction = np.vstack((prediction, np.array(array)))
    prediction = np.round(prediction)
    # print prediction

    label_rare_testset = testset_y
    prediction_rare_testset = prediction

    # predicting to submission
    submission = np.empty((0,21))

    for i in cluster_file:
        data = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
        data = data['count'].values
        data = data[-21:]
        submission = np.vstack((submission,data))

    submission_x_1 = np.round(np.mean(submission[:, [ 6, 13,  20]], axis=1)).astype(int)
    submission_x_234 = np.round(np.mean(submission[:, [0, 1, 2, 7, 8, 9, 13, 14, 15, 16]], axis=1)).astype(int)
    submission_x_5 = np.round(np.mean(submission[:, [3, 10, 17]], axis=1)).astype(int)
    submission_x_67 = np.round(np.mean(submission[:, [4,5,11,12,18,19]], axis=1)).astype(int)

    submission = np.empty((0,14))
    for i in range(len(cluster_file)):
        array = [submission_x_234[i]]*3
        array.extend([submission_x_5[i]])
        array.extend([submission_x_67[i]]*2)
        array.extend([submission_x_1[i]])
        array.extend([submission_x_234[i]]*3)
        array.extend([submission_x_5[i]])
        array.extend([submission_x_67[i]]*2)
        array.extend([submission_x_1[i]])

        submission = np.vstack((submission,np.array(array)))
    submission = np.round(submission)
    predict_rare_submission = submission

    testset_y = np.empty((0, 14), dtype=int)
    for i in testset_file:
        data = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
        data = data['count'].values
        len_data = len(data)
        testset_y = np.vstack((testset_y, data[len_data - 14:]))
    print testset_y.shape

    np.savetxt('test_set/baseline_2_clus_' + str(cluster) + '_label.csv', testset_y, fmt='%d' , delimiter=',')

    prediction_testset = predict_fluent_testset+prediction_rare_testset
    prediction_testset[prediction_testset<0]=0
    print prediction_testset.shape
    np.savetxt('test_set/baseline_2_clus_' + str(cluster) + '_predict.csv', prediction_testset, fmt='%d' , delimiter=',')

    prediction_submission = predict_fluent_submission+predict_rare_submission
    prediction_submission = np.round(prediction_submission).astype(int)
    # scoring
    print 'scoring for cluster : '+str(cluster)
    sum = 0.
    for i in range(testset_y.shape[0]):
        for j in range(testset_y.shape[1]):
            sum += np.absolute((prediction_testset[i, j] - testset_y[i, j]) / (prediction_testset[i, j] + testset_y[i, j] + 0.000000001))
    nt = float((testset_y.shape[0] * testset_y.shape[1]))
    score = sum / nt
    print score
    np.savetxt('submission/baseline_2_clus_' + str(cluster) + '_predict.csv', prediction_submission, fmt='%d' ,delimiter=',')

    # visualizing check
    # counter = 0
    # for i in cluster_file:
    #     print counter
    #     data = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
    #     data = data['count'].values
    #     pyplot.figure()
    #     pyplot.plot(np.arange(0,495),data)
    #     pyplot.plot(np.arange(495,509),prediction_submission[counter])
    #     pyplot.show()
    #