import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import seaborn

degree = 1
cluster = 1
# load train set and test set
cluster_file = np.loadtxt('classification/cluster_' + str(cluster) + '.csv', dtype=int)
trainset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
testset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)

prediction_testset = np.empty((0, 14), dtype=int)
y_testset = np.empty((0, 14), dtype=int)
y_predict = np.empty((0, 14), dtype=int)


# loading time series data
for i in range(cluster_file.size):
    print i
    print cluster_file[i]

    # generating tset file
    ################################################################################################

    # real line
    file = "/home/liuyingqi/Desktop/CustomerFlow/flow_per_shop/" + str(cluster_file[i]) + ".csv"
    info = pd.read_csv(file)
    info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                          411,412,413,414,415,416,417,
                          460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
    ts = info['count'].values
    ts_real = ts[390:]
    print ts_real.shape
    # fluent
    # part trend
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/fluent/ "+str(i+1)+" _clus_"+str(cluster)+"_trend.csv"
    info = pd.read_csv(file)
    ts_fluent_trend = info['x'].values
    ts_len = ts_fluent_trend.shape[0]-14
    x = np.arange(ts_len)+1
    print x.shape
    print ts_fluent_trend.shape
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression(fit_intercept=False))])
    model.fit(x[:, np.newaxis],ts_fluent_trend[:-14])
    prediction_fluent_trend = model.predict(np.arange(ts_len,ts_len+14)[:, np.newaxis])

    # part seasonal
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/fluent/ " + str(i + 1) + " _clus_"+str(cluster)+"_seasonal.csv"
    info = pd.read_csv(file)
    ts_fluent_seasonal = info['x'].values
    prediction_fluent_seasonal = ts_fluent_seasonal[-14:]

    # ts
    ts_fluent = ts_fluent_seasonal + ts_fluent_trend
    prediction_fluent = prediction_fluent_trend + prediction_fluent_seasonal


    # rare
    # part trend
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/rare/ "+str(i+1)+" _clus_"+str(cluster)+"_trend.csv"
    info = pd.read_csv(file)
    ts_rare_trend = info['x'].values
    ts_len = ts_rare_trend.shape[0]-14
    x = np.arange(ts_len)+1
    print x.shape
    print ts_rare_trend.shape
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression(fit_intercept=False))])
    model.fit(x[:, np.newaxis],ts_rare_trend[:-14])
    prediction_rare_trend = model.predict(np.arange(ts_len,ts_len+14)[:, np.newaxis])

    # part seasonal
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/rare/ " + str(i + 1) + " _clus_"+str(cluster)+"_seasonal.csv"
    info = pd.read_csv(file)
    ts_rare_seasonal = info['x'].values
    prediction_rare_seasonal = ts_rare_seasonal[-14:]

    # ts
    ts_rare = ts_rare_seasonal + ts_rare_trend
    prediction_rare = prediction_rare_trend + prediction_rare_seasonal

    y_testset = np.vstack((y_testset,np.round(ts_real[-14:])))
    prediction_testset = np.vstack((prediction_testset,np.round(prediction_fluent+prediction_rare)))

    # generating submission
    #####################################################################################################

    # real line
    file = "/home/liuyingqi/Desktop/CustomerFlow/flow_per_shop/" + str(cluster_file[i]) + ".csv"
    info = pd.read_csv(file)
    info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                          411, 412, 413, 414, 415, 416, 417,
                          460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
    ts = info['count'].values
    ts_real = ts[390:]

    # fluent
    # part trend
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/fluent/ " + str(i + 1) + " _clus_"+str(cluster)+"_trend.csv"
    info = pd.read_csv(file)
    ts_fluent_trend = info['x'].values
    ts_len = ts_fluent_trend.shape[0]
    x = np.arange(ts_len) + 1
    print x.shape
    print ts_fluent_trend.shape
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model.fit(x[:, np.newaxis], ts_fluent_trend)
    prediction_fluent_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

    # part seasonal
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/fluent/ " + str(i + 1) + " _clus_"+str(cluster)+"_seasonal.csv"
    info = pd.read_csv(file)
    ts_fluent_seasonal = info['x'].values
    prediction_fluent_seasonal = ts_fluent_seasonal[-14:]

    # ts
    ts_fluent = ts_fluent_seasonal + ts_fluent_trend
    prediction_fluent = prediction_fluent_trend + prediction_fluent_seasonal

    # rare
    # part trend
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/rare/ " + str(i + 1) + " _clus_"+str(cluster)+"_trend.csv"
    info = pd.read_csv(file)
    ts_rare_trend = info['x'].values
    ts_len = ts_rare_trend.shape[0]
    x = np.arange(ts_len) + 1
    print x.shape
    print ts_rare_trend.shape
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model.fit(x[:, np.newaxis], ts_rare_trend)
    prediction_rare_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

    # part seasonal
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/rare/ " + str(i + 1) + " _clus_"+str(cluster)+"_seasonal.csv"
    info = pd.read_csv(file)
    ts_rare_seasonal = info['x'].values
    prediction_rare_seasonal = ts_rare_seasonal[-14:]

    # ts
    ts_rare = ts_rare_seasonal + ts_rare_trend
    prediction_rare = prediction_rare_trend + prediction_rare_seasonal
    y_predict = np.vstack((y_predict,np.round(prediction_fluent+prediction_rare)))

print prediction_testset.shape
print y_predict.shape
prediction_testset[prediction_testset<0]=0

# test set scoring
sum = 0.
for i in range(prediction_testset.shape[0]):
    for j in range(prediction_testset.shape[1]):
        sum += np.absolute(
            (y_testset[i, j] - prediction_testset[i, j]) / (y_testset[i, j] + prediction_testset[i, j] + 0.000000001))
nt = float((prediction_testset.shape[0] * prediction_testset.shape[1]))
score = sum / nt
print score

np.savetxt('test_set/baseline_1_clus_' + str(cluster) + '_label.csv', y_testset, fmt='%d', delimiter=',')
np.savetxt('test_set/baseline_1_clus_' + str(cluster) + '_predict.csv', prediction_testset, fmt='%d', delimiter=',')
np.savetxt('submission/baseline_1_clus_' + str(cluster) + '_predict.csv', y_predict, fmt='%d', delimiter=',')