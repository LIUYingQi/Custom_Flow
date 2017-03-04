import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import seaborn

degree = 1
cluster = 2
# load train set and test set
cluster_file = np.loadtxt('classification/cluster_' + str(cluster) + '.csv', dtype=int)
trainset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
testset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)

y_predict_stl_arima = np.empty((0, 14), dtype=int)
y_predict_stl_ets = np.empty((0, 14), dtype=int)

# loading time series data
for i in range(cluster_file.size):
    print i
    print cluster_file[i]

    # real line
    file = "/home/liuyingqi/Desktop/CustomerFlow/flow_per_shop/" + str(cluster_file[i]) + ".csv"
    info = pd.read_csv(file)
    info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                          411, 412, 413, 414, 415, 416, 417,
                          460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
    ts = info['count'].values
    ts_real = ts[243:]

    # fluent
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/fluent/" + str(i + 1) + "_clus_"+str(cluster)+"_ets.csv"
    info = pd.read_csv(file)
    ts_fluent_ets = info['Point.Forecast'].values

    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/fluent/" + str(i + 1) + "_clus_"+str(cluster)+"_stl_arima.csv"
    info = pd.read_csv(file)
    ts_fluent_arima = info['Point.Forecast'].values

    # rare
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/rare/" + str(i + 1) + "_clus_" + str(cluster) + "_stl_ets.csv"
    info = pd.read_csv(file)
    ts_rare_ets = info['Point.Forecast'].values

    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/rare/" + str(i + 1) + "_clus_" + str(cluster) + "_stl_arima.csv"
    info = pd.read_csv(file)
    ts_rare_arima = info['Point.Forecast'].values

    ts_ets = ts_fluent_ets + ts_rare_ets
    ts_stl_arima = ts_fluent_arima + ts_rare_arima
    ts_ets = np.round(ts_ets)
    ts_stl_arima = np.round(ts_stl_arima)
    y_predict_stl_arima = np.vstack((y_predict_stl_arima,ts_stl_arima))
    y_predict_stl_ets = np.vstack((y_predict_stl_ets,ts_ets))

y_predict_stl_ets[y_predict_stl_ets<0]=0
y_predict_stl_arima[y_predict_stl_arima<0]=0

np.savetxt('submission/baseline_3_clus_' + str(cluster) + '_predict.csv', y_predict_stl_ets, fmt='%d', delimiter=',')
np.savetxt('submission/baseline_4_clus_' + str(cluster) + '_predict.csv', y_predict_stl_arima, fmt='%d', delimiter=',')