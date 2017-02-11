# baseline 2 : using ARMA model to predict

import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import sklearn.preprocessing

cluster = 0
# load train set and test set
cluster_file = np.loadtxt('classification/cluster_' + str(cluster) + '.csv', dtype=int)
trainset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
testset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)

def draw_trend(timeSeries, size):
    rol_mean = pd.rolling_mean(timeSeries,size)
    rol_weighted_mean = pd.ewma(timeSeries, span=size)
    # f = plt.figure(facecolor='white')
    # plt.plot(timeSeries,'b',rol_mean,'g',rol_weighted_mean,'r')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean')
    # plt.show()
    return  rol_weighted_mean

def draw_acf_pacf(ts, lags=31):
    # type: (object, object) -> object
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

# array to save
predict_result = np.empty((0,14))
test_set_predict_result = np.empty((0,14))
# predicting
for i in cluster_file:
    print '##########################################################'
    print i
    # loading time series data
    file = 'flow_per_shop/' + str(i + 1) + '.csv'
    info = pd.read_csv(file)
    ts = info['count'].values

    # truncted
    label = ts[-14:]
    ts = ts[-154:-14]

    # pre treatment
    ts_mean = draw_trend(ts, 7)
    ts_detrending = ts - ts_mean
    ts_diff_1 = np.diff(ts_detrending)

    # # visulazing
    # plt.figure()
    # plt.plot(ts_diff_1)
    # plt.show()

    # test stationary
    d_order0 = sm.tsa.adfuller(ts_diff_1)
    print 'adf: ', d_order0[0]
    print 'p-value: ', d_order0[1]
    print'Critical values: ', d_order0[4]

    if d_order0[0] > d_order0[4]['5%']:
        print 'Time Series is  nonstationary'
    else:
        print 'Time Series is stationary'

    # # selecting parameter
    order = sm.tsa.arma_order_select_ic(ts_diff_1, max_ar=6, max_ma=3, ic=['aic'])
    # print order

    try:

        # ARMA model
        model = ARMA(ts_diff_1,(order['aic_min_order'][0],order['aic_min_order'][1]))
        predict_diff_1 = model.fit(disp=False).forecast(14)[0]

        # restore
        predict = np.cumsum(predict_diff_1)
        predict = predict + np.mean(ts[-7:])

        # use continuity is better
        stander = sklearn.preprocessing.StandardScaler()
        predict = stander.fit_transform(predict)
        predict = stander.fit(ts[-7:]).inverse_transform(predict)
        predict = np.round(predict)

        print predict
        predict_result = np.vstack((predict_result,predict))

        # visualizing
        # print predict.shape
        # plt.figure()
        # plt.plot(np.arange(len(ts)),ts)
        # plt.plot(np.arange(len(ts),len(ts)+14),predict)
        # plt.plot(np.arange(len(ts),len(ts)+14),label)
        # plt.show()

    except:
        print 'error'
        predict = ts[-7:]
        predict = np.concatenate((predict, predict), axis=0)

        print predict
        predict_result = np.vstack((predict_result,predict))

        continue

    if i in testset_file:
        print 'in test set'
        test_set_predict_result = np.vstack((test_set_predict_result, predict))

np.savetxt('test_set/baseline_4_clus_'+str(cluster)+'_predict.csv',test_set_predict_result,fmt='%d')
np.savetxt('submission/baseline_4_clus_' + str(cluster) + '_predict.csv', predict_result, fmt='%d')
