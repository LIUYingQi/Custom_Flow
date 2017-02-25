import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

cluster = 2
# load train set and test set
cluster_file = np.loadtxt('classification/cluster_' + str(cluster) + '.csv', dtype=int)
trainset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
testset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)

# loading time series data
for i in range(30,50):
    file = "/home/liuyingqi/Desktop/CustomerFlow/decomposition/ "+str(i+1)+" _clus_3_trend.csv"
    info = pd.read_csv(file)
    ts = info['x'].values
    ts_len = ts.shape[0]
    x = np.arange(ts_len)+1
    print x.shape
    print ts.shape
    model = Pipeline([('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression(fit_intercept=False))])
    model.fit(x[:, np.newaxis],ts)
    prediction = model.predict(np.arange(ts_len,ts_len+20)[:, np.newaxis])
    plt.figure()
    plt.plot(ts)
    plt.plot(np.arange(ts_len,ts_len+20),prediction)
    plt.show()