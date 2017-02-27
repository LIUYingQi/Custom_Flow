import pandas as pd
import numpy as np
from scipy import signal as sps
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import seaborn

######################################################################################
# save train set
######################################################################################

# stander = preprocessing.StandardScaler()
# trainset_x = np.empty((0,14))
# trainset_y = np.empty((0,21))
#
# for item in range(1500):
#     print item
#     shop_id = trainset_shop_info.iloc[item,0]
#     file = 'flow_per_shop/' + str(shop_id) + '.csv'
#     info = pd.read_csv(file)
#     ts = info['count'].values
#     ts = ts[-21:]
#     # print ts
#     week1 = stander.fit_transform(ts[0:7])
#     # print week1
#     week2 = stander.fit_transform(ts[7:14])
#     # print week2
#     week3 = stander.fit_transform(ts[14:21])
#     # print week2
#     trainset_y = np.vstack((trainset_y,np.hstack((week1,np.hstack((week2,week3))))))
#     city = trainset_shop_info.iloc[item,1]
#     weather = pd.read_csv('input/city_weather/'+city,names=['date','max_t','min_t','class','wind_orientation','wind_class'])
#     weather = weather.iloc[-51:-30,:]
#     weather['class'] = weather['class'].apply(change_2_pinyin)
#     del weather['wind_orientation']
#     del weather['wind_class']
#     weather['day'] = np.array([2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1])
#     del weather['date']
#     info = trainset_shop_info.iloc[item,:]
#     info = info.values
#     weather = weather.values
#     for day in weather:
#         day_info = np.hstack((info,day))
#         trainset_x = np.vstack((trainset_x,day_info))
#
# trainset_x = pd.DataFrame(trainset_x)
# trainset_x.to_csv('input/baseline7_trainset_x.csv')
# np.savetxt('input/baseline7_trainset_y.csv',trainset_y,fmt='%f')


cluster_list = [2,3]
for cluster in cluster_list:
    # load train set and test set
    cluster_file = np.loadtxt('classification/cluster_' + str(cluster) + '.csv', dtype=int)
    trainset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
    testset_file = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)

    prediction_testset = np.empty((0, 14), dtype=int)
    y_testset = np.empty((0, 14), dtype=int)
    y_predict = np.empty((0, 14), dtype=int)

    # loading fluent time series data
    for i in range(cluster_file.size):
        print i
        print cluster_file[i]

        file = "/home/liuyingqi/Desktop/CustomerFlow/flow_per_shop/" + str(cluster_file[i]) + ".csv"
        info = pd.read_csv(file)
        info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                              404 ,405, 406, 407, 408, 409 , 410, 411, 412, 413, 414, 415, 416, 417,
                              460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
        ts = info['count'].values
        ts_real = ts[243:]

        file = "/home/liuyingqi/Desktop/CustomerFlow/flow_per_shop/" + str(cluster_file[i]) + "_fluent.csv"
        info = pd.read_csv(file)
        info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                              404 ,405, 406, 407, 408, 409 , 410, 411, 412, 413, 414, 415, 416, 417,
                              460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
        ts = info['count'].values
        ts_fluent = ts[243:]

        file = "/home/liuyingqi/Desktop/CustomerFlow/flow_per_shop/" + str(cluster_file[i]) + "_rare.csv"
        info = pd.read_csv(file)
        info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                              404 ,405, 406, 407, 408, 409 , 410, 411, 412, 413, 414, 415, 416, 417,
                              460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
        ts = info['count'].values
        ts_rare = ts[243:]

        size = 7  #  use odd
        RM_ts_fluent = pd.rolling_mean(ts_fluent,size,center=True)
        RM_ts_fluent[:size/2] = RM_ts_fluent[size/2]
        RM_ts_fluent[-(size/2)-1:] = RM_ts_fluent[-(size/2)-1]

        RM_ts_rare = pd.rolling_mean(ts_rare,size,center=True)
        RM_ts_rare[:size/2] = RM_ts_rare[size/2]
        RM_ts_rare[-(size/2)-1:] = RM_ts_rare[-(size/2)-1]

        plt.figure()
        # plt.plot(ts_real)
        plt.plot(ts_fluent)
        plt.plot(RM_ts_fluent)
        plt.plot(ts_fluent-RM_ts_fluent)
        plt.show()
