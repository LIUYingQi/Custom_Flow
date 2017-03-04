import numpy as np
import pandas as pd
from pypinyin import lazy_pinyin
from sklearn import preprocessing
import xgboost as xgb
from sklearn import metrics
from matplotlib import pyplot
import seaborn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

num_cluster = 4

# load train test set
trainset = np.loadtxt('trainset.csv',dtype=int) -1
testset = np.loadtxt('testset.csv',dtype=int) -1

def change_2_pinyin(location):
    city_name = []
    for item in lazy_pinyin(unicode(location,encoding='utf-8')):
        city_name.extend(item.encode())
    city_name = ''.join(city_name)
    return city_name

# load shop info
shop_info = pd.read_csv('input/shop_info.txt')
shop_info['city_name'] = shop_info['city_name'].apply(change_2_pinyin)
shop_info['cate_1_name'] = shop_info['cate_1_name'].apply(change_2_pinyin)
shop_info['cate_2_name'] = shop_info['cate_2_name'].apply(change_2_pinyin)
del shop_info['cate_3_name']
pattern = np.loadtxt('classification/pattern.csv',dtype=float)
shop_info['pattern'] = pattern
shop_info = shop_info.fillna(value=0.0)
trainset_shop_info = shop_info.iloc[trainset,:]
testset_shop_info = shop_info.iloc[testset,:]

######################################################################################
# save train set  ( fluent part and rare part )
######################################################################################

stander = preprocessing.StandardScaler()
trainset_x = np.empty((0,14))
fluent_trainset_y = np.empty((0,21))
rare_trainset_y = np.empty((0,21))

# 1500 item in train set
# save x as (1500*21,14) here 14 is coulum num  y as (1500,21)

for item in range(1500):
    print item
    shop_id = trainset_shop_info.iloc[item,0]

    # part fluent
    file = 'flow_per_shop/' + str(shop_id) + '_fluent.csv'
    info = pd.read_csv(file)
    info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                          404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                          460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
    ts_fluent = info['count'].values
    ts_fluent = ts_fluent[243:]

    size = 7  # use odd
    RM_ts_fluent = pd.rolling_mean(ts_fluent, size, center=True)
    RM_ts_fluent[:size / 2] = RM_ts_fluent[size / 2]
    RM_ts_fluent[-(size / 2) - 1:] = RM_ts_fluent[-(size / 2) - 1]
    WAVE_ts_fluent = ts_fluent - RM_ts_fluent

    WAVE_ts_fluent = WAVE_ts_fluent[-21:]
    # print ts
    week1 = stander.fit_transform(WAVE_ts_fluent[0:7])
    # print week1
    week2 = stander.fit_transform(WAVE_ts_fluent[7:14])
    # print week2
    week3 = stander.fit_transform(WAVE_ts_fluent[14:21])
    # print week2
    fluent_trainset_y = np.vstack((fluent_trainset_y,np.hstack((week1,np.hstack((week2,week3))))))

    # part rare
    file = 'flow_per_shop/' + str(shop_id) + '_rare.csv'
    info = pd.read_csv(file)
    info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                          404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                          460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
    ts_rare = info['count'].values
    ts_rare = ts_rare[243:]

    size = 7  # use odd
    RM_ts_rare = pd.rolling_mean(ts_rare, size, center=True)
    RM_ts_rare[:size / 2] = RM_ts_rare[size / 2]
    RM_ts_rare[-(size / 2) - 1:] = RM_ts_rare[-(size / 2) - 1]
    WAVE_ts_rare = ts_rare - RM_ts_rare

    WAVE_ts_rare = WAVE_ts_rare[-21:]
    # print ts
    week1 = stander.fit_transform(WAVE_ts_rare[0:7])
    # print week1
    week2 = stander.fit_transform(WAVE_ts_rare[7:14])
    # print week2
    week3 = stander.fit_transform(WAVE_ts_rare[14:21])
    # print week2
    rare_trainset_y = np.vstack((rare_trainset_y,np.hstack((week1,np.hstack((week2,week3))))))

    # read weather as x part
    city = trainset_shop_info.iloc[item,1]
    weather = pd.read_csv('input/city_weather/'+city,names=['date','max_t','min_t','class','wind_orientation','wind_class'])
    weather = weather.iloc[-51:-30,:]
    weather['class'] = weather['class'].apply(change_2_pinyin)
    del weather['wind_orientation']
    del weather['wind_class']
    weather['day'] = np.array([2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1])
    del weather['date']
    info = trainset_shop_info.iloc[item,:]
    info = info.values
    weather = weather.values
    for day in weather:
        day_info = np.hstack((info,day))
        trainset_x = np.vstack((trainset_x,day_info))

trainset_x = pd.DataFrame(trainset_x)
trainset_x.to_csv('input/baseline7_trainset_x.csv')
np.savetxt('input/baseline7_fluent_trainset_y.csv',fluent_trainset_y,fmt='%f')
np.savetxt('input/baseline7_rare_trainset_y.csv',rare_trainset_y,fmt='%f')

###########################################################################################
# save test set
############################################################################################

stander = preprocessing.StandardScaler()
testset_x = np.empty((0,14))
fluent_testset_y = np.empty((0,21))
rare_testset_y = np.empty((0,21))


for item in range(500):
    print item
    shop_id = testset_shop_info.iloc[item,0]

    # fluent part
    file = 'flow_per_shop/' + str(shop_id) + '_fluent.csv'
    info = pd.read_csv(file)
    info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                           404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                          460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
    ts_fluent = info['count'].values
    ts_fluent = ts_fluent[243:]

    size = 7  # use odd
    RM_ts_fluent = pd.rolling_mean(ts_fluent, size, center=True)
    RM_ts_fluent[:size / 2] = RM_ts_fluent[size / 2]
    RM_ts_fluent[-(size / 2) - 1:] = RM_ts_fluent[-(size / 2) - 1]
    WAVE_ts_fluent = ts_fluent - RM_ts_fluent

    WAVE_ts_fluent = WAVE_ts_fluent[-21:]
    # print ts
    week1 = stander.fit_transform(WAVE_ts_fluent[0:7])
    # print week1
    week2 = stander.fit_transform(WAVE_ts_fluent[7:14])
    # print week2
    week3 = stander.fit_transform(WAVE_ts_fluent[14:21])
    # print week2
    fluent_testset_y = np.vstack((fluent_testset_y,np.hstack((week1,np.hstack((week2,week3))))))

    # fluent part
    file = 'flow_per_shop/' + str(shop_id) + '_rare.csv'
    info = pd.read_csv(file)
    info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                           404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                          460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
    ts_rare = info['count'].values
    ts_rare = ts_rare[243:]

    size = 7  # use odd
    RM_ts_rare = pd.rolling_mean(ts_rare, size, center=True)
    RM_ts_rare[:size / 2] = RM_ts_rare[size / 2]
    RM_ts_rare[-(size / 2) - 1:] = RM_ts_rare[-(size / 2) - 1]
    WAVE_ts_rare = ts_rare - RM_ts_rare

    WAVE_ts_rare = WAVE_ts_rare[-21:]
    # print ts
    week1 = stander.fit_transform(WAVE_ts_rare[0:7])
    # print week1
    week2 = stander.fit_transform(WAVE_ts_rare[7:14])
    # print week2
    week3 = stander.fit_transform(WAVE_ts_rare[14:21])
    # print week2
    rare_testset_y = np.vstack((rare_testset_y, np.hstack((week1, np.hstack((week2, week3))))))

    # weather paer test x
    city = testset_shop_info.iloc[item,1]
    weather = pd.read_csv('input/city_weather/'+city,names=['date','max_t','min_t','class','wind_orientation','wind_class'])
    weather = weather.iloc[-51:-30,:]
    weather['class'] = weather['class'].apply(change_2_pinyin)
    del weather['wind_orientation']
    del weather['wind_class']
    weather['day'] = np.array([2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1])
    del weather['date']
    info = testset_shop_info.iloc[item,:]
    info = info.values
    weather = weather.values
    for day in weather:
        day_info = np.hstack((info,day))
        testset_x = np.vstack((testset_x,day_info))

testset_x = pd.DataFrame(testset_x)
testset_x.to_csv('input/baseline7_testset_x.csv')
np.savetxt('input/baseline7_fluent_testset_y.csv',fluent_testset_y,fmt='%f')
np.savetxt('input/baseline7_rare_testset_y.csv',rare_testset_y,fmt='%f')

############################################################################
# generating submission file
############################################################################

for cluster in range(num_cluster):
    cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)

    submission_x = np.empty((0,14))
    for item in cluster_file:
        print item
        shop_id = shop_info.iloc[item-1,0]
        city = shop_info.iloc[item-1,1]
        weather = pd.read_csv('input/city_weather/'+city,names=['date','max_t','min_t','class','wind_orientation','wind_class'])
        weather = weather.iloc[-30:-16,:]
        weather['class'] = weather['class'].apply(change_2_pinyin)
        del weather['wind_orientation']
        del weather['wind_class']
        weather['day'] = np.array([2,3,4,5,6,7,1,2,3,4,5,6,7,1])
        del weather['date']
        info = shop_info.iloc[item-1,:]
        info = info.values
        weather = weather.values
        for day in weather:
            day_info = np.hstack((info,day))
            submission_x = np.vstack((submission_x,day_info))

    submission_x = pd.DataFrame(submission_x)
    submission_x.to_csv('input/baseline7_submission_x_'+str(cluster)+'.csv')

###############################################################################
# label encoder for test set x
###############################################################################

trainset_x = pd.read_csv('input/baseline7_trainset_x.csv')
testset_x = pd.read_csv('input/baseline7_testset_x.csv')
totalset = pd.concat([trainset_x,testset_x])
print totalset.shape
print totalset

# cleaning unseen label
for item in range(num_cluster):
    with open('input/baseline7_submission_x_'+str(item)+'.csv', 'r') as file :
        filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('xiaodaozhongxue', 'xiaoyu')
    filedata = filedata.replace('xiaoxue', 'xiaoyu')
    filedata = filedata.replace('zhongdaodaxue', 'xiaoyu')
    # Write the file out again
    with open('input/baseline7_submission_x_'+str(item)+'.csv', 'w') as file:
        file.write(filedata)

submission_x_0 = pd.read_csv('input/baseline7_submission_x_'+str(0)+'.csv')
submission_x_1 = pd.read_csv('input/baseline7_submission_x_'+str(1)+'.csv')
submission_x_2 = pd.read_csv('input/baseline7_submission_x_'+str(2)+'.csv')
submission_x_3 = pd.read_csv('input/baseline7_submission_x_'+str(3)+'.csv')

# encode
le = preprocessing.LabelEncoder()

totalset.iloc[:,2] = le.fit_transform(totalset.iloc[:,2])
trainset_x.iloc[:,2] = le.transform(trainset_x.iloc[:,2])
testset_x.iloc[:,2] = le.transform(testset_x.iloc[:,2])
submission_x_0.iloc[:,2] = le.transform(submission_x_0.iloc[:,2])
submission_x_1.iloc[:,2] = le.transform(submission_x_1.iloc[:,2])
submission_x_2.iloc[:,2] = le.transform(submission_x_2.iloc[:,2])
submission_x_3.iloc[:,2] = le.transform(submission_x_3.iloc[:,2])


totalset.iloc[:,8] = le.fit_transform(totalset.iloc[:,8])
trainset_x.iloc[:,8] = le.transform(trainset_x.iloc[:,8])
testset_x.iloc[:,8] = le.transform(testset_x.iloc[:,8])
submission_x_0.iloc[:,8] = le.transform(submission_x_0.iloc[:,8])
submission_x_1.iloc[:,8] = le.transform(submission_x_1.iloc[:,8])
submission_x_2.iloc[:,8] = le.transform(submission_x_2.iloc[:,8])
submission_x_3.iloc[:,8] = le.transform(submission_x_3.iloc[:,8])


totalset.iloc[:,9] = le.fit_transform(totalset.iloc[:,9])
trainset_x.iloc[:,9] = le.transform(trainset_x.iloc[:,9])
testset_x.iloc[:,9] = le.transform(testset_x.iloc[:,9])
submission_x_0.iloc[:,9] = le.transform(submission_x_0.iloc[:,9])
submission_x_1.iloc[:,9] = le.transform(submission_x_1.iloc[:,9])
submission_x_2.iloc[:,9] = le.transform(submission_x_2.iloc[:,9])
submission_x_3.iloc[:,9] = le.transform(submission_x_3.iloc[:,9])


totalset.iloc[:,13] = le.fit_transform(totalset.iloc[:,13])
trainset_x.iloc[:,13] = le.transform(trainset_x.iloc[:,13])
testset_x.iloc[:,13] = le.transform(testset_x.iloc[:,13])
submission_x_0.iloc[:,13] = le.transform(submission_x_0.iloc[:,13])
submission_x_1.iloc[:,13] = le.transform(submission_x_1.iloc[:,13])
submission_x_2.iloc[:,13] = le.transform(submission_x_2.iloc[:,13])
submission_x_3.iloc[:,13] = le.transform(submission_x_3.iloc[:,13])

totalset.iloc[:,14] = le.fit_transform(totalset.iloc[:,14])
trainset_x.iloc[:,14] = le.transform(trainset_x.iloc[:,14])
testset_x.iloc[:,14] = le.transform(testset_x.iloc[:,14])
submission_x_0.iloc[:,14] = le.transform(submission_x_0.iloc[:,14])
submission_x_1.iloc[:,14] = le.transform(submission_x_1.iloc[:,14])
submission_x_2.iloc[:,14] = le.transform(submission_x_2.iloc[:,14])
submission_x_3.iloc[:,14] = le.transform(submission_x_3.iloc[:,14])

trainset_x['4'] = trainset_x['4'].astype(int)
trainset_x['5'] = trainset_x['5'].astype(int)
testset_x['4'] = testset_x['4'].astype(int)
testset_x['5'] = testset_x['5'].astype(int)
submission_x_0['5'] = submission_x_0['5'].astype(int)
submission_x_0['4'] = submission_x_0['4'].astype(int)
submission_x_1['5'] = submission_x_1['5'].astype(int)
submission_x_1['4'] = submission_x_1['4'].astype(int)
submission_x_2['5'] = submission_x_2['5'].astype(int)
submission_x_2['4'] = submission_x_2['4'].astype(int)
submission_x_3['5'] = submission_x_3['5'].astype(int)
submission_x_3['4'] = submission_x_3['4'].astype(int)

print trainset_x.dtypes
print testset_x.dtypes

encoder = preprocessing.OneHotEncoder()
numed_label = totalset.iloc[:,[2,8,9,13,14]].values
encoder.fit(numed_label)

trainset_x_add = encoder.transform(trainset_x.iloc[:,[2,8,9,13,14]].values).toarray()
trainset_x = trainset_x.iloc[:,[3,4,5,6,7,10,11,12]].values
trainset_x = np.hstack((trainset_x,trainset_x_add))
print trainset_x.shape
del trainset_x_add

testset_x_add = encoder.transform(testset_x.iloc[:,[2,8,9,13,14]].values).toarray()
testset_x = testset_x.iloc[:,[3,4,5,6,7,10,11,12]].values
testset_x = np.hstack((testset_x,testset_x_add))
print testset_x.shape
del testset_x_add

trainset_fluent_y = np.loadtxt('input/baseline7_fluent_trainset_y.csv',dtype=float)
trainset_rare_y = np.loadtxt('input/baseline7_rare_trainset_y.csv',dtype=float)
testset_fluent_y = np.loadtxt('input/baseline7_fluent_testset_y.csv',dtype=float)
testset_rare_y = np.loadtxt('input/baseline7_rare_testset_y.csv',dtype=float)
trainset_fluent_y = trainset_fluent_y.reshape((31500,1))
testset_fluent_y = testset_fluent_y.reshape((10500,1))
trainset_rare_y = trainset_rare_y.reshape((31500,1))
testset_rare_y = testset_rare_y.reshape((10500,1))


trainset_x_add = encoder.transform(submission_x_0.iloc[:,[2,8,9,13,14]].values).toarray()
submission_x_0 = submission_x_0.iloc[:,[3,4,5,6,7,10,11,12]].values
submission_x_0 = np.hstack((submission_x_0,trainset_x_add))
print submission_x_0.shape
del trainset_x_add

trainset_x_add = encoder.transform(submission_x_1.iloc[:,[2,8,9,13,14]].values).toarray()
submission_x_1 = submission_x_1.iloc[:,[3,4,5,6,7,10,11,12]].values
submission_x_1 = np.hstack((submission_x_1,trainset_x_add))
print submission_x_1.shape
del trainset_x_add

trainset_x_add = encoder.transform(submission_x_2.iloc[:,[2,8,9,13,14]].values).toarray()
submission_x_2 = submission_x_2.iloc[:,[3,4,5,6,7,10,11,12]].values
submission_x_2 = np.hstack((submission_x_2,trainset_x_add))
print submission_x_2.shape
del trainset_x_add

trainset_x_add = encoder.transform(submission_x_3.iloc[:,[2,8,9,13,14]].values).toarray()
submission_x_3 = submission_x_3.iloc[:,[3,4,5,6,7,10,11,12]].values
submission_x_3 = np.hstack((submission_x_3,trainset_x_add))
print submission_x_3.shape
del trainset_x_add

submission_x = [submission_x_0,submission_x_1,submission_x_2,submission_x_3]
print submission_x

# ###############################################################################
# # train xgboost (each cluster a GBDT tree)
# ###############################################################################
score_list = []

for cluster in range(4):

    # find train set and test set for each
    trainset_cluster = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
    testset_cluster = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)
    total_cluster =  np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)

    trainset_indice = []
    for i in range(trainset_cluster.size):
        trainset_indice.extend(np.nonzero(trainset == (trainset_cluster[i]-1))[0].tolist())
    print len(trainset_indice)
    trainset_cluster = []
    for i in trainset_indice:
        trainset_cluster.extend((np.ones(21,dtype=int)*i*21+np.arange(21,dtype=int)).tolist())
    print trainset_cluster
    trainset_indice = trainset_cluster

    testset_indice = []
    for i in range(testset_cluster.size):
        testset_indice.extend(np.nonzero(testset == (testset_cluster[i]-1))[0].tolist())
    print len(testset_indice)
    testset_cluster = []
    for i in testset_indice:
        testset_cluster.extend((np.ones(21,dtype=int)*i*21+np.arange(21,dtype=int)).tolist())
    print testset_cluster
    testset_indice = testset_cluster

    # train fluent part

    regressor_fluent = xgb.XGBRegressor(max_depth=6,n_estimators=300,min_child_weight=0.9)
    print trainset_x[trainset_indice,:].shape
    print trainset_fluent_y[trainset_indice].shape

    print "Building fluent part model.."
    regressor_fluent.fit(trainset_x[trainset_indice,:],trainset_fluent_y[trainset_indice])
    print("Predicting..")
    preds_fluent = regressor_fluent.predict(testset_x[testset_indice,:])
    print preds_fluent
    preds_fluent = preds_fluent.reshape(-1,21)
    preds_fluent = preds_fluent[:,-14:]
    print preds_fluent.shape

    # train rare part

    regressor_rare = xgb.XGBRegressor(max_depth=6, n_estimators=300, min_child_weight=0.9)
    print trainset_x[trainset_indice, :].shape
    print trainset_rare_y[trainset_indice].shape

    print "Building rare part model.."
    regressor_rare.fit(trainset_x[trainset_indice, :], trainset_rare_y[trainset_indice])
    print("Predicting..")
    preds_rare = regressor_rare.predict(testset_x[testset_indice, :])
    print preds_rare
    preds_rare = preds_rare.reshape(-1, 21)
    preds_rare = preds_rare[:, -14:]
    print preds_rare.shape

    # generating test set file

    stander = preprocessing.StandardScaler()
    counter = 0
    cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)

    result_y = np.empty((0, 14))
    testset_y = np.empty((0, 14))

    # find train set and test set for each
    trainset_cluster = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
    testset_cluster = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)

    # fluent part restore
    for i in testset_cluster:
        if cluster==2 or cluster==3:

            # part fluent
            info = pd.read_csv('flow_per_shop/' + str(i) + '_fluent.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_fluent = info['count'].values
            ts_fluent = ts_fluent[243:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  #  use odd
            RM_ts_fluent = pd.rolling_mean(ts_fluent,size,center=True)
            RM_ts_fluent[:size/2] = RM_ts_fluent[size/2]
            RM_ts_fluent[-(size/2)-1:] = RM_ts_fluent[-(size/2)-1]
            WAVE_ts_fluent = ts_fluent - RM_ts_fluent

            ts_len = RM_ts_fluent.shape[0] - 14
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_fluent[:-14])
            prediction_fluent_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(preds_fluent[counter])
            prediction_fluent = prediction_fluent_trend+np.round(stander.fit(WAVE_ts_fluent[:-14]).inverse_transform(value))
            prediction_fluent[prediction_fluent<0]=0

            # part rare
            info = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_rare = info['count'].values
            ts_rare = ts_rare[243:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  #  use odd
            RM_ts_rare = pd.rolling_mean(ts_rare,size,center=True)
            RM_ts_rare[:size/2] = RM_ts_rare[size/2]
            RM_ts_rare[-(size/2)-1:] = RM_ts_rare[-(size/2)-1]
            WAVE_ts_rare = ts_rare - RM_ts_rare

            ts_len = RM_ts_rare.shape[0] - 14
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_rare[:-14])
            prediction_rare_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(preds_rare[counter])
            prediction_rare = prediction_rare_trend+np.round(stander.fit(WAVE_ts_rare[:-14]).inverse_transform(value))
            prediction_rare[prediction_rare<0]=0

            # real time series
            info = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
            ts = info['count'].values
            counter += 1
            testset_y = np.vstack((testset_y, ts[-14:]))
            prediction_y = prediction_rare+prediction_fluent
            result_y = np.vstack((result_y,prediction_y))

        if cluster==1:

            # part fluent
            info = pd.read_csv('flow_per_shop/' + str(i) + '_fluent.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_fluent = info['count'].values
            ts_fluent = ts_fluent[390:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  # use odd
            RM_ts_fluent = pd.rolling_mean(ts_fluent, size, center=True)
            RM_ts_fluent[:size / 2] = RM_ts_fluent[size / 2]
            RM_ts_fluent[-(size / 2) - 1:] = RM_ts_fluent[-(size / 2) - 1]
            WAVE_ts_fluent = ts_fluent - RM_ts_fluent

            ts_len = RM_ts_fluent.shape[0] - 14
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_fluent[:-14])
            prediction_fluent_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(preds_fluent[counter])
            prediction_fluent = prediction_fluent_trend + np.round(
                stander.fit(WAVE_ts_fluent[:-14]).inverse_transform(value))
            prediction_fluent[prediction_fluent < 0] = 0

            # part rare
            info = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_rare = info['count'].values
            ts_rare = ts_rare[390:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  # use odd
            RM_ts_rare = pd.rolling_mean(ts_rare, size, center=True)
            RM_ts_rare[:size / 2] = RM_ts_rare[size / 2]
            RM_ts_rare[-(size / 2) - 1:] = RM_ts_rare[-(size / 2) - 1]
            WAVE_ts_rare = ts_rare - RM_ts_rare

            ts_len = RM_ts_rare.shape[0] - 14
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_rare[:-14])
            prediction_rare_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(preds_rare[counter])
            prediction_rare = prediction_rare_trend + np.round(stander.fit(WAVE_ts_rare[:-14]).inverse_transform(value))
            prediction_rare[prediction_rare < 0] = 0

            # real time series
            info = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
            ts = info['count'].values
            counter += 1
            testset_y = np.vstack((testset_y, ts[-14:]))
            prediction_y = prediction_rare + prediction_fluent
            result_y = np.vstack((result_y,prediction_y))

        if cluster==0:
            # part fluent
            info = pd.read_csv('flow_per_shop/' + str(i) + '_fluent.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_fluent = info['count'].values
            ts_fluent = ts_fluent[432:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  # use odd
            RM_ts_fluent = pd.rolling_mean(ts_fluent, size, center=True)
            RM_ts_fluent[:size / 2] = RM_ts_fluent[size / 2]
            RM_ts_fluent[-(size / 2) - 1:] = RM_ts_fluent[-(size / 2) - 1]
            WAVE_ts_fluent = ts_fluent - RM_ts_fluent

            ts_len = RM_ts_fluent.shape[0] - 14
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=1)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_fluent[:-14])
            prediction_fluent_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(preds_fluent[counter])
            prediction_fluent = prediction_fluent_trend + np.round(
                stander.fit(WAVE_ts_fluent[:-14]).inverse_transform(value))
            prediction_fluent[prediction_fluent < 0] = 0

            # part rare
            info = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_rare = info['count'].values
            ts_rare = ts_rare[432:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  # use odd
            RM_ts_rare = pd.rolling_mean(ts_rare, size, center=True)
            RM_ts_rare[:size / 2] = RM_ts_rare[size / 2]
            RM_ts_rare[-(size / 2) - 1:] = RM_ts_rare[-(size / 2) - 1]
            WAVE_ts_rare = ts_rare - RM_ts_rare

            ts_len = RM_ts_rare.shape[0] - 14
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=1)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_rare[:-14])
            prediction_rare_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(preds_rare[counter])
            prediction_rare = prediction_rare_trend + np.round(stander.fit(WAVE_ts_rare[:-14]).inverse_transform(value))
            prediction_rare[prediction_rare < 0] = 0

            # real time series
            info = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
            ts = info['count'].values
            counter += 1
            testset_y = np.vstack((testset_y, ts[-14:]))
            prediction_y = prediction_rare + prediction_fluent
            result_y = np.vstack((result_y, prediction_y))

            # pyplot.figure()
            # pyplot.plot(ts)
            # pyplot.plot(np.arange(ts.size-14,ts.size),prediction_y)
            # pyplot.show()

    print testset_y
    print result_y
    np.savetxt('test_set/baseline_7_clus_' + str(cluster) + '_label.csv', testset_y, fmt='%d',delimiter=',')
    np.savetxt('test_set/baseline_7_clus_' + str(cluster) + '_predict.csv', result_y, fmt='%d',delimiter=',')

    # test set scoring
    sum = 0.
    for i in range(testset_y.shape[0]):
        for j in range(testset_y.shape[1]):
            sum += np.absolute(
                (result_y[i, j] - testset_y[i, j]) / (result_y[i, j] + testset_y[i, j] + 0.000000001))
    nt = float((testset_y.shape[0] * testset_y.shape[1]))
    score = sum / nt
    print score
    score_list.append(score)

    # generating submission set
    prediction_fluent = regressor_fluent.predict(submission_x[cluster])
    prediction_fluent = prediction_fluent.reshape(-1,14)

    prediction_rare = regressor_rare.predict(submission_x[cluster])
    prediction_rare = prediction_rare.reshape(-1,14)

    result_y = np.empty((0,14),dtype=int)
    counter = 0
    for i in total_cluster:
        if cluster == 2 or cluster == 3:
            # part fluent
            info = pd.read_csv('flow_per_shop/' + str(i) + '_fluent.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_fluent = info['count'].values
            ts_fluent = ts_fluent[243:]

            size = 7  # use odd
            RM_ts_fluent = pd.rolling_mean(ts_fluent, size, center=True)
            RM_ts_fluent[:size / 2] = RM_ts_fluent[size / 2]
            RM_ts_fluent[-(size / 2) - 1:] = RM_ts_fluent[-(size / 2) - 1]
            WAVE_ts_fluent = ts_fluent - RM_ts_fluent

            ts_len = RM_ts_fluent.shape[0]
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_fluent)
            prediction_fluent_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(prediction_fluent[counter])
            prediction_fluent_ts = prediction_fluent_trend + np.round(
                stander.fit(WAVE_ts_fluent).inverse_transform(value))
            prediction_fluent_ts[prediction_fluent_ts < 0] = 0

            # part rare
            info = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_rare = info['count'].values
            ts_rare = ts_rare[243:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  # use odd
            RM_ts_rare = pd.rolling_mean(ts_rare, size, center=True)
            RM_ts_rare[:size / 2] = RM_ts_rare[size / 2]
            RM_ts_rare[-(size / 2) - 1:] = RM_ts_rare[-(size / 2) - 1]
            WAVE_ts_rare = ts_rare - RM_ts_rare

            ts_len = RM_ts_rare.shape[0]
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_rare)
            prediction_rare_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(prediction_rare[counter])
            prediction_rare_ts = prediction_rare_trend + np.round(stander.fit(WAVE_ts_rare).inverse_transform(value))
            prediction_rare_ts[prediction_rare_ts < 0] = 0
            counter += 1
            prediction_y = prediction_rare_ts + prediction_fluent_ts
            result_y = np.vstack((result_y, prediction_y))

        if cluster == 1:
            # part fluent
            info = pd.read_csv('flow_per_shop/' + str(i) + '_fluent.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_fluent = info['count'].values
            ts_fluent = ts_fluent[390:]

            size = 7  # use odd
            RM_ts_fluent = pd.rolling_mean(ts_fluent, size, center=True)
            RM_ts_fluent[:size / 2] = RM_ts_fluent[size / 2]
            RM_ts_fluent[-(size / 2) - 1:] = RM_ts_fluent[-(size / 2) - 1]
            WAVE_ts_fluent = ts_fluent - RM_ts_fluent

            ts_len = RM_ts_fluent.shape[0]
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_fluent)
            prediction_fluent_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(prediction_fluent[counter])
            prediction_fluent_ts = prediction_fluent_trend + np.round(
                stander.fit(WAVE_ts_fluent).inverse_transform(value))
            prediction_fluent_ts[prediction_fluent_ts < 0] = 0

            # part rare
            info = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_rare = info['count'].values
            ts_rare = ts_rare[390:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  # use odd
            RM_ts_rare = pd.rolling_mean(ts_rare, size, center=True)
            RM_ts_rare[:size / 2] = RM_ts_rare[size / 2]
            RM_ts_rare[-(size / 2) - 1:] = RM_ts_rare[-(size / 2) - 1]
            WAVE_ts_rare = ts_rare - RM_ts_rare

            ts_len = RM_ts_rare.shape[0]
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_rare)
            prediction_rare_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(prediction_rare[counter])
            prediction_rare_ts = prediction_rare_trend + np.round(stander.fit(WAVE_ts_rare).inverse_transform(value))
            prediction_rare_ts[prediction_rare_ts < 0] = 0
            counter += 1
            prediction_y = prediction_rare_ts + prediction_fluent_ts
            result_y = np.vstack((result_y, prediction_y))

        if cluster == 0:
            # part fluent
            info = pd.read_csv('flow_per_shop/' + str(i) + '_fluent.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_fluent = info['count'].values
            ts_fluent = ts_fluent[432:]

            size = 7  # use odd
            RM_ts_fluent = pd.rolling_mean(ts_fluent, size, center=True)
            RM_ts_fluent[:size / 2] = RM_ts_fluent[size / 2]
            RM_ts_fluent[-(size / 2) - 1:] = RM_ts_fluent[-(size / 2) - 1]
            WAVE_ts_fluent = ts_fluent - RM_ts_fluent

            ts_len = RM_ts_fluent.shape[0]
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=1)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_fluent)
            prediction_fluent_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(prediction_fluent[counter])
            prediction_fluent_ts = prediction_fluent_trend + np.round(
                stander.fit(WAVE_ts_fluent).inverse_transform(value))
            prediction_fluent_ts[prediction_fluent_ts < 0] = 0

            # part rare
            info = pd.read_csv('flow_per_shop/' + str(i) + '_rare.csv')
            info.drop(info.index[[348, 349, 350, 351, 352, 353, 354, 446, 447, 448, 449, 450, 451, 452,
                                  404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473]], inplace=True)
            ts_rare = info['count'].values
            ts_rare = ts_rare[432:]

            # data is line to fit
            # 1,find var 2,find mean 3,inverse transform
            size = 7  # use odd
            RM_ts_rare = pd.rolling_mean(ts_rare, size, center=True)
            RM_ts_rare[:size / 2] = RM_ts_rare[size / 2]
            RM_ts_rare[-(size / 2) - 1:] = RM_ts_rare[-(size / 2) - 1]
            WAVE_ts_rare = ts_rare - RM_ts_rare

            ts_len = RM_ts_rare.shape[0]
            x = np.arange(ts_len) + 1
            model = Pipeline([('poly', PolynomialFeatures(degree=1)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x[:, np.newaxis], RM_ts_rare)
            prediction_rare_trend = model.predict(np.arange(ts_len, ts_len + 14)[:, np.newaxis])

            value = stander.fit_transform(prediction_rare[counter])
            prediction_rare_ts = prediction_rare_trend + np.round(stander.fit(WAVE_ts_rare).inverse_transform(value))
            prediction_rare_ts[prediction_rare_ts < 0] = 0
            counter += 1
            prediction_y = prediction_rare_ts + prediction_fluent_ts
            result_y = np.vstack((result_y, prediction_y))

    np.savetxt('submission/baseline_7_clus_'+str(cluster)+'_predict.csv',result_y,fmt='%d',delimiter=',')
print score_list
