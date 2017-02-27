import numpy as np
import pandas as pd
from pypinyin import lazy_pinyin
from sklearn import preprocessing
import xgboost as xgb
from sklearn import metrics
from matplotlib import pyplot
import seaborn

num_cluster = 4

# load train test set
trainset = np.loadtxt('trainset.csv',dtype=int) -1
testset = np.loadtxt('testset.csv',dtype=int) -1

# load shop info
shop_info = pd.read_csv('input/shop_info.txt')
def change_2_pinyin(location):
    city_name = []
    for item in lazy_pinyin(unicode(location,encoding='utf-8')):
        city_name.extend(item.encode())
    city_name = ''.join(city_name)
    return city_name

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
# save train set
######################################################################################
#
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
#
# ###########################################################################################
# # save test set
# ############################################################################################
#
# stander = preprocessing.StandardScaler()
# testset_x = np.empty((0,14))
# testset_y = np.empty((0,21))
#
# for item in range(500):
#     print item
#     shop_id = testset_shop_info.iloc[item,0]
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
#     testset_y = np.vstack((testset_y,np.hstack((week1,np.hstack((week2,week3))))))
#     city = testset_shop_info.iloc[item,1]
#     weather = pd.read_csv('input/city_weather/'+city,names=['date','max_t','min_t','class','wind_orientation','wind_class'])
#     weather = weather.iloc[-51:-30,:]
#     weather['class'] = weather['class'].apply(change_2_pinyin)
#     del weather['wind_orientation']
#     del weather['wind_class']
#     weather['day'] = np.array([2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1])
#     del weather['date']
#     info = testset_shop_info.iloc[item,:]
#     info = info.values
#     weather = weather.values
#     for day in weather:
#         day_info = np.hstack((info,day))
#         testset_x = np.vstack((testset_x,day_info))
#
# testset_x = pd.DataFrame(testset_x)
# testset_x.to_csv('input/baseline7_testset_x.csv')
# np.savetxt('input/baseline7_testset_y.csv',testset_y,fmt='%f')
#
# ############################################################################
# # generating submission file
# ############################################################################
#
# for cluster in range(num_cluster):
#     cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)
#
#     submission_x = np.empty((0,14))
#     for item in cluster_file:
#         print item
#         shop_id = shop_info.iloc[item-1,0]
#         city = shop_info.iloc[item-1,1]
#         weather = pd.read_csv('input/city_weather/'+city,names=['date','max_t','min_t','class','wind_orientation','wind_class'])
#         weather = weather.iloc[-30:-16,:]
#         weather['class'] = weather['class'].apply(change_2_pinyin)
#         del weather['wind_orientation']
#         del weather['wind_class']
#         weather['day'] = np.array([2,3,4,5,6,7,1,2,3,4,5,6,7,1])
#         del weather['date']
#         info = shop_info.iloc[item-1,:]
#         info = info.values
#         weather = weather.values
#         for day in weather:
#             day_info = np.hstack((info,day))
#             submission_x = np.vstack((submission_x,day_info))
#
#     submission_x = pd.DataFrame(submission_x)
#     submission_x.to_csv('submission/baseline7_submission_x_'+str(cluster)+'.csv')

###############################################################################
# label encoder
###############################################################################

trainset_x = pd.read_csv('input/baseline7_trainset_x.csv')
testset_x = pd.read_csv('input/baseline7_testset_x.csv')
totalset = pd.concat([trainset_x,testset_x])
print totalset.shape
print totalset

trainset_x = pd.read_csv('input/baseline7_trainset_x.csv')
testset_x = pd.read_csv('input/baseline7_testset_x.csv')

for item in range(4):
    with open('submission/baseline7_submission_x_'+str(item)+'.csv', 'r') as file :
        filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('xiaodaozhongxue', 'xiaoyu')
    filedata = filedata.replace('xiaoxue', 'xiaoyu')
    filedata = filedata.replace('zhongdaodaxue', 'xiaoyu')

    # Write the file out again
    with open('submission/baseline7_submission_x_'+str(item)+'.csv', 'w') as file:
        file.write(filedata)


submission_x_0 = pd.read_csv('submission/baseline7_submission_x_'+str(0)+'.csv')
submission_x_1 = pd.read_csv('submission/baseline7_submission_x_'+str(1)+'.csv')
submission_x_2 = pd.read_csv('submission/baseline7_submission_x_'+str(2)+'.csv')
submission_x_3 = pd.read_csv('submission/baseline7_submission_x_'+str(3)+'.csv')

totalset = pd.concat([trainset_x,testset_x])
print totalset.shape
print totalset

print '#######################################################'
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

trainset_y = np.loadtxt('input/baseline7_trainset_y.csv',dtype=float)
testset_y = np.loadtxt('input/baseline7_testset_y.csv',dtype=float)
trainset_y = trainset_y.reshape((31500,1))
testset_y = testset_y.reshape((10500,1))
print trainset_y.shape
print testset_y.shape

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
###############################################################################
# train xgboost (each cluster a GBDT tree)
###############################################################################
score_list = []

for cluster in range(num_cluster):

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


    # define trainer
    regressor = xgb.XGBRegressor(max_depth=6,n_estimators=300,min_child_weight=0.9)

    # train
    print trainset_x[trainset_indice,:].shape
    print trainset_y[trainset_indice].shape
    print "Building model.."
    regressor.fit(trainset_x[trainset_indice,:],trainset_y[trainset_indice])

    print("Predicting..")

    preds = regressor.predict(testset_x[testset_indice,:])

    print preds
    preds = preds.reshape(-1,21)
    preds = preds[:,-14:]
    print preds.shape

    # generating test set file
    stander = preprocessing.StandardScaler()
    counter = 0
    cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)

    result_y = np.empty((0, 14))
    testset_y = np.empty((0, 14))

    # find train set and test set for each
    trainset_cluster = np.loadtxt('classification/cluster_'+str(cluster)+'_trainset.csv',dtype=int)
    testset_cluster = np.loadtxt('classification/cluster_'+str(cluster)+'_testset.csv',dtype=int)

    for i in testset_cluster:
        data = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
        data = data['count'].values
        value = stander.fit_transform(preds[counter])
        result_y = np.vstack((result_y, np.round(stander.fit(data[-14-7:-14]).inverse_transform(value))))
        counter += 1
        testset_y = np.vstack((testset_y, data[-14:]))

    print testset_y
    print result_y
    np.savetxt('test_set/baseline_7_clus_' + str(cluster) + '_label.csv', testset_y, fmt='%d')
    np.savetxt('test_set/baseline_7_clus_' + str(cluster) + '_predict.csv', result_y, fmt='%d')

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
    preds = regressor.predict(submission_x[cluster])
    print preds
    preds = preds.reshape(-1,14)
    print preds

    result_y = np.empty((0,14),dtype=int)
    counter = 0
    for i in total_cluster:
        data = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
        data = data['count'].values
        value = stander.fit_transform(preds[counter])
        result_y = np.vstack((result_y, np.round(stander.fit(data[-21:]).inverse_transform(value))))
        counter += 1
    np.savetxt('submission/baseline_7_clus_'+str(cluster)+'_predict.csv',result_y,fmt='%d')
print score_list
