# this file is to define train set and test set
# do labeling for cluster first before this

import numpy as np

# split train set and test set
trainset = np.arange(1,2001)
trainset = np.random.choice(trainset,1500,False)
trainset = np.sort(trainset)
trainset = trainset.tolist()
testset = []
for i in range(1,2001):
    if i not in trainset:
        testset.append(i)
print trainset
print testset
trainset = np.array(trainset,dtype=np.int32)
testset = np.array(testset,dtype=np.int32)

with open('trainset.csv','wb') as trainset_file:
    np.savetxt(trainset_file,trainset,fmt='%d')

with open('testset.csv','wb') as test_file:
    np.savetxt(test_file,testset,fmt='%d')

# load labels
labels = np.loadtxt('labels.csv',dtype=int)
print labels
cluster_num = np.max(labels)+1

# generating trainset and testset for each clustring
for i in range(cluster_num):
    print i
    indice = np.where(labels == i)[0]+1
    cluster_trainset = np.intersect1d(indice,trainset)
    cluster_testset = np.intersect1d(indice,testset)
    print indice
    print cluster_trainset
    print cluster_testset
    print ''
    np.savetxt('classification/cluster_'+str(i)+'_trainset.csv',cluster_trainset,fmt='%d')
    np.savetxt('classification/cluster_'+str(i)+'_testset.csv',cluster_testset,fmt='%d')
