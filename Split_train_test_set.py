# this file is to define train set and test set

import numpy as np

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