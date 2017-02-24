# this file is to clustering

import sklearn.cluster
import sklearn.preprocessing
import sklearn.cluster
import pandas as pd
import numpy as np
from matplotlib import pyplot

labels = []
pattern = np.loadtxt('classification/pattern.csv')

counter = 0
zero_count = 0

# get 4 classification here

# fist class select these who have cliff in recent 3 month
# 2 nd class select these not long enough but with complete information for recent 3 month
# 3 th class full information after chinese new year and more custom flow during weekends
# 4 th calss full information after chinese new year and more custom flow during workday

for pattern_i in pattern:

    file = 'flow_per_shop/'+str(counter+1)+'.csv'
    info = pd.read_csv(file)
    ts = info['count'].values
    ts = ts[242:]
    if (120-np.count_nonzero(ts[-120:]))>0:
        labels.append(0)
        counter += 1
        continue
    if np.count_nonzero(ts)!=ts.size:
        labels.append(1)
        counter += 1
        continue

    if pattern_i < 1:
        labels.append(2)
    else:
        labels.append(3)
    counter+=1

np.savetxt('labels.csv',labels,fmt='%d')
print zero_count
