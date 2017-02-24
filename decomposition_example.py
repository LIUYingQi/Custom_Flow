# this file try to show how decomposite ts
import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn

# load train set and test set
cluster = 3
example = 135

cluster_file = np.loadtxt('classification/cluster_'+str(cluster)+'.csv',dtype=int)

print cluster_file[example]
file = 'flow_per_shop/'+str(cluster_file[example])+'.csv'
info = pd.read_csv(file)
info.drop(info.index[[348,349,350,351,352,353,354,446,447,448,449,450,451,452,
                      411,412,413,414,415,416,417,
                      460,461,462,463,464,465,466,467,468,469,470,471,472,473]],inplace=True)
date = info['target_dates'].values
print date
ts = info['count'].values
ts = ts[243:]
print ts
pyplot.figure()
pyplot.plot(ts)
pyplot.show()