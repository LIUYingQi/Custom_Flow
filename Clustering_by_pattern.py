# this file is to clustering (may be not work)

import sklearn.cluster
import sklearn.preprocessing
import sklearn.cluster
import pandas as pd
import numpy as np
from matplotlib import pyplot

labels = []
pattern = np.loadtxt('classification/pattern.csv')

for i in pattern:
    if i < 0.8:
        labels.append(0)
    elif i > 1.2:
        labels.append(2)
    else:
        labels.append(1)

np.savetxt('labels.csv',labels,fmt='%d')

