#   this  file is to look dataset information

import pandas as pd
from matplotlib import pyplot
import seaborn

shop_info = pd.read_csv('input/shop_info.txt')

for i in range(1000,2000):
    print i+1
    file = 'flow_per_shop/'+str(i+1)+'.csv'
    info = pd.read_csv(file)
    ts = info['count'].values
    location = shop_info.iloc[i,1]
    title1 = shop_info.iloc[i,7]
    title2 = shop_info.iloc[i,8]
    print location
    print title1
    print title2
    pyplot.figure(figsize=(10,8))
    pyplot.plot(ts)
    pyplot.axvline(98,color='green')
    pyplot.axvline(110,color='red')
    pyplot.axvline(239,color='yellow')
    pyplot.axvline(464,color='green')
    pyplot.show()
