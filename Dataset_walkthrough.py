#   this  file is to look dataset information

import pandas as pd
from matplotlib import pyplot
import seaborn
import numpy as np

shop_info = pd.read_csv('input/shop_info.txt')

for i in range(100,2000):
    print i+1

    file = 'flow_per_shop/'+str(i+1)+'.csv'
    info = pd.read_csv(file)
    ts = info['count'].values
    if np.mean(ts[90:160])<50:
        continue
    print 'mean 2 week before ' + str(np.mean(ts[115:129]))
    print 'mean 11,11 ' + str(np.mean(ts[129:143]))
    print 'mean 2 week after ' + str(np.mean(ts[143:157]))

    location = shop_info.iloc[i,1]
    title1 = shop_info.iloc[i,7]
    title2 = shop_info.iloc[i,8]
    title3 = shop_info.iloc[i,9]

    print location
    print title1
    print title2
    print title3

    pyplot.figure(figsize=(18,8))
    pyplot.subplot(1,2,1)
    pyplot.plot(ts)
    pyplot.axvline(98,color='brown')
    pyplot.axvline(170,color='red')
    pyplot.axvline(227,color='yellow')
    pyplot.axvline(311,color='green')
    pyplot.axvline(448,color='green')
    pyplot.axvline(464,color='brown')
    pyplot.axvspan(129,142, facecolor='0.5', alpha=0.5)
    pyplot.legend(['ts','10.1','1212','spring D','5.1','Moon','10.1','1111'])

    file = 'view_per_shop/' + str(i + 1) + '.csv'
    info = pd.read_csv(file)
    ts = info['count'].values
    pyplot.subplot(1,2,2)
    pyplot.plot(ts)
    pyplot.axvline(106,color='green')
    pyplot.axvline(122,color='brown')
    pyplot.show()