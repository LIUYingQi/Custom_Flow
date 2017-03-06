# this file is to check submission file

import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn
from pypinyin import lazy_pinyin

submission = np.loadtxt('submmission.csv',dtype=int,delimiter=',')
submission[submission<0]=0

submission_1 = np.loadtxt('submmission_1.csv',dtype=int,delimiter=',')
submission_1[submission_1<0]=0

shop_info = pd.read_csv('input/shop_info.txt')
def change_2_pinyin(location):
    city_name = []
    for item in lazy_pinyin(unicode(location,encoding='utf-8')):
        city_name.extend(item.encode())
    city_name = ''.join(city_name)
    return city_name

# for i in range(2000):
#     print i+1
#     title3 = str(shop_info.iloc[i,9])
#     print title3
#     if change_2_pinyin(title3)=='xishikuaican':
#         submission[i,1:] = np.round(submission[i,1:]*1.3)

submission = submission.astype(int)
np.savetxt('submmission.csv',submission,fmt='%d',delimiter=',')
np.savetxt('submmission_1.csv',submission_1,fmt='%d',delimiter=',')

for i in range(2000):
    print np.mean(submission[i,1:])
    print np.mean(submission_1[i,1:])
    print '##'

print np.mean(submission[:,1:])
print np.mean(submission_1[:,1:])
print np.var(submission[:,1:])
print np.var(submission_1[:,1:])

for i in range(1525,2000):
    print i+1
    file = 'flow_per_shop/'+str(i+1)+'.csv'
    info = pd.read_csv(file)
    ts = info['count'].values
    location = shop_info.iloc[i,1]
    title1 = shop_info.iloc[i,7]
    title2 = shop_info.iloc[i,8]
    title3 = shop_info.iloc[i,9]

    print location
    print title1
    print title2
    print title3

    pyplot.figure(figsize=(15,8))
    pyplot.plot(np.arange(0,495),ts)
    pyplot.plot(np.arange(495,509),submission[i,1:],color='green')
    # pyplot.plot(np.arange(495,509),submission_1[i,1:],color='brown')
    pyplot.axvline(98,color='brown')
    pyplot.axvline(170,color='red')
    pyplot.axvline(227,color='yellow')
    pyplot.axvline(311,color='green')
    pyplot.axvline(448,color='green')
    pyplot.axvline(464,color='brown')
    pyplot.axvspan(129,142, facecolor='0.5', alpha=0.5)
    pyplot.legend(['ts','10.1','1212','spring D','5.1','Moon','10.1','1111'])
    pyplot.show()
