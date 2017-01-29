# this file is to add and view feature of holiday and workday influence for each shop

import numpy as np
import pandas as pd
from matplotlib import pyplot

# find holidays
holidays = ['2015-09-26','2015-10-01','2015-10-02','2015-10-03','2015-10-04','2015-10-05',
            '2015-10-06','2015-10-07','2016-01-01','2016-01-02','2016-01-03',
            '2016-02-07','2016-02-08','2016-02-09','2016-02-10','2016-02-11',
            '2016-02-12','2016-02-13','2016-04-02','2016-04-03','2016-04-04',
            '2016-04-30','2016-05-01','2016-05-02','2016-06-09','2016-06-10',
            '2016-06-11','2016-09-15','2016-09-16','2016-09-17','2016-10-01',
            '2016-10-02','2016-10-03','2016-10-04','2016-10-05','2016-10-06',
            '2016-10-07']

exchangedays = ['2015-10-11','2015-10-10','2016-02-06','2016-02-14','2016-06-12','2016-09-18','2016-10-08',
                '2016-10-09']

def is_holiday(date):
    if date in holidays:
        return True
    else:
        return False
def is_exchangeday(date):
    if date in exchangedays:
        return True
    else:
        return False

not_work_days= []
file_name = 'flow_per_shop/1000.csv'
shop_file = open(file_name,'rb')
info = pd.read_csv(shop_file)
info['target_dates'] = info['target_dates'].astype(str)
dates = info['target_dates'].values
flow = info['count'].values
count = 0
for i in dates:
    print i
    print count
    count +=1
    print ((not np.is_busday(np.datetime64(i)) and not is_exchangeday(i)) or is_holiday(i))
    print ''
    not_work_days.append(((not np.is_busday(np.datetime64(i)) and not is_exchangeday(i)) or is_holiday(i)))
not_work_days = np.array(not_work_days,dtype=int)
pyplot.figure()
pyplot.axis([0,500,0,2])
pyplot.plot(not_work_days)
pyplot.show()

total_not_workday = not_work_days.sum()
print total_not_workday
total_workday = not_work_days.shape[0] - total_not_workday
print total_workday

# show how does it change the customer flow for hoilidays (not work days)
flow_not_work_days = flow * not_work_days
print flow_not_work_days.sum()/total_not_workday

flow_work_days = flow * (1-not_work_days)
print flow_work_days.sum()/total_workday