# this file is to add label

import pandas as pd
import numpy as np
import csv

file_user_pay = 'input/user_pay.txt'
file = open(file_user_pay,'rb')
file_reader = csv.reader(file,delimiter=',')
file_reader.next()

buffer_csv = np.empty((0,3))
row_num = '1862'

def save_label(buffer_csv,row_num,count):
    print 'handle '+str(count)+' th label : '+str(row_num)
    info = csv_buffer_pretreatment(buffer_csv)
    target = build_target()
    info = pd.merge(target,info,how='left',on=['year','month','day'])
    target['count'] = info.groupby(by=['year','month','day'])['user_id'].count().values
    target.to_csv('flow_per_shop/'+str(row_num)+'.csv')

def csv_buffer_pretreatment(buffer_csv):
    # buffer_csv pretreatment
    shop_id = str(buffer_csv[0:1])
    info = pd.DataFrame(buffer_csv,columns=['user_id','shop_id','time_stamp'])
    info['user_id']=info['user_id'].astype(np.int64)
    info['shop_id']=info['shop_id'].astype(np.int64)
    info['time_stamp']=info['time_stamp'].astype(str)
    time_value = info['time_stamp'].values
    year = []
    month = []
    day = []
    hour = []
    for row in time_value:
        year.append(row[0:4])
        month.append(row[5:7])
        day.append(row[8:10])
        hour.append(row[11:13])
    info['year'] = pd.Series(year,dtype=np.int16)
    info['month'] = pd.Series(month,dtype=np.int8)
    info['day'] = pd.Series(day,dtype=np.int8)
    info['hour'] = pd.Series(hour,dtype=np.int8)
    return info

def build_target():
    # build label target
    start = pd.Timestamp('2015-06-25')
    end = pd.Timestamp('2016-10-31')
    dates = pd.date_range(start, end, freq='D')
    target = pd.DataFrame({'target_dates':dates,
                           'count':np.zeros(len(dates),dtype=np.int64)})
    target['target_dates']=target['target_dates'].astype(str)
    time_value = target['target_dates'].values
    year = []
    month = []
    day = []
    for row in time_value:
        year.append(row[0:4])
        month.append(row[5:7])
        day.append(row[8:10])
    target['year'] = pd.Series(year, dtype=np.int16)
    target['month'] = pd.Series(month, dtype=np.int8)
    target['day'] = pd.Series(day, dtype=np.int8)
    return target

#  labeling   row by row
count=0
for row in file_reader:
    if row[1] == row_num:
        buffer_csv = np.vstack((buffer_csv,row))
    else:
        count+=1
        save_label(buffer_csv,row_num,count)
        row_num = row[1]
        buffer_csv = np.empty((0, 3))
        buffer_csv = np.vstack((buffer_csv,row))
save_label(buffer_csv,row_num,count)



