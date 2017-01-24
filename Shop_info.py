# a walkthrough for shop information
print __doc__
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

file = 'input/shop_info.txt'
info = pd.read_csv(file)

# city name
print info.groupby(by='city_name')['shop_id'].count()

# location id
location_id = info['location_id'].values
plt.figure()
plt.title('location_id')
sns.distplot(location_id)
plt.show()

# per_pay
per_pay = info['per_pay'].values
plt.figure()
plt.title('per_pay')
sns.distplot(per_pay)
plt.show()

# shop_level
shop_level = info['shop_level'].values
plt.figure()
plt.title('shop_level')
sns.distplot(shop_level)
plt.show()

# shop type
print info.groupby(by=['cate_1_name','cate_2_name'])['shop_id'].count()


