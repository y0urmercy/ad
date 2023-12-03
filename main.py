import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ecom_yl.csv", sep=',', encoding='utf8', skipinitialspace=False)
for col in df.columns:
    new_col = ''.join(['_' if ch == ' ' else ch.lower() for ch in col])
    df.rename(columns={col: new_col[:]}, inplace=True)
#убрали записи, в которых не указан канал привлечения, т.к. не несет полезной информации для исследования.
df = df.dropna(subset=['region', 'device', 'channel'])
#дата
date_columns = ['session_start', 'session_end', 'session_date', 'order_dt']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])
# дубликаты
df = df.drop_duplicates()

def cat_reg(x):
    if x[0] == 'U':
        return 'United States'
    if x[0] == 'F':
        return 'France'
    if x[0].islower():
        return x[0].upper() + x[1:]
    else:
        return x


def cat_parts_time(x):
    if x < 6:
        return 'ночь'
    if x < 10:
        return 'утро'
    if x < 17:
        return 'день'
    if x < 22:
        return 'вечер'
    else:
        return 'ночь'


df['region'] = df['region'].apply(cat_reg)
df['channel'] = df['channel'].apply(lambda x:
                                    'контекстная реклама' if x == 'контексная реклама' else x)
df['device'] = df['device'].apply(lambda x: 'Android' if x == 'android' else x)
df['promo_code'] = df['promo_code'].apply(lambda x: 0 if np.isnan(x) or x == 0 else 1)


#добавление столбцов
df['total_amount'] = df.apply(lambda x:
                              x['revenue'] * 1.1 if x['promo_code'] == 1 else x['revenue'], axis=1)
df['payer'] = df['revenue'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['parts_time'] = df['hour_of_day'].apply(cat_parts_time)
# разделили фрейм на два: с заказом и без.
df_order = df.dropna(subset=['order_dt'])
df_not_order = df[df['order_dt'].isnull()].dropna(axis=1)
#выбросы
med = df_order['revenue'].median()
Q1 = np.percentile(df_order['revenue'], 25, method='midpoint')
Q3 = np.percentile(df_order['revenue'], 75, method='midpoint')
IQR = Q3 - Q1
upper = Q3 + 3 * IQR
lower = Q1 - 3 * IQR
df_order.loc[(df_order['revenue'] > upper) | (df['revenue'] < lower), 'revenue'] = med


sales_region = df.groupby(['region'], as_index=False)['revenue'].sum()
sales_channel = df.groupby(['channel'], as_index=False)['revenue'].sum()
sales_device = df.groupby(['device'], as_index=False)['revenue'].sum()
sales_payment = df.groupby(['payment_type'], as_index=False)['revenue'].count()
payer_region = df.groupby(['region', 'payer'], as_index=False).size()

f, ax = plt.subplots(3, 3)
ax[0, 0].pie(sales_region['revenue'], labels=sales_region['region'], autopct='%1.1f%%')
ax[0, 0].grid()
ax[0, 0].set_title('Доля продаж по регионам')
ax[0, 1].grid()
ax[0, 1].pie(sales_channel['revenue'], labels=sales_channel['channel'], autopct='%1.1f%%')
ax[0, 1].set_title('Доля продаж по источникам')
ax[0, 2].grid()
ax[0, 2].pie(sales_device['revenue'], labels=sales_device['device'], autopct='%1.1f%%')
ax[0, 2].set_title('Доля продаж по устройствам')
ax[1, 0].grid()
ax[1, 0].pie(sales_payment['revenue'], labels=sales_payment['payment_type'], autopct='%1.1f%%')
ax[1, 0].set_title('Количество покупок по типу оплаты')
plt.show()



