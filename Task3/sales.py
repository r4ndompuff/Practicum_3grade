import pandas as pd
import datetime as dt
import numpy as np

def daily(store_name):
    goods_recieved = pd.read_csv(source_path + store_name + supply)
    goods_sold = pd.read_csv(source_path + store_name + sell)
    daily_stats = pd.DataFrame({'date' : [dt.date(2006, 1, 1)], 'apple': [0], 'pen': [0]})
    daily_stats.reset_index()
    pens = apples = 0
    index = 0 #столбец date сделать индексом потом
    index_income = 0 #для поставок
    current_date = dt.date(2006, 1, 1)
    #в этом блоке считаем дневные продажи яблок и ручек
    for i in range(len(goods_sold['date']) - 1):
        if (index_income < len(goods_recieved['date'])) and (goods_recieved['date'][index_income] == goods_sold['date'][i]):
            daily_stats.loc[index, 'apple'] += goods_recieved['apple'][index_income]
            daily_stats.loc[index, 'pen'] += goods_recieved['pen'][index_income]
            print("Recieved apples ",goods_recieved['apple'][index_income])
            print("Recieved pens ",goods_recieved['pen'][index_income])
            index_income += 1
        if goods_sold['sku_num'][i].find("-ap-") != -1:
            apples += 1
        elif goods_sold['sku_num'][i].find("-ap-") == -1:
            pens += 1
        if goods_sold['date'][i] != goods_sold['date'][i + 1]:
            daily_stats.loc[index, 'date'] = current_date
            daily_stats.loc[index, 'apple'] -= apples
            daily_stats.loc[index, 'pen'] -= pens
            index += 1
            current_date = current_date + dt.timedelta(days = 1)
            apples = 0
            pens = 0
            daily_stats.loc[index] = [current_date, daily_stats['apple'][index - 1], daily_stats['pen'][index -1]]
        print("Step ", i)
    daily_stats.set_index('date', inplace = True)
    print(daily_stats)
    daily_stats.to_csv(out_path + store_name + "-daily.csv")

def stolen(store_name):
    daily_stats = pd.read_csv(out_path + store_name + "-daily.csv")
    monthly_inventory = pd.read_csv(source_path + store_name + inventory)
    monthly_stolen = pd.DataFrame({'date' : [dt.date(2006, 1, 31)], 'apple': [0], 'pen': [0]})
    index = 0
    for i in range(len(daily_stats['date'])):
        if (index < len(monthly_inventory['date'])) and (daily_stats['date'][i] == monthly_inventory['date'][index]):
            monthly_stolen.loc[index, 'date'] = daily_stats['date'][i]
            monthly_stolen.loc[index, 'apple'] = daily_stats['apple'][i] - monthly_inventory['apple'][index]
            monthly_stolen.loc[index, 'pen'] = daily_stats['pen'][i] - monthly_inventory['pen'][index]
            index += 1
            monthly_stolen.loc[index] = [monthly_inventory['date'][index - 1], 0, 0]
        print("Step ", i)
    monthly_stolen.set_index('date', inplace = True)
    print(monthly_stolen)
    monthly_stolen.to_csv(out_path + store_name + "-steal.csv")

#M A I N
source_path = "ref/out/input/"
out_path = "res/"
names = np.array(["MS-b1", "MS-b2", "MS-m1", "MS-m2", "MS-s1", "MS-s2", "MS-s3", "MS-s4", "MS-s5"])
inventory = "-inventory.csv"
sell = "-sell.csv"
supply = "-supply.csv"

#daily(names[6])
stolen(names[6])
