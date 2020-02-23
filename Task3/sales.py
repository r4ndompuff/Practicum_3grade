import pandas as pd
import datetime as dt

goods_recieved = pd.read_csv('ref/out/input/MS-s4-supply.csv')
goods_sold = pd.read_csv('ref/out/input/MS-s4-sell.csv')

daily_stats = pd.DataFrame({'date' : [dt.date(2006, 1, 1)], 'apple': [0], 'pen': [0]})
daily_stats.reset_index()

pens = apples = 0
year  = 2006 #юзлесс
month = 1 #юзлесс
day = 1 #юзлесс
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
    else:
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

print(daily_stats)

daily_stats.to_csv("res/MS-s3-daily.csv")
