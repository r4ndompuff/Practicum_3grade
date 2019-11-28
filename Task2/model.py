import numpy as np
import pandas as pd
import statsmodels.api as sm
import xlrd as xl
import matplotlib.pyplot as plt

def avg_data(df):
    averages = []
    averages.append(df['Value'][0])
    for i in range(1, (df.size)//2):
        elem = 0
        for j in range(i):
            #print(j)
            #print(df['Value'][j])
            elem = elem + df['Value'][j]
        elem = elem/i
        averages.append(elem)
    #print(averages[359])
    return averages

def df_test(df):
    values_avg = np.average(df['Value'].to_numpy())
    variation = np.var(df['Value'].to_numpy())
    variation_2 = (df.size//2)/(df.size//2 -2)
    #print(variation)
    t = values_avg/(((variation**2)/(df.size//2)))**(1/2)
    return t

# MAIN

#training = xl.open_workbook("training.xlsx")
training = pd.read_excel('training.xlsx')
#print(training.columns) #названия столбов
#print(training.index) #количество строк (считается с нуля)
#print(training['Value'])
#print(training.size)


training['Average'] = avg_data(training) #добавляем новый столбец в наш dataframe

stacked = plt.gca() #2 plots 1 figure

training.plot(kind='line',x='Date',y='Value',ax=stacked)
training.plot(kind='line',x='Date',y='Average',color='green',ax=stacked)
#plt.show()

print(df_test(training))
print(sm.tsa.adfuller(training['Value']))


#training_matrix = training.to_numpy()
#print(training_matrix[0,0])
#print(training_matrix[0,1])
#rows, columns = training_matrix.shape
#print(rows)
#print(columns)
#excpected_value = np.zeros(rows, dtype='float') #вектор математических ожиданий элементов временного ряда
#variation = np.zeros(rows, dtype='float') ##вектор дисперсий элементов временного ряда
