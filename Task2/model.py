import numpy as np
import pandas as pd
import statsmodels as sm
import xlrd as xl
import matplotlib.pyplot as plt

def avg_data(df):
    averages = []
    for i in df.index:
        elem = 0
        for j in range(i+1):
            elem = elem + df['Value'][j]/(i+1)
        averages.append(elem)
    return averages

# MAIN

#training = xl.open_workbook("training.xlsx")
training = pd.read_excel('training.xlsx')
print(training.columns) #названия столбов
print(training.index) #количество строк (считается с нуля)

training['Average'] = avg_data(training) #добавляем новый столбец в наш dataframe

stacked = plt.gca() #2 plots 1 figure

training.plot(kind='line',x='Date',y='Value',ax=stacked)
training.plot(kind='line',x='Date',y='Average',color='green',ax=stacked)
plt.show()

#print(training['Value'][5])
training_matrix = training.to_numpy()
#print(training_matrix[0,0])
#print(training_matrix[0,1])
rows, columns = training_matrix.shape
#print(rows)
#print(columns)
excpected_value = np.zeros(rows, dtype='float') #вектор математических ожиданий элементов временного ряда
variation = np.zeros(rows, dtype='float') ##вектор дисперсий элементов временного ряда
