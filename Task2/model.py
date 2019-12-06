import numpy as np
import pandas as pd
import statsmodels.api as sm
import xlrd as xl
import matplotlib.pyplot as plt
import statistics as st
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit

def avg_data(df): #скользящая средняя
    rows, columns = df.shape
    averages = []
    averages.append(df['Value'][0])
    for i in range(1, rows):
        elem = 0
        for j in range(i):
            elem = elem + df['Value'][j]
        elem = elem/i
        averages.append(elem)
    return averages

def white_noise(df): #первые разности
    rows, columns = df.shape
    noise = []
    noise.append(df['Value'][0])
    #print(df.size)
    for i in range(1, rows):
        noise.append(df['Value'][i]-df['Value'][i-1])
    noise[0] = noise[1]
    return noise

def df_test_old(df): #типа тест Дики-Фуллера, но на самом деле хуйня
    rows, columns = df.shape
    #print(rows)
    #values_avg = np.average(df['Value'].to_numpy())
    #variation = np.var(df['Value'].to_numpy())
    #variation_2 = (rows)/(rows -2)
    #t = values_avg/(((variation**2)/(rows)))**(1/2) Закоменченное - какое-то странное число, нормальная формула ниже
    avg = np.average(df['Value'].to_numpy()) #среднее зачение
    mode = st.mode(df['Value']) #мода (хз, она ли должна быть в формуле)
    sigma = 0 #сигма (в квадрате) из формулы
    for i in range(rows):
        sigma = sigma + (df['Value'][i] - avg)**2
    sigma = sigma/(rows - 1)
    sigma = sigma**(1/2)
    t = (avg - mode)/(sigma*(rows**(1/2))) #сама формула
    return t

def df_test(df, ):
    df_vect = df['Value'].to_numpy() #значения ряда из входного Dataframe
    maxlag = None
    regression = 'c'
    autolag = None
    regresults = False
    regressions = {None: 'nc', 0: 'c', 1: 'ct', 2: 'ctt'}

    df_size = df_vect.shape[0] #размер временного ряда
    ntrend = len(regression) #размер тренда (?)

    maxlag = int(np.ceil(12. * np.power(df_size / 100., 1/2))) #Максимальное запаздывание, вычисляется как ТВГ соотвествующего выражения
    maxlag = min(df_size // 2 - ntrend, maxlag)
    if maxlag < 0:
        raise ValueError('Dataset is too short')

    df_diff = np.diff(df_vect) #массив с первыми разностями: элем_i = a[i+1] - a[i]
    xdall = sm.tsa.lagmat(df_diff[:, None], maxlag, trim='both', original='in') #создает массив с лагами, где maxlag - число "сдвигов" вниз
    df_size = xdall.shape[0] #количество столбцов в массиве лагов

    xdall[:, 0] = df_vect[-df_size - 1:-1]  # replace 0 df_diff with level of df_vect
    xdshort = df_diff[-df_size:]

    usedlag = maxlag
    #icbest = None

    resols = sm.OLS(xdshort, sm.tsa.add_trend(xdall[:, :usedlag + 1], regression)).fit()
    adfstat = resols.tvalues[0]

    pvalue = mackinnonp(adfstat, regression = regression, N = 1)
    critvalues = mackinnoncrit(N = 1, regression = regression, nobs = df_size)
    #critvalues = {"1%" : critvalues[0], "5%" : critvalues[1], "10%" : critvalues[2]}

    #return adfstat, pvalue, usedlag, df_size, critvalues
    if adfstat < critvalues[1]:
        print("Time series is stationary")
    else:
        print("Time series is not stationary")

# MAIN

# страница 54 и далее

training = pd.read_excel('training.xlsx')
#print(training.columns) #названия столбов

training['Average'] = avg_data(training) #добавляем новый столбец в наш dataframe
training['Noise'] = white_noise(training) #добавляем новый столбец в наш dataframe

stacked = plt.gca() #2 plots 1 figure

training.plot(kind='line',x='Date',y='Value',ax=stacked)
training.plot(kind='line',x='Date',y='Average',color='green',ax=stacked)
training.plot(kind='line',x='Date',y='Noise',color='purple',ax=stacked)
plt.show()

print("Our test:")
#print(df_test(training))
df_test(training)
#print(st.mean(training['Noise'].to_numpy()))
print("Library test:")
print(sm.tsa.adfuller(training['Value'])) #проверяем рабочесть нашего теста Дики-Фуллера на библиотечном


#training_matrix = training.to_numpy()
#print(training_matrix[0,0])
#print(training_matrix[0,1])
#rows, columns = training_matrix.shape
#print(rows)
#print(columns)
#excpected_value = np.zeros(rows, dtype='float') #вектор математических ожиданий элементов временного ряда
#variation = np.zeros(rows, dtype='float') ##вектор дисперсий элементов временного ряда
