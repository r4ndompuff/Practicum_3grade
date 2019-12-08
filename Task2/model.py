import numpy as np
import pandas as pd
import statsmodels.api as sm
import xlrd as xl
import matplotlib.pyplot as plt
import statistics as st
from math import factorial
from statsmodels.tsa.seasonal import seasonal_decompose as sdecomp
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from statsmodels.compat.python import iteritems
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from pandas import datetime, DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings

def permutation(m, n):
    return factorial(n) / (factorial(n - m) * factorial(m))

def diff_operator(set, k):
    size = len(set)
    sum = set[0] * (-1)**(size-1)
    for i in range(1, size):
        minus_counter = (-1)**(size-i-1)
        sum = sum + permutation(i,size-1)*minus_counter*set[i]
    return sum

def AIC_finder(test, predict, k, p, counter, q):
    if (counter > 0):
        c = 0
    else:
        c = 1
    n = p + q + c + 1
    predict = np.array(predict)
    ressid = test - predict.ravel()
    sse = sum(ressid**2)
    AIC = n*np.log(sse/n)+2*k
    return AIC

def arima_optimizer_AIC(training, testing, p, max_k, q):
    print(training)
    print()
    print(testing)
    min_AIC_lib = min_our_AIC = 99999
    min_p_lib = min_p_our = 0
    min_q_lib = min_q_our = 0
    for i in range(p+1):
        for j in range(q+1):
            print("ARIMA(%d,%d,%d)" % (i, max_k, j))
            AIC_lib, our_AIC, error, r2_score, test, predict = arima_learn_predict(training, testing, i, max_k, j)
            if (AIC_lib < min_AIC_lib):
                min_AIC_lib = AIC_lib
                min_test_lib = test
                min_predict_lib = predict
                min_p_lib = i
                min_q_lib = j
                r2_score_lib = r2_score
                error_lib = error
            if (our_AIC < min_our_AIC):
                min_our_AIC = our_AIC
                min_test_our = test
                min_predict_our = predict
                min_p_our = i
                min_q_our = j
                r2_score_our = r2_score
                error_our = error
    print()
    print("Best ARIMA by loglikehood AIC = ARIMA(%d,%d,%d)" % (min_p_lib, max_k, min_q_lib))
    print("Minimal loglikehood AIC = ", min_AIC_lib)
    print('Test MSE: %.3f' % error_lib)
    print("r2 score: ", r2_score_lib)
    print()
    print("Best ARIMA by SSE AIC = ARIMA(%d,%d,%d)" % (min_p_our, max_k, min_q_our))
    print("Minimal SSE AIC = ", min_our_AIC)
    print('Test MSE: %.3f' % error_our)
    print("r2 score: ", r2_score_our)
    plt.plot(test)
    plt.plot(predict, color='red')
    plt.show()

def arima_learn_predict(training, testing, p, max_k, q):
    warnings.simplefilter('ignore')
    size = int(len(training.values))
    train, test = training.values, testing.values
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(p,max_k,q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    print("r2 score: ", r2_score(test, predictions))
    regr = OLS(test, predictions).fit()
    our_aic = AIC_finder(test, predictions,len(test), p, max_k, q)
    print("OLS AIC: ", regr.aic)
    print("Our AIC: ", our_aic)
    return regr.aic, our_aic, error, r2_score(test, predictions), test, predictions

def integral_definer(df):
    values = df
    oper_values = np.array([0]).astype(float)
    counter = 0
    flag = 0
    max_k = 0
    for k in range(1,len(values)):
        if flag:
                continue;
        for i in range(1, len(values)+1):
            if ((i-k-1) >= 0):
                oper_values = np.append(oper_values, 0)
                oper_values[i-1] = diff_operator(values[i-k-1:i], k)
        oper_values_cutted = oper_values[k:len(oper_values)+(1-k)*(len(values)-k) + counter]
        counter = counter - (k-1)
        print(k, " test: ")
        if df_test(oper_values_cutted):
            flag = 1
            max_k = k
            ans = oper_values_cutted;
    return max_k, ans

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

def get_lag(mod, endog, exog, startlag, maxlag, method, modargs = ()):
    results = {} #dict
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        #mod_instance = mod(endog, exog[:, :lag], *modargs) #в нашем случае это класс OLS (первый аргумент функции)
        #results[lag] = mod_instance.fit()
        results[lag] = mod(endog, exog[:, :lag], *modargs).fit()
    if method == "aic":
        icbest, bestlag = min((v.aic, k) for k, v in iteritems(results)) #перебор по значениям из results
    return icbest, bestlag

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

def df_test(df):
    df_vect = df
    df_size = len(df_vect)
    autolag = 'AIC'
    maxlag = None
    regression = 'c'
    #regressions = {None: 'nc', 0: 'c', 1: 'ct', 2: 'ctt'}

    trend_size = len(regression) #размер тренда

    maxlag = int(np.ceil(12. * np.power(df_size / 100., 1/2))) #Максимальное запаздывание, вычисляется как ТВГ соотвествующего выражения
    maxlag = min(df_size // 2 - trend_size, maxlag) #очевидная строчка
    if maxlag < 0:
        raise ValueError('Dataset is too short')

    df_diff = np.diff(df_vect) #массив с первыми разностями: элем_i = a[i+1] - a[i]
    df_diff_all = sm.tsa.lagmat(df_diff[:, None], maxlag, trim='both', original='in') #массив с лагами, где maxlag - число "сдвигов" вниз
    df_size = df_diff_all.shape[0] #количество столбцов в массиве лагов

    df_diff_all[:, 0] = df_vect[-df_size - 1:-1]  #заменяем первый столбец df_diff_all на df_vect
    df_diff_short = df_diff[-df_size:] #оставляем последние df_size элементов

    fullRHS = df_diff_all
    startlag = fullRHS.shape[1] - df_diff_all.shape[1] + 1 #начальный лаг
    icbest, bestlag = get_lag(sm.OLS, df_diff_short, fullRHS, startlag, maxlag, autolag)

    bestlag -= startlag  #оптимальное значение лага

    df_diff_all = sm.tsa.lagmat(df_diff[:, None], bestlag, trim='both', original='in') #массив с лагами, но уже при оптимальном значении лага
    df_size = df_diff_all.shape[0]
    df_diff_all[:, 0] = df_vect[-df_size - 1:-1]  #заменяем первый столбец df_diff_all на df_vect
    df_diff_short = df_diff[-df_size:]
    usedlag = bestlag

    resols = sm.OLS(df_diff_short, sm.tsa.add_trend(df_diff_all[:, :usedlag + 1], regression)).fit() #аппроксимация ряда методом наименьших квадратов
    adfstat = resols.tvalues[0] #получение необходимой статистики

    pvalue = mackinnonp(adfstat, regression = regression, N = 1)
    critvalues = mackinnoncrit(N = 1, regression = regression, nobs = df_size)
    #critvalues = {"1%" : critvalues[0], "5%" : critvalues[1], "10%" : critvalues[2]}

    #return adfstat, pvalue, usedlag, df_size, critvalues, icbest
    if adfstat < critvalues[1]:
        print("Time series is stationary with crit value ", adfstat)
        return True
    else:
        print("Time series is not stationary with crit value ", adfstat)
        return False

# MAIN
# страница 54 и далее (отмена, не читайте эту парашу)

training = pd.read_excel('training.xlsx')
#print(training.columns) #названия столбов

#training['Average'] = avg_data(training) #добавляем новый столбец в наш dataframe
#training['Noise'] = white_noise(training) #добавляем новый столбец в наш dataframe

#stacked = plt.gca() #2 plots 1 figure

#training.plot(kind='line',x='Date',y='Value',ax=stacked)
#training.plot(kind='line',x='Date',y='Average',color='green',ax=stacked)
#training.plot(kind='line',x='Date',y='Noise',color='purple',ax=stacked)
#plt.show()

print("Our test:")
#print(df_test(training))
df_test(training['Value'])
#print(st.mean(training['Noise'].to_numpy()))
print("Library test:")
print(sm.tsa.adfuller(training['Value'])) #проверяем рабочесть нашего теста Дики-Фуллера на библиотечном

print()

training.reset_index(inplace=True)
training['Date'] = pd.to_datetime(training['Date'])
training_s = training.set_index('Date')


training_decomposed = sdecomp(training_s, model = 'additive')
training_decomposed.plot()
plt.show()

# Поиск порядка интегрируемости
values =  training['Value'].to_numpy()
max_k, training_max_k = integral_definer(values)
print("Порядок интегрируемости: ", max_k)

# Ввод необходимый для работы ARIMA
training = pd.read_excel('training.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True)
testing = pd.read_excel('testing.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True)

# Тут мы определяем параметры ARMA модели (p,q)
# Для нашей модели надо проверить p = 0,1,2 и q = 0,1,2,3,4
plt.figure()
plt.subplot(211)
plot_acf(training_max_k, ax=plt.gca())
plt.subplot(212)
plot_pacf(training_max_k, ax=plt.gca())
plt.show()

# Модель обучается и предсказывает
arima_optimizer_AIC(training, testing, 1, 1, 4)

#training_matrix = training.to_numpy()
#print(training_matrix[0,0])
#print(training_matrix[0,1])
#rows, columns = training_matrix.shape
#print(rows)
#print(columns)
#excpected_value = np.zeros(rows, dtype='float') #вектор математических ожиданий элементов временного ряда
#variation = np.zeros(rows, dtype='float') ##вектор дисперсий элементов временного ряда
