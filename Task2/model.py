import numpy as np
import pandas as pd
import statsmodels as sm
import xlrd as xl
import matplotlib.pyplot as plt


print("the beginning of task2")

#training = xl.open_workbook("training.xlsx")
training = pd.read_excel('training.xlsx')
print(training.columns)

training.plot(kind='line',x='Date',y='Value')
plt.show()

print(training['Value'][5])
training_matrix = training.to_numpy()
#print(training_matrix[0,0])
#print(training_matrix[0,1])
rows, columns = training_matrix.shape
#print(rows)
#print(columns)
excpected_value = np.zeros(rows, dtype='float') #вектор математических ожиданий элементов временного ряда
variation = np.zeros(rows, dtype='float') ##вектор дисперсий элементов временного ряда
