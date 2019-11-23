import numpy as np
import pandas as pd
import statsmodels as sm
import xlrd as xl

print("the beginning of task2")

#training = xl.open_workbook("training.xlsx")
training = pd.read_excel('training.xlsx')
training_matrix = training.to_numpy()
print(training_matrix[0,0])
print(training_matrix[0,1])
