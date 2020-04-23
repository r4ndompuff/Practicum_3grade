import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class OneStepLinReg():
    def __init__(self):
        self.beta = np.zeros(2)

    def fit(self, vect):
        self.y = vect
        x = np.array(range(len(vect)))
        y_mean, x_mean = np.mean(vect), np.mean(x)
        SS_xy = np.sum(vect * x) - len(vect) * y_mean * x_mean
        SS_xx = np.sum(x * x) - len(vect) * x_mean * x_mean
        self.beta[1] = SS_xy / SS_xx
        self.beta[0] = y_mean - self.beta[1] * x_mean

    def predict(self, coord):
        return self.beta[0] + coord * self.beta[1]

    def coef(self):
        return self.beta

    def display(self):
        fig = plt.figure()
        l = plt.plot([0, len(self.y) - 1], [self.beta[0], self.beta[0] + self.beta[1] * (len(self.y) - 1)], color = "purple", label = "predicted")
        d = plt.scatter(np.array(range(len(self.y))), self.y, color = "orange", label = "real")
        plt.legend()
        plt.grid(linewidth = 1)
        plt.title('Our own regressor', fontsize = 'xx-large')
        plt.show()

def monthly_produced(df):
    for i in range(len(df["UnsId"])):
        if df['Supplier'][i] == sup1:
            H[int(df['ProductDate'][i]) - 1] += int(float(df['Produced'][i]))
        if df['Supplier'][i] == sup2:
            W[int(df['ProductDate'][i]) - 1] += int(float(df['Produced'][i]))
    print("H monthly production:", H)
    print("W monthly production:", W)
    fir = plt.figure()
    colors = ['red','blue']
    p1 = plt.bar(months, H, width=0.2, align='edge', color='orange')
    p2 = plt.bar(months, W, width=-0.2, align='edge', color='purple')
    plt.legend((p1, p2), ('harpy.co', 'westeros.inc'), loc=4, frameon = None )
    plt.grid(linewidth = 1)
    plt.title('Produced in month', fontsize = 'xx-large')
    plt.show()

def monthly_broken_metrics(df):
    global H_metrics
    global W_metrics
    for i in range(len(df["UnsId"])):
        if df['Supplier'][i] == sup1:
            H_broken[int(df['ProductDate'][i]) - 1] += int(float(df['Defects'][i]))
            H_metrics = H_metrics + int(float(df['Produced'][i])) / (int(df['ReportDate'][i]) - int(df['ProductDate'][i]) + 1)
        if df['Supplier'][i] == sup2:
            W_broken[int(df['ProductDate'][i]) - 1] += int(float(df['Defects'][i]))
            W_metrics += int(float(df['Produced'][i])) / (int(df['ReportDate'][i]) - int(df['ProductDate'][i]) + 1)
    print("H monthly broken:", H_broken)
    print("W monthly broken:", W_broken)
    H_metrics = 100000 / H_metrics
    W_metrics = 100000 / W_metrics
    fir = plt.figure()
    colors = ['red','blue']
    p1 = plt.bar(months, H_broken, width=0.2, align='edge', color='orange')
    p2 = plt.bar(months, W_broken, width=-0.2, align='edge', color='purple')
    plt.legend((p1, p2), ('harpy.co', 'westeros.inc'), loc=4, frameon = None )
    plt.grid(linewidth = 1)
    plt.title('Broken in month', fontsize = 'xx-large')
    plt.show()
    print("H metrics:", H_metrics)
    print("W metrics:", W_metrics)
    return 0

def monthly_stats():
    H_s = H_broken / H
    W_s = W_broken / W
    fir = plt.figure()
    colors = ['red','blue']
    p1 = plt.bar(months, H_s, width=0.2, align='edge', color='orange')
    p2 = plt.bar(months, W_s, width=-0.2, align='edge', color='purple')
    plt.legend((p1, p2), ('harpy.co', 'westeros.inc'), loc=4, frameon = None )
    plt.grid(linewidth = 1)
    plt.title('% of broken in month', fontsize = 'xx-large')
    plt.show()
    print("Probabilities H breaks:", H_s)
    print("Probabilities W breaks:", W_s)
    return H_s, W_s

def make_forecast(training_set):
    X = []
    y = []
    for i in range(len(training_set) - 2):
        monthly = training_set[i : i + 2]
        X.append(list(monthly))
        y.append(training_set[i + 2])
    LinReg = LinearRegression()
    LinReg.fit(X, y)
    prediction = LinReg.predict(training_set[-2 : ].reshape(1, 2))
    return prediction

# M A I N
product = pd.read_csv('Book1.csv', sep = ';', names = ['UnsId','ProductDate','ReportDate','Produced','Defects','Supplier'])

sup1 = 'harpy.co'
sup2 = 'westeros.inc'

H = np.zeros(6, dtype="int")
W = np.zeros(6, dtype="int")
H_broken = np.zeros(6, dtype="int")
W_broken = np.zeros(6, dtype="int")
H_metrics = W_metrics = 0
months = np.array(range(1, 7))

monthly_produced(product)
monthly_broken_metrics(product)
H_stats, W_stats = monthly_stats()

H_next = make_forecast(H)[0]
W_next = make_forecast(W)[0]
H_next_broken = make_forecast(H_broken)[0]
W_next_broken = make_forecast(W_broken)[0]
print("Predicted to produce H next month:", H_next)
print("Predicted to produce W next month:", W_next)
print("Predicted to break H next month:", H_next_broken)
print("Predicted to break W next month:", W_next_broken)
print("Probability H breaks next month (sklearn):", H_next_broken / H_next)
print("Probability W breaks next month (sklearn):", W_next_broken / W_next)

#наш собственный регрессор, для H
oslr = OneStepLinReg()
oslr.fit(H_stats)
print("Probability H breaks next month (OUR REGRESSOR):", oslr.predict(len(H_stats)))
oslr.display()

#наш собственный регрессор, для W
oslr = OneStepLinReg()
oslr.fit(W_stats)
print("Probability W breaks next month (OUR REGRESSOR):", oslr.predict(len(W_stats)))
oslr.display()
