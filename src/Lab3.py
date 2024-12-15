import itertools
import numpy as np
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("Student_Performance.csv")
test_set_size = int(len(data) * 0.95)
train_data = data[:test_set_size]
data[test_set_size:]=None
for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)
for column in data.select_dtypes(include=['object']).columns:
    data[column] = data[column].replace({"Yes": 1, "No": 0})
stats = data.describe()
for column in data.columns:
    data[column].fillna(data[column].mean(), inplace=True)
print(tabulate(stats, headers=stats.keys(), tablefmt=""))
plt.figure(figsize=(12, 6))
for i, column in enumerate(data.columns):
    plt.subplot(2, (len(data.columns) + 1) // 2, i + 1)
    sns.histplot(data[column], kde=True)
    plt.title(f'Гистограмма {column}')
    plt.xlabel(column)
    plt.ylabel('Частота')
plt.tight_layout()
plt.show()
data = pd.get_dummies(data, drop_first=True)
numerical_cols = data.columns
for column in numerical_cols:
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
for col in data.keys():
    y = data[col].to_numpy()
    fas = np.array(['Hours Studied', 'Previous Scores', 'Extracurricular Activities',
                    'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index'])
    asd = data[fas[fas != col]]
    R_squared_max = 0
    W_max = 0
    for L in range(2,len(asd) + 1):
        for X in itertools.combinations(asd, L):
            X=data[list(X)].to_numpy()
            n, k = X.shape
            X = np.column_stack((X, np.ones(n)))
            W = np.linalg.inv(X.T @ X) @ X.T @ y
            y_pred = X @ W.T
            N = df = n - k
            Var_res = np.sum((y - y_pred) ** 2) / N
            y_mean = np.mean(y)
            Var_data = np.sum((y - y_mean) ** 2) / N
            Var_reg = np.sum((y_pred - y_mean) ** 2) / N
            R_squared = Var_reg / Var_data
            if R_squared>R_squared_max:
                R_squared_max = R_squared
                W_max=W
    print(col)
    print(f'Оценки наименьших квадратов для параметров регрессии:\n{W_max}')
    print(f'Коэффициент детерминации: {R_squared_max}')