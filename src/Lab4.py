import itertools

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")
data.fillna(data.mean(), inplace=True)

numerical_cols = data.columns[:-1]
means = data[numerical_cols].mean()
stds = data[numerical_cols].std()

data[numerical_cols] = (data[numerical_cols] - means) / stds

stats = data.describe()
print(tabulate(stats))

data.hist(bins=10, figsize=(15, 10))
plt.suptitle('Гистограммы признаков')
plt.show()

num_combinations = len(list(itertools.combinations(numerical_cols, 3)))
cols = 5
rows = (num_combinations + cols - 1) // cols

fig = plt.figure(figsize=(25, 5 * rows))

for i, X in enumerate(itertools.combinations(numerical_cols, 3)):
    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    x = data[X[0]]
    y = data[X[1]]
    z = data[X[2]]
    outcome = data['Outcome']
    colors = ['r' if o == 1 else 'b' for o in outcome]
    ax.scatter(x, y, z, c=colors, alpha=0.6)
    ax.set_xlabel(X[0])
    ax.set_ylabel(X[1])
    ax.set_zlabel(X[2])
    ax.set_title(f'3D Визуализация')
plt.subplots_adjust(wspace=2)
plt.tight_layout(pad=3.0)
plt.show()

X = data[numerical_cols]
y = data['Outcome'].values
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [np.bincount([
            self.y_train[i] for i in np.argsort(np.linalg.norm(self.X_train - x, axis=1))[:self.k]]).argmax()
                       for x in X]
        return np.array(predictions)


k_values = [3, 5, 10]
num_k = len(k_values)

fig, axs = plt.subplots(num_k, 2, figsize=(10, 5 * num_k))
accuracy_scores_full = []
accuracy_scores_random = []
for i, k in enumerate(k_values):
    model_full_features = KNN(k)
    model_full_features.fit(X_train, y_train)
    predictions_full = model_full_features.predict(X_test)

    cm_full = confusion_matrix(y_test, predictions_full)

    np.random.seed(42)
    random_features_indices = np.random.choice(X_train.shape[1], size=4, replace=False)
    X_train_random = X_train[:, random_features_indices]
    X_test_random = X_test[:, random_features_indices]

    model_random_features = KNN(k)
    model_random_features.fit(X_train_random, y_train)
    predictions_random = model_random_features.predict(X_test_random)

    cm_random = confusion_matrix(y_test, predictions_random)

    disp_full = ConfusionMatrixDisplay(confusion_matrix=cm_full)
    disp_full.plot(ax=axs[i, 0], cmap=plt.cm.Blues)
    axs[i, 0].set_title(f'Матрица ошибок полной модели k={k},\n {accuracy_score(y_test, predictions_full)}')
    accuracy_scores_full.append(accuracy_score(y_test, predictions_full))
    disp_random = ConfusionMatrixDisplay(confusion_matrix=cm_random)
    disp_random.plot(ax=axs[i, 1], cmap=plt.cm.Blues)
    axs[i, 1].set_title(f'Матрица ошибок с случайными признаками k={k},\n {accuracy_score(y_test, predictions_random)}')
    accuracy_scores_random.append(accuracy_score(y_test, predictions_random))
plt.tight_layout()
plt.show()

print(
    f'Максимальная точность для полной модели при k = {k_values[accuracy_scores_full.index(max(accuracy_scores_full))]}')
print(
    f'Максимальная точность для модели с случайными признаками при k = {k_values[accuracy_scores_random.index(max(accuracy_scores_random))]}')
