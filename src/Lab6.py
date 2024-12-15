import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
from tabulate import tabulate

from LogisticRegression import LogisticRegression, evaluate_hyperparameters

data = pd.read_csv("diabetes.csv")
data.fillna(data.mean(), inplace=True)

numerical_cols = data.columns[:-1]
means = data[numerical_cols].mean()
stds = data[numerical_cols].std()

data[numerical_cols] = (data[numerical_cols] - means) / stds

stats = data.describe()
num_stats = len(stats.index)
fig, axes = plt.subplots(num_stats//2, 2, figsize=(90, 50))

for i, stat_name in enumerate(stats.index):
    sns.barplot(x=stats.columns, y=stats.loc[stat_name], ax=axes[i//2,i%2])
    axes[i//2,i%2].set_title(f'{stat_name.capitalize()} по каждому числовому столбцу')
    axes[i//2,i%2].set_ylabel(stat_name.capitalize())
plt.subplots_adjust(hspace=1.5, wspace=1.5)
plt.show()

print(tabulate(stats))
y = data['Outcome'].values
X = data[numerical_cols]
X = pd.DataFrame(X, columns=X.columns).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

results = evaluate_hyperparameters(X_train, y_train, X_test, y_test)
print(tabulate(results, headers=["Learning Rate", "Iterations", "Method", "Accuracy", "Precision",
                      "Recall:", "F1-Score:"]))
def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
    return ret
def p_auc(x, y):
    x = column_or_1d(x)
    y = column_or_1d(y)
    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
    area = direction * trapezoid(y, x)
    if isinstance(area, np.memmap):
        area = area.dtype.type(area)
    return area

def _binary_clf_curve(y_true, y_score):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    y_true = y_true == 1
    print(y_true,y_score)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    print(desc_score_indices,232)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    print(y_true,y_score)
    print(np.diff(y_score))
    distinct_value_indices = np.where(np.diff(y_score))[0]
    print(distinct_value_indices)
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    print(threshold_idxs)
    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    print(tps,fps)
    return fps, tps
def roc_curve1(y_true, y_score):
    fps, tps= _binary_clf_curve(y_true, y_score)
    if len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return fpr, tpr
def precision_recall_curve2(y_true, y_score=None):
    fps, tps = _binary_clf_curve(y_true, y_score)
    ps = tps + fps
    print(ps)
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))
    print(precision)
    recall = tps / tps[-1]
    print(recall)
    sl = slice(None, None, -1)
    print(precision[sl])
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0))
