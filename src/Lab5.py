import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, roc_curve, \
    precision_recall_curve, auc
from sklearn.metrics._ranking import _binary_clf_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
from tabulate import tabulate

from DecisionTree import DecisionTree

data = pd.read_csv("DATA.csv")
threshold = 2

def ID_to_int(student_id):
    students = []
    for i in student_id:
        students.append(int(i.replace("STUDENT", "")))
    return students


def determine_success(grades):
    average_grade = sum(grades) / len(grades)
    status = []
    for grade in grades:
        if grade >= average_grade:
            status.append(1)
        else:
            status.append(0)
    return status


def evaluate(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_pred)):
        TP += (y_true[i] == 1) == True & (y_pred[i] == 1) == True
        TN += ((y_true[i] == 0) == True & (y_pred[i] == 0) == True)
        FP += (y_true[i] == 0) == True & (y_pred[i] == 1) == True
        FN += (y_true[i] == 1) == True & (y_pred[i] == 0) == True
    accuracy = (TP + TN) / (TP + TN + FP + FN)if  (TP + TN + FP + FN) != 0 else 0
    precision = TP / (TP + FP)if TP + FP != 0 else 0
    recall = TP / (TP + FN)if (TP + FN) != 0 else 0
    return accuracy, precision, recall

def compute_roc(y_true, y_scores):
    thresholds = np.linspace(0, 1, num=100)
    tpr = []
    fpr = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        tpr.append(TP / (TP + FN) if (TP + FN) > 0 else 0)  # True Positive Rate
        fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0)  # False Positive Rate
    fpr=np.r_[0,fpr]
    tpr=np.r_[0,tpr]
    return np.array(fpr), np.array(tpr)

def p_auc(x, y):
    return np.abs((np.diff(x) * (y[tuple([slice(1, None)])] + y[tuple( [slice(None, -1)])]) / 2.0).sum(-1))

def compute_pr(y_true, y_scores):
    thresholds = np.linspace(0, 1.1, num=100)
    precision = []
    recall = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        precision.append(TP / (TP + FP) if (TP + FP) > 0 else 1)
        recall.append(TP / (TP + FN) if (TP + FN) > 0 else 0)

    return np.array(precision), np.array(recall)
data["STATUS"] = determine_success(data['GRADE'])
data["STUDENT ID"] = ID_to_int(data['STUDENT ID'])
num_features_to_select = int(np.sqrt(data.shape[1]))
y_param="STATUS"#np.random.choice(data.columns, 1, replace=False).tolist()[0]
df = data.drop(columns=[y_param])
selected_features = np.random.choice(df.columns, num_features_to_select-1, replace=False).tolist()
selected_features.append("GRADE")
#if(y_param ):
 #   for column in data.select_dtypes(include=['object']).columns:
 #       data[column] = data[column].replace({"Yes": 1, "No": 0})
#else:
print("Отобранные признаки:")
print(selected_features)
print(y_param)
X_train, X_test, y_train, y_test = train_test_split(data[selected_features], data[y_param], test_size=0.1, random_state=42)
tree = DecisionTree()
tree.fit(X_train, y_train, y_param)
#tree.print_tree(tree.tree)
predictions=tree.predict_probe(X_test)
pr=tree.predict(X_test)
cm = confusion_matrix(y_test, pr)
disp_full = ConfusionMatrixDisplay(confusion_matrix=cm)
disp_full.plot(cmap=plt.cm.Blues)
plt.title(f'Матрица ошибок')
plt.show()
accuracy, precision, recall = evaluate(y_test.values, predictions)
print(f'Accuracy: {accuracy} \nPrecision: {precision} \nRecall: {recall}')
fpr,tpr=compute_roc(y_test, predictions)
fpr.sort()
tpr.sort()
auc_roc = p_auc(fpr, tpr)
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(auc_roc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower left')

precision_curve, recall_curve = compute_pr(y_test.values, predictions)
pr_auc = p_auc(recall_curve, precision_curve)
plt.subplot(1, 2, 2)
plt.plot(recall_curve, precision_curve,  label='PR curve (area = {:.2f})'.format(pr_auc))
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
print(f"AUC-ROC: {auc_roc:.2f}")
print(f"AUC-PR: {pr_auc:.2f}")
