import numpy as np
from matplotlib import pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method='gradient_descent'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None

    def log_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = -1/m * np.sum(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-linear_model))

            if self.method == 'gradient_descent':
                dw = (1/m) * np.dot(X.T, (y_pred - y))
                db = (1/m) * np.sum(y_pred - y)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            elif self.method == 'newton':
                gradient = (1/m) * np.dot(X.T, (y_pred - y))
                H = (1/m) * np.dot(X.T, X * (y_pred * (1 - y_pred)).reshape(-1, 1))
                self.weights -= np.linalg.inv(H).dot(gradient)
                self.bias -= np.mean(y_pred - y)

            if i % 100 == 0:
                loss = self.log_loss(y, y_pred)
                print(f'Итерация {i}, Потеря: {loss:.4f}')

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-linear_model))
        return np.where(y_pred >= 0.5, 1, 0)

    def predict_probe(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-linear_model))
        return y_pred
def p_auc(x, y):
    return np.abs((np.diff(x) * (y[tuple([slice(1, None)])] + y[tuple( [slice(None, -1)])]) / 2.0).sum(-1))
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
def evaluate_hyperparameters(X_train, y_train, X_test, y_test):
    learning_rates = [0.001, 0.01, 0.1]
    n_iterations_list = [100, 500, 1000]
    methods = ['gradient_descent', 'newton']

    results = []

    for lr in learning_rates:
        for n_iter in n_iterations_list:
            for method in methods:
                model = LogisticRegression(learning_rate=lr, n_iterations=n_iter, method=method)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy,precision,recall = evaluate(y_test,predictions)
                f1=2*(precision*recall)/(precision+recall)
                results.append((lr, n_iter, method, accuracy, precision, recall, f1))
                print(f'Learning Rate: {lr}, Iterations: {n_iter}, Method: {method}, '
                      f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
                      f'Recall: {recall:.4f}, F1-Score: {f1:.4f}')
                predictions=model.predict_probe(X_test)
                fpr, tpr = compute_roc(y_test, predictions)
                tpr.sort()
                fpr.sort()
                auc_roc = p_auc(fpr, tpr)
                plt.subplot(1, 2, 1)
                plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(auc_roc))
                plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                plt.title('ROC Curve')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc='lower left')

                precision_curve, recall_curve = compute_pr(y_test, predictions)
                pr_auc = p_auc(recall_curve, precision_curve)
                plt.subplot(1, 2, 2)
                plt.plot(recall_curve, precision_curve, label='PR curve (area = {:.2f})'.format(pr_auc))
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

    return results


