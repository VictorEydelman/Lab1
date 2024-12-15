import numpy as np
from sklearn.tree import DecisionTreeClassifier

from TreeNode import TreeNode, Group
import math


class DecisionTree:
    def __init__(self, max_depth=None):
        self.tree = None

    def fit(self, X, y,y_param):
        self.tree = self._grow_tree(X, y,y_param)

    def _grow_tree(self, X, y,y_param):
        gain_ratio_max = 0
        column_max=""
        group_max=[]
        if X.columns.size>1:
            for column in X.columns:
                count = X.shape[0]
                info = self.info(y)
                infox = 0
                X_y = X.copy()
                X_y[y_param] = y
                grouped = X_y.groupby(column)
                grouped_dfs = [group.reset_index(drop=True) for name, group in grouped]
                split=0
                for t in grouped_dfs:
                    count_group = t.shape[0]
                    y_df = t[[y_param]].reset_index(drop=True)
                    infox += (count_group/count*self.info(y_df))
                    split -= (count_group/count*math.log2(count_group/count))
                if split!=0:
                    gain_ratio=(info-infox)/split
                    if(gain_ratio > gain_ratio_max):
                        gain_ratio_max = gain_ratio
                        column_max = column
                        group_max = grouped_dfs
            group_res = []
            for t in group_max:
                y_df = t[[y_param]].reset_index(drop=True)
                n = t[column_max].values[0]
                t.drop(columns=[y_param, column_max], inplace=True)
                if len(y_df.value_counts()) == 1:
                    group_res.append(Group(name=n, status=y_df.iloc[0][y_param]))
                else:
                    status=0
                    c=0
                    mo=0
                    for counts in y_df.value_counts():
                        if counts>c:
                            c=counts
                            status=y_df.values[mo]
                        mo+=1
                    group_res.append(Group(name=n, value=self._grow_tree(t, y_df,y_param), status=status[0]))
        else:
            X_y = X.copy()
            X_y[y_param] = y
            group_max = [X_y]
            column_max=X_y.columns[0]
            group_res = []
            for t in group_max:
                y_df = t[[y_param]].reset_index(drop=True)
                n = t[column_max].values[0]
                t.drop(columns=[y_param, column_max], inplace=True)
                group_res.append(Group(name=n, status=y_df.iloc[0][y_param]))
        return TreeNode(group=group_res,column=column_max)

    def info(self, y):
        info = 0
        count = y.shape[0]
        for counts in y.value_counts():
            info -= (counts / count * math.log2(counts / count))
        return info

    def predict(self, X):
        return np.array([self._predict_sample(num, self.tree,X) for num in range(X.shape[0])])

    def _predict_sample(self, num,node,X):
        for group in node.group:
            if X.iloc[num][node.column]==group.name:
                if group.value is None:
                    return group.status
                else:
                    return self._predict_sample(num,group.value,X)
        min=1000000000000
        next=""
        t=False
        number=0
        nu=0
        f=True
        for i in node.group:
            if i is not None:
                if abs(X.iloc[num][node.column]-i.name)<min:
                    if i.value is None:
                        f=True
                    else:
                        min=abs(X.iloc[num][node.column]-i.name)
                        next=i
                        nu=number
                number+=1
                if i.value is not None:
                    t=True
        if t and not f:
            return self._predict_sample(num, next, X)
        else:
            return node.group[nu].status
    def print_tree(self,node, depth=0):
        print(" "*depth,"column:",node.column)
        for i in node.group:
            print(" "*depth,"status",i.status)
            print(" "*depth,"name",i.name)
            if i.value is not None:
                self.print_tree(i.value,depth+1)
    def predict_probe(self, X):
        return np.array([self._predict_sample_probe(num, self.tree,X) for num in range(X.shape[0])])

    def _predict_sample_probe(self, num,node,X):
        for group in node.group:
            if X.iloc[num][node.column]==group.name:
                if group.value is None:
                    return group.status
                else:
                    return self._predict_sample_probe(num,group.value,X)
        min=1000000000000
        next=""
        t=False
        number=0
        nu=0
        f=True
        nul,ed=0,0
        for i in node.group:
            if i is not None:
                if i.status==0:
                    nul+=1
                else:
                    ed+=1
                if abs(X.iloc[num][node.column]-i.name)<min:
                    if i.value is None:
                        f=True
                    else:
                        min=abs(X.iloc[num][node.column]-i.name)
                        next=i
                        nu=number
                number+=1
                if i.value is not None:
                    t=True
        if t and not f:
            return self._predict_sample_probe(num, next, X)
        else:
            if node.group[nu].status==0:
                return nul/(len(node.group))
            else:
                return ed/(len(node.group))



"""
    def _predict_sample(self,):
        if node.value is not None:
            return node.value
        if sample[node.feature] < node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
    def _grow_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth > self.max_depth):
            return TreeNode(value=np.unique(y)[0])
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return TreeNode(value=np.random.choice(np.unique(y)))
        left = X[:, best_feature] < best_threshold
        right = X[:, best_feature] > best_threshold
        left_node = self._grow_tree(X[left], y[left],depth+1)
        right_node = self._grow_tree(X[right], y[right],depth+1)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)
    def _best_split(self, X, y):
        best_feature, best_threshold, best_gini = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:

                gini = self._gini_index(y[left], y[right])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    def _gini_index(self,left_y,right_y):
        total_samples = len(left_y)+len(right_y)
        if total_samples == 0:
            return 0

        p_left = len(left_y)/total_samples
        p_right = len(right_y)/total_samples

        gini_left = 1 - sum((np.sum(left_y==c)/len(left_y))**2 for c in np.unique(left_y))
        gini_right = 1 - sum((np.sum(right_y==c)/len(right_y))**2 for c in np.unique(right_y))
        return p_left*gini_left+p_right*gini_right
"""
