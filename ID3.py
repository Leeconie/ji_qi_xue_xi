import pandas as pd
import numpy as np

def id3(X, y, depth=0, max_depth=None):
    if max_depth is not None and depth >= max_depth:
        return np.bincount(y).argmax()
    
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    
    if len(X.columns) == 0:
        return np.bincount(y).argmax()
    
    best_feature, best_threshold = None, None
    best_gain = -1
    
    for feature in X.columns:
        thresholds = np.unique(X[feature])
        for threshold in thresholds:
            gain = information_gain(X, y, feature, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    if best_feature is None:
        return np.bincount(y).argmax()
    
    tree = {best_feature: {}}
    
    for value in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == value].drop(columns=[best_feature])
        sub_y = y[X[best_feature] == value]
        subtree = id3(sub_X, sub_y, depth + 1, max_depth)
        tree[best_feature][value] = subtree
    
    return tree

def information_gain(X, y, feature, value):
    parent_entropy = entropy(y)
    true_y = y[X[feature] == value]
    false_y = y[X[feature] != value]
    
    if len(true_y) == 0 or len(false_y) == 0:
        return 0
    
    true_probability = len(true_y) / len(y)
    false_probability = len(false_y) / len(y)
    
    true_entropy = entropy(true_y) * true_probability
    false_entropy = entropy(false_y) * false_probability
    
    child_entropy = true_entropy + false_entropy
    gain = parent_entropy - child_entropy
    return gain

def entropy(y):
    probabilities = np.bincount(y) / len(y)
    probabilities = probabilities[probabilities != 0]  # 避免除以零
    return -np.sum(probabilities * np.log2(probabilities))

# 示例数据
X = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
y = pd.Series([1, 2, 3])

tree = id3(X, y, max_depth=1)
print(tree)