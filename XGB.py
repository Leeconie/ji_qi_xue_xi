import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return self.Node(value=np.mean(y))

        best_feature, best_threshold = None, None
        best_mse = float('inf')

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                current_mse = self._calculate_mse(y[left_indices], y[right_indices])
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return self.Node(value=np.mean(y))

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return self.Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def _calculate_mse(self, y_left, y_right):
        mse_left = np.mean((y_left - np.mean(y_left)) ** 2) if len(y_left) > 0 else 0
        mse_right = np.mean((y_right - np.mean(y_right)) ** 2) if len(y_right) > 0 else 0
        return mse_left + mse_right


class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.base_prediction = np.mean(y)
        residuals = y - self.base_prediction

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        predictions = np.full(X.shape[0], self.base_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 生成示例数据
X_train = np.array([[1], [2], [3]])
y_train = np.array([10, 20, 30])
X_test = np.array([[4], [5]])
y_test = np.array([40, 50])

# 创建梯度提升模型
model = GradientBoosting(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = calculate_mse(y_test, y_pred)
print(f"均方误差: {mse:.4f}")

# 打印预测结果
print("预测结果:")
for i in range(len(y_pred)):
    print(f"输入: {X_test[i][0]}, 预测值: {y_pred[i]:.2f}, 真实值: {y_test[i]}")