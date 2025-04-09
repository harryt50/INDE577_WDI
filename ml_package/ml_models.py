from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

class CustomLogisticRegression:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred, class_names=None):
        print("\nLogistic Regression Classification Report")
        print(classification_report(y_true, y_pred, target_names=class_names))
        self._plot_confusion(y_true, y_pred, class_names)

    def _plot_confusion(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title("Logistic Regression Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


class CustomKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred, class_names=None):
        print("\nKNN Classifier Report")
        print(classification_report(y_true, y_pred, target_names=class_names))
        self._plot_confusion(y_true, y_pred, class_names)

    def _plot_confusion(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title("KNN Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


class CustomRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred, class_names=None):
        print("\nRandom Forest Classifier Report")
        print(classification_report(y_true, y_pred, target_names=class_names))
        self._plot_confusion(y_true, y_pred, class_names)

    def _plot_confusion(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title("Random Forest Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
