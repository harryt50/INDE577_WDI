import pytest
import sys
import os

# Debugging: Print sys.path to check if the ml_package folder is in it
print("Python's sys.path:")
for p in sys.path:
    print(p)

# Add the path to your ml_package directory so that we can import from it
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml_package'))
sys.path.append(package_path)

# Debugging: Print sys.path again to ensure the path is correctly added
print(f"Added path: {package_path}")
print("Python's sys.path after modification:")
for p in sys.path:
    print(p)

# Now try importing
try:
    from Income_Group_Project_package.ml_package.clkr_models import (
        CustomLogisticRegression,
        CustomKNNClassifier,
        CustomRandomForestClassifier
    )
    print("Imports succeeded!")
except ModuleNotFoundError as e:
    print(f"Error: {e}")

# Create dummy classification data for testing
@pytest.fixture
def sample_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    return X, y

def test_logistic_regression_predicts(sample_data):
    X, y = sample_data
    model = CustomLogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

def test_knn_classifier_predicts(sample_data):
    X, y = sample_data
    model = CustomKNNClassifier()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

def test_random_forest_predicts(sample_data):
    X, y = sample_data
    model = CustomRandomForestClassifier()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
