# Machine Learning Package for Income Group Prediction

##  Model Package Structure Purpose
This package provides custom implementations of machine learning models to predict income groups using data from the World Bank API (WBGAPI). The models include:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**

These models are used for training on income-related data and evaluating predictions. The package includes functionality for training models, making predictions, and evaluating performance via classification reports and confusion matrix visualizations.

### Model Package Structure
- **ml_models.py**: Contains the machine learning models (`CustomLogisticRegression`, `CustomKNNClassifier`, `CustomRandomForestClassifier`) with methods to train, predict, and evaluate using confusion matrix and classification reports.
- **__init__.py**: This is an empty file, however it is important because this file tells Python to treat the folder as a package.

## Functionality Breakdown

### `CustomLogisticRegression`
- **Purpose**: Custom wrapper around the `LogisticRegression` model from scikit-learn.
- **Methods**:
  - `fit(X, y)`: Trains the logistic regression model on data `X` and target `y`.
  - `predict(X)`: Makes predictions on new data `X`.
  - `evaluate(y_true, y_pred, class_names=None)`: Prints the classification report and confusion matrix for evaluation.

### `CustomKNNClassifier`
- **Purpose**: Custom wrapper around the `KNeighborsClassifier` model from scikit-learn.
- **Methods**:
  - `fit(X, y)`: Trains the KNN model on data `X` and target `y`.
  - `predict(X)`: Makes predictions on new data `X`.
  - `evaluate(y_true, y_pred, class_names=None)`: Prints the classification report and confusion matrix for evaluation.

### `CustomRandomForestClassifier`
- **Purpose**: Custom wrapper around the `RandomForestClassifier` model from scikit-learn.
- **Methods**:
  - `fit(X, y)`: Trains the Random Forest model on data `X` and target `y`.
  - `predict(X)`: Makes predictions on new data `X`.
  - `evaluate(y_true, y_pred, class_names=None)`: Prints the classification report and confusion matrix for evaluation.


## Installation
To install the necessary dependencies, create a virtual environment and install the required libraries:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include dependencies like `scikit-learn`, `numpy`, `seaborn`, `matplotlib`, and `pytest` etc.

## Running Test

To test the functionality of the package, use `pytest` to ensure that the models work correctly.

1. Make sure your test file is named `test_ml_models.py` and is located in the same directory as `ml_models.py`.
2. Run the following command in your terminal:

```bash
pytest
```

This will run all tests, checking if the models' predictions are correct and that the code behaves as expected.

---


## Predictive Modeling on Country-Level Indicators Using Machine Learning Jupyter Notebook

The notebook uses World Bank data to classify countries based on infant mortality rates.
Socioeconomic indicators like primary school enrollment, health expenditure, and GDP per capita are considered in the classification process.
The aim is to uncover patterns in these indicators to gain insights into country-level development and well-being.

### Machine Learning Models
The notebook demonstrates the use of a custom machine learning package that includes:

- Logistic Regression

- K-Nearest Neighbors (KNN)

- Random Forest

Real-world data from the World Bank is used to perform the classification of countries by infant mortality rates.

### Feature Correlation
A heatmap illustrates the relationships between the features in the dataset.

**Key correlations:**

- 1.00 indicates perfect self-correlation (always true).

- 0.00 indicates no correlation between features.

- 0.56 indicates a moderate correlation (as one increases, so does the other).

### Confusion Matrix
The confusion matrix is used to evaluate the performance of the machine learning models.

It highlights how well each model identifies different income classes (high vs low income) in relation to infant mortality.

### Model Accuracies
- Random Forest: Highest accuracy at 85%.
Best model, with a strong balance in predicting both high and low-income countries.

- Logistic Regression: Achieved 73% accuracy.
Struggled with low-income cases, resulting in lower recall for this group.

- K-Nearest Neighbors (KNN): Lowest accuracy at 66%.
Missed many low-income cases, leading to its poor performance.


### Conclusion of The nNotebook
Random Forest is identified as the most reliable model for classifying countries based on infant mortality, given its high accuracy and balanced performance across income classes.


## Troubleshooting

- If `pytest` doesn't detect the tests, make sure:
  - Your test file is named `test_ml_models.py`.
  - Test functions are prefixed with `test_` (e.g., `test_logistic_regression_predicts`).
  - You are running `pytest` in the correct directory (same level as `ml_models.py` and `test_ml_models.py`).
  