# Logistic Regression Analysis

## Background
Logistic Regression classifies countries into income levels (high vs. low) using health, GHG emissions per capita, and military expenditure as predictors. This probabilistic classification algorithm helps us understand how various development indicators contribute to a country's economic status, providing interpretable odds ratios that directly measure each factor's impact on the likelihood of being classified as a high-income country.

## Theory
Logistic Regression is a probabilistic supervised learning algorithm that:
1. Models the probability of binary outcomes using the logistic (sigmoid) function
2. Transforms linear combinations of features into probabilities between 0 and 1
3. Uses maximum likelihood estimation to find optimal parameters
4. Provides interpretable coefficients in terms of log-odds ratios

Key concepts:
- Sigmoid function and logistic transformation
- Maximum likelihood estimation
- Log-odds and odds ratios
- Decision boundaries
- Probability calibration
- Regularization (L1/L2)

## Scope
### Primary Objectives
- Classify countries into income categories based on development indicators
- Quantify the impact of each factor on income classification
- Analyze the relationship between environmental impact and economic status
- Provide interpretable insights for policy decisions

### Research Questions
1. What is the relationship between economic development, environmental impact, and military expenditure?
2. How do health outcomes and labor force participation influence income classification?
3. Can we identify key thresholds in development indicators that separate income groups?

## Implementation Details
### Data Preprocessing
1. Feature Selection and Engineering:
   - Target variable: Income Level (binary: high/low)
   - Primary predictors: Life Expectancy, GHG Emissions per Capita
   - Secondary predictors: Labor Force Participation, Military Expenditure, Debt Service
   - Feature scaling: StandardScaler for consistent model performance

2. Data Cleaning:
   - Handling missing values using forward fill and mean imputation
   - Removing outliers using IQR method
   - Log transformation for skewed features (GDP, GHG emissions)
   - Encoding income levels into binary categories

3. Feature Transformation:
   - Polynomial features for capturing non-linear relationships
   - Interaction terms between key variables
   - Feature selection using L1 regularization

### Algorithm Configuration
1. Model Selection:
   - Logistic Regression with L2 regularization
   - Cross-validation for hyperparameter tuning
   - Probability calibration using CalibratedClassifierCV

2. Model Validation:
   - Stratified k-fold cross-validation (k=5)
   - Train-test split (80-20) with stratification
   - ROC curve and AUC analysis

3. Diagnostics:
   - Confusion matrix analysis
   - Classification report
   - Feature importance analysis
   - Probability calibration plots

### Visualization Approach
1. Model Diagnostics:
   - ROC curves
   - Precision-Recall curves
   - Calibration plots
   - Learning curves

2. Feature Analysis:
   - Coefficient magnitude plots
   - Feature importance heatmap
   - Decision boundary visualization
   - Probability distribution plots

3. Results Visualization:
   - Confusion matrix heatmap
   - Classification boundary plots
   - Probability threshold analysis
   - Misclassification analysis

## Technical Details
- Library: scikit-learn
- Algorithm Parameters:
  ```python
  LogisticRegression(
      penalty='l2',
      C=1.0,
      solver='lbfgs',
      max_iter=1000,
      multi_class='ovr',
      class_weight='balanced'
  )
  ```
- Model Pipeline:
  ```python
  Pipeline([
      ('scaler', StandardScaler()),
      ('poly', PolynomialFeatures(degree=2)),
      ('classifier', LogisticRegression())
  ])
  ```
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Average Precision Score

## Dependencies
- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0
- wbdata>=0.3.0
- statsmodels>=0.13.0

## References
1. Hastie, T., et al. (2009). The Elements of Statistical Learning
2. World Bank Income Classifications Documentation
3. Hosmer, D. W., et al. (2013). Applied Logistic Regression
