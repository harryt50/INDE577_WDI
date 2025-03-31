# Gradient Descent Analysis

## Background
Implementation of Gradient Descent for optimizing predictions of life expectancy based on development indicators. This algorithm helps us understand how different socio-economic and environmental factors contribute to health outcomes, with a focus on finding the optimal weights for each factor through iterative optimization.

## Theory
Gradient Descent is a first-order iterative optimization algorithm widely used in machine learning and data science to minimize a cost function, which measures the error between predicted and actual values in a model. The core idea is to iteratively adjust the model’s parameters (e.g., weights in a neural network or coefficients in linear regression) by moving in the direction of the steepest descent, as determined by the negative gradient of the cost function. This gradient, computed using first-order derivatives, indicates the direction of the greatest increase in the cost function, so moving in the opposite direction (negative gradient) reduces the error. For example, in linear regression, the cost function might be the mean squared error, and Gradient Descent updates the slope and intercept to better fit the data points, aiming to find the parameter values that yield the lowest possible error.

The process of Gradient Descent involves updating the parameters iteratively based on the gradient of the cost function with respect to each parameter. At each iteration, the algorithm computes the gradient, which acts as a compass pointing toward the steepest ascent, and then takes a step in the opposite direction to descend the cost landscape. The size of this step is controlled by a hyperparameter called the learning rate, which scales the gradient update. Mathematically, for a parameter \( \theta \), the update rule is \( \theta = \theta - \eta \cdot \frac{\partial J}{\partial \theta} \), where \( \eta \) is the learning rate and \( \frac{\partial J}{\partial \theta} \) is the partial derivative of the cost function \( J \) with respect to \( \theta \). If the learning rate is too large, the algorithm might overshoot the minimum, leading to divergence; if too small, convergence becomes painfully slow. Variants like Stochastic Gradient Descent (SGD) and Mini-Batch Gradient Descent introduce randomness by updating parameters using subsets of the data, improving efficiency for large datasets while still following the same descent principle.

The ultimate goal of Gradient Descent is to converge to a local (or global) minimum of the cost function, where the gradient approaches zero, indicating no further improvement in the error. Convergence depends heavily on careful selection of the learning rate and the shape of the cost function. For convex cost functions (e.g., in linear regression), there is a single global minimum, ensuring Gradient Descent will find the optimal solution if the learning rate is appropriately chosen. However, for non-convex functions (e.g., in deep neural networks), multiple local minima exist, and the algorithm may settle in a suboptimal local minimum depending on the starting point and learning rate. Techniques like learning rate scheduling (e.g., reducing the learning rate over time) or adaptive methods like Adam (Adaptive Moment Estimation) can improve convergence by dynamically adjusting the step size. In practice, monitoring the cost function’s value over iterations and using a validation set helps ensure the model converges to a solution that generalizes well, aligning with the optimization goal of minimizing error while avoiding overfitting.

Key concepts:
- Learning rate: Controls step size in parameter updates
- Cost function: Measures prediction error (typically MSE for regression)
- Convergence criteria: When to stop iteration
- Batch vs. Stochastic vs. Mini-batch approaches
- Feature scaling importance for convergence

## Scope
### Primary Objectives
- Predict life expectancy using multiple development indicators
- Optimize the prediction model using gradient descent
- Analyze the impact of different factors on health outcomes
- Understand the trade-offs between development and health

### Research Questions
1. How do economic indicators (GDP, debt) influence life expectancy?
2. What is the relationship between environmental impact (GHG emissions) and health outcomes?
3. How do labor force participation and military expenditure affect life expectancy?

## Implementation Details
### Data Preprocessing
1. Feature Selection and Engineering:
   - Target variable: Life Expectancy
   - Primary predictors: GDP per capita, GHG Emissions per Capita
   - Secondary predictors: Labor Force Participation, Military Expenditure, Debt Service
   - Feature scaling: StandardScaler for consistent gradient updates

2. Data Cleaning:
   - Handling missing values using forward fill and mean imputation
   - Removing outliers using IQR method
   - Log transformation for skewed features

3. Feature Transformation:
   - Polynomial features for non-linear relationships
   - Interaction terms between key variables
   - Standardization for consistent gradient updates

### Algorithm Configuration
1. Model Selection:
   - Custom implementation of gradient descent
   - Comparison with SGDRegressor from scikit-learn
   - Linear regression as baseline

2. Hyperparameter Tuning:
   - Learning rates: [0.001, 0.01, 0.1]
   - Maximum iterations: [1000, 5000, 10000]
   - Convergence tolerance: 1e-6
   - Mini-batch sizes: [32, 64, 128]

3. Validation Strategy:
   - Time-series cross-validation
   - Train-test split (80-20)
   - Learning curves analysis

### Visualization Approach
1. Training Diagnostics:
   - Cost function convergence plots
   - Learning rate comparison plots
   - Gradient magnitude over iterations

2. Model Insights:
   - Feature importance plots
   - Prediction vs. actual scatter plots
   - Residual analysis plots

3. Results Visualization:
   - Country-wise prediction accuracy
   - Feature contribution breakdown
   - Error distribution analysis

## Technical Details
- Library: scikit-learn (SGDRegressor)
- Algorithm Parameters:
  ```python
  SGDRegressor(
      loss='squared_error',
      learning_rate='adaptive',
      eta0=0.01,
      max_iter=1000,
      tol=1e-6,
      random_state=42
  )
  ```
- Custom Implementation:
  ```python
  def gradient_descent(X, y, learning_rate=0.01, max_iter=1000):
      weights = np.zeros(X.shape[1])
      for i in range(max_iter):
          prediction = np.dot(X, weights)
          error = prediction - y
          gradient = 2/len(X) * np.dot(X.T, error)
          weights -= learning_rate * gradient
      return weights
  ```
- Evaluation Metrics:
  - Mean Squared Error (MSE)
  - R-squared score
  - Mean Absolute Error (MAE)
  - Learning curves

## Dependencies
- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0
- wbdata>=0.3.0

## References
1. Bottou, L. (2010). Large-Scale Machine Learning with Stochastic Gradient Descent
2. Ruder, S. (2016). An overview of gradient descent optimization algorithms
3. World Bank Development Indicators Documentation
