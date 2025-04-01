# Linear Regression Analysis

## Visualization Example

![Linear Regression Visualization](/images/linear_reg.png)

This image illustrates the relationship between GDP per capita and life expectancy, showcasing the fitted regression line and data points. It provides a visual representation of how the model captures trends in the data.
## Background
Linear Regression is one of the most foundational tools in data science, used to examine how one or more independent variables influence a dependent variable. In this project, it helps us understand how GDP per capita, GHG emissions per capita, and labor force participation impact life expectancy across countries. By fitting a straight line to the data, the model enables us to interpret how much each factor contributes to changes in life expectancy. This simplicity and interpretability make Linear Regression a great starting point for modeling complex relationships in development and health data.

The model outputs a set of coefficients, each corresponding to an input feature, which indicates the average change in life expectancy associated with a one-unit change in that feature, holding all others constant. For example, a positive coefficient for GDP would suggest that wealthier nations, on average, experience higher life expectancy. The clarity of these coefficients makes Linear Regression a powerful explanatory tool, even before diving into more complex algorithms like Gradient Descent or non-linear models.


## Theory

Linear Regression is a **parametric, supervised learning algorithm** that seeks to model the relationship between a dependent variable (in our case, life expectancy) and one or more independent variables (like GDP, GHG emissions, and labor participation). It assumes that this relationship can be described by a linear equation of the form:  
$$
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n
$$
where \( \beta_0 \) is the intercept, and \( \beta_1, \beta_2, ..., \beta_n \) are the coefficients for each feature.

To find the optimal values of these coefficients, Linear Regression minimizes the **sum of squared residuals**, which are the squared differences between the observed and predicted values. This process is also known as **Ordinary Least Squares (OLS)**. The resulting model provides interpretable coefficients that quantify the contribution of each feature to the outcome. However, for the model to yield valid insights, certain assumptions must be met: the relationship between features and target should be linear, residuals should be independent, have constant variance (homoscedasticity), and be normally distributed. Violating these assumptions may lead to biased or inefficient results.

Linear Regression is a parametric supervised learning algorithm that:
1. Models the relationship between dependent and independent variables using a linear equation
2. Minimizes the sum of squared residuals to find optimal parameters
3. Provides interpretable coefficients representing the impact of each feature
4. Assumes linearity, independence, homoscedasticity, and normality of residuals

Key concepts:
- Ordinary Least Squares (OLS) estimation
- R-squared and adjusted R-squared for model fit
- Multicollinearity and VIF (Variance Inflation Factor)
- Feature scaling and transformation
- Residual analysis and diagnostics

## Scope
### Primary Objectives
- Model life expectancy using multiple development indicators
- Quantify the impact of each factor on health outcomes
- Analyze interactions between economic and environmental variables
- Provide interpretable insights for policy decisions

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
   - Feature scaling: StandardScaler for consistent model performance

2. Data Cleaning:
   - Handling missing values using forward fill and mean imputation
   - Removing outliers using IQR method
   - Log transformation for skewed features (GDP, GHG emissions)

3. Feature Transformation:
   - Polynomial features for non-linear relationships
   - Interaction terms between key variables
   - One-hot encoding for categorical variables (if any)

### Algorithm Configuration
1. Model Selection:
   - Linear Regression with OLS
   - Ridge Regression for regularization
   - Polynomial features for non-linear relationships

2. Model Validation:
   - Cross-validation (k=5 folds)
   - Train-test split (80-20)
   - Residual analysis

3. Diagnostics:
   - Multicollinearity check using VIF
   - Heteroscedasticity tests
   - Normality tests for residuals
   - Influence analysis (Cook's distance)

### Visualization Approach
1. Model Diagnostics:
   - Q-Q plots for normality
   - Residual plots
   - Scale-location plots
   - Leverage plots

2. Feature Analysis:
   - Correlation matrix heatmap
   - Partial regression plots
   - Feature importance plots

3. Results Visualization:
   - Actual vs. predicted scatter plots
   - Residual distribution plots
   - Coefficient magnitude plots

## Technical Details
- Library: scikit-learn
- Algorithm Parameters:
  ```python
  LinearRegression(
      fit_intercept=True,
      normalize=False,
      copy_X=True,
      n_jobs=None
  )
  ```
- Model Pipeline:
  ```python
  Pipeline([
      ('scaler', StandardScaler()),
      ('poly', PolynomialFeatures(degree=2)),
      ('regressor', LinearRegression())
  ])
  ```
- Evaluation Metrics:
  - R-squared score
  - Adjusted R-squared
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

## Dependencies
- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0
- wbdata>=0.3.0
- statsmodels>=0.13.0

## References
1. James, G., et al. (2013). An Introduction to Statistical Learning
2. World Bank Development Indicators Documentation
3. Kutner, M. H., et al. (2004). Applied Linear Statistical Models
