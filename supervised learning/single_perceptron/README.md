# Single Perceptron Analysis

## 🧠 Single Perceptron

### Background

The **Single Perceptron** is the most fundamental building block of artificial neural networks and serves as a binary classifier. In this project, the perceptron is used to classify countries into different development categories based on socio-economic and environmental indicators such as **GDP per capita**, **GHG emissions**, and **labor force participation**. This approach highlights how simple neural network structures can detect basic patterns in global development data.

Although it has a straightforward architecture, the perceptron offers critical insights into **linear separability**—the ability to separate data classes using a straight line (or hyperplane in higher dimensions). If the given indicators enable a clear boundary between high and low development countries, the perceptron can effectively model this. Its simplicity makes it not just a learning tool but also a diagnostic one—indicating whether more advanced models are required.

---

### Theory

The **Single Perceptron** is a binary classifier that performs classification based on a weighted sum of input features. Here's how it works:

1. **Takes multiple input features**: \( x_1, x_2, ..., x_n \)
2. **Computes a weighted sum** with a bias term:  

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

3. **Applies a step activation function** to produce binary output:

$$
\hat{y} =
\begin{cases}
1 & \text{if } z \geq 0 \\\\
0 & \text{otherwise}
\end{cases}
$$

4. **Updates weights using the Perceptron Learning Rule**:

$$
w_i = w_i + \eta (y - \hat{y}) x_i
$$

where:

- \( \eta \) is the learning rate  
- \( y \) is the true label  
- \( \hat{y} \) is the predicted label

This process continues iteratively until convergence—either achieving perfect classification (if the data is linearly separable) or hitting a stopping condition. While simple, the Single Perceptron is historically significant and forms the foundation of more complex neural network architectures.


The Single Perceptron is a binary classifier that:
1. Takes multiple input features and learns weights for each
2. Computes a weighted sum of inputs plus a bias term
3. Applies a step activation function to produce binary output
4. Updates weights through iterative learning using the perceptron learning rule

Key concepts:
- Linear separability
- Weight initialization and updates
- Learning rate and convergence
- Bias term
- Step activation function
- Perceptron learning rule
- Convergence guarantee for linearly separable data

## Scope
### Primary Objectives
- Implement a basic neural network classifier for development status
- Analyze linear separability of development indicators
- Study the convergence properties with real-world data
- Compare performance with logistic regression baseline

### Research Questions
1. Are development indicators linearly separable for classification?
2. How does the perceptron perform compared to other linear classifiers?
3. What insights can we gain about feature importance from learned weights?
4. How do learning rate and number of iterations affect model performance?

## Implementation Details
### Data Preprocessing
1. Feature Selection and Engineering:
   - Target variable: Development Status (binary)
   - Primary features: GDP per capita, Life Expectancy, GHG Emissions
   - Secondary features: Labor Force Participation, Military Expenditure
   - Feature scaling: StandardScaler for consistent convergence

2. Data Cleaning:
   - Handling missing values with mean imputation
   - Removing outliers using IQR method
   - Log transformation for skewed features
   - Binary encoding of development status

3. Feature Transformation:
   - Standardization of all features
   - Addition of bias term
   - Feature selection based on correlation analysis

### Algorithm Configuration
1. Model Setup:
   - Single layer perceptron
   - Binary step activation function
   - Configurable learning rate
   - Maximum iterations control
   - Early stopping on convergence

2. Model Validation:
   - Train-test split (80-20)
   - K-fold cross-validation
   - Learning curve analysis
   - Convergence monitoring

3. Diagnostics:
   - Weight evolution tracking
   - Misclassification analysis
   - Convergence verification
   - Decision boundary visualization

### Visualization Approach
1. Training Diagnostics:
   - Learning curves
   - Weight evolution plots
   - Convergence analysis
   - Error rate tracking

2. Feature Analysis:
   - Weight magnitude visualization
   - Decision boundary plots
   - Feature importance ranking
   - Misclassification patterns

3. Results Visualization:
   - Classification boundary plots
   - Performance metrics plots
   - Comparison with logistic regression
   - Feature space projections

## Technical Details
- Library: scikit-learn
- Algorithm Parameters:
  ```python
  Perceptron(
      eta0=0.1,  # learning rate
      max_iter=1000,
      tol=1e-3,
      random_state=42,
      early_stopping=True,
      validation_fraction=0.1
  )
  ```
- Model Pipeline:
  ```python
  Pipeline([
      ('scaler', StandardScaler()),
      ('classifier', Perceptron())
  ])
  ```
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Number of iterations to convergence
  - Final weight values

## Dependencies
- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0
- wbdata>=0.3.0

## References
1. Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain
2. Minsky, M., & Papert, S. (1969). Perceptrons: An Introduction to Computational Geometry
3. World Bank Development Indicators Documentation
4. Bishop, C. M. (2006). Pattern Recognition and Machine Learning
