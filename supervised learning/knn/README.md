## K-Nearest Neighbors (KNN) Analysis

## Background
K-Nearest Neighbors (KNN) implementation for classifying countries based on their development indicators. This algorithm helps us understand how countries cluster together based on their wealth, health, and environmental characteristics, providing insights into development patterns and peer groups.

## Theory
KNN is a non-parametric method used for classification and regression that:
1. Stores all available cases in n-dimensional space (where n is the number of features)
2. Classifies new cases based on a distance measure (typically Euclidean distance)
3. Uses majority vote of k nearest neighbors for classification

Key concepts:
- Distance metrics: Euclidean, Manhattan, Minkowski
- Choice of k: Balancing bias and variance
- Curse of dimensionality considerations
- Weighted voting (optional): Using distance-based weights

## Scope
### Primary Objectives
- Classify countries into development categories (High, Upper-Middle, Lower-Middle, Low)
- Identify similar countries based on multiple indicators
- Analyze the role of GHG emissions in country classification

### Research Questions
1. Which countries form natural peer groups based on development indicators?
2. How does the inclusion of GHG emissions affect country classifications?
3. What role does labor force participation play in country similarity?

## Implementation Details
### Data Preprocessing
1. Feature Selection and Engineering:
   - Primary features: GDP per capita, Life Expectancy, GHG Emissions per Capita
   - Secondary features: Labor Force Participation, Military Expenditure, Debt Service
   - Feature importance analysis using correlation matrix

2. Data Cleaning:
   - Handling missing values using forward fill and mean imputation
   - Removing outliers using IQR method
   - Standardizing features using StandardScaler

3. Feature Transformation:
   - Log transformation for GDP and GHG emissions
   - Binning continuous variables where appropriate
   - Creating composite indicators if needed

### Algorithm Configuration
1. Model Selection:
   - KNeighborsClassifier from scikit-learn
   - Distance metric: Euclidean distance
   - K selection: Using elbow method with cross-validation

2. Hyperparameter Tuning:
   - k values range: [3, 5, 7, 9, 11, 13, 15]
   - weights: ['uniform', 'distance']
   - metric: ['euclidean', 'manhattan']

3. Validation Strategy:
   - Stratified K-fold cross-validation (k=5)
   - Train-test split (80-20)

### Visualization Approach
1. Data Exploration:
   - Pair plots of key features
   - Correlation heatmap
   - Distribution plots for each feature

2. Model Insights:
   - Decision boundary plots (2D projections)
   - Neighbor distance plots
   - Confusion matrix heatmap

3. Results Visualization:
   - Country clustering map
   - Feature importance bar plots
   - Performance metrics comparison

## Technical Details
- Library: scikit-learn (KNeighborsClassifier)
- Algorithm Parameters:
  ```python
  KNeighborsClassifier(
      n_neighbors=5,
      weights='uniform',
      metric='euclidean',
      n_jobs=-1
  )
  ```
- Evaluation Metrics:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - Silhouette Score (for cluster quality)

## Dependencies
- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0
- wbdata>=0.3.0
- folium>=0.12.0 (for map visualizations)
