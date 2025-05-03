# **INDE 577 - Spring 2025**

# The Tale of Wealth, Health, and the Planet

![##](/images/tale.png)
## Project Overview
This project explores the complex relationships between wealth, health, and environmental impact across countries using machine learning. We analyze how economic development, measured by GDP per capita, influences health outcomes (life expectancy) and environmental impact (GHG emissions per capita), while considering additional factors like labor force participation and military expenditure.

## The Story
Our journey begins with a fundamental question: How do wealth, health, and environmental sustainability interact in the global landscape? Through machine learning, we uncover patterns that reveal:

1. **The Wealth-Health Nexus**: How economic prosperity affects life expectancy, and how labor force participation mediates this relationship.
2. **The Environmental Impact**: How development patterns influence GHG emissions, including the role of military expenditure.
3. **Development Pathways**: Distinct patterns in how countries balance economic growth, health outcomes, and environmental sustainability.

## Algorithms Implemented

### Supervised Learning

#### 1. K-Nearest Neighbors (KNN)
- **Purpose**: Classify countries based on development indicators
- **Key Features**: 
  - Uses k=5 neighbors with Euclidean distance
  - StandardScaler for feature normalization
  - Evaluates country similarities in development space

#### 2. Gradient Descent
- **Purpose**: Optimize predictions of life expectancy
- **Key Features**:
  - Learning rate: 0.01
  - Max iterations: 1000

#### 3. Linear Regression
- **Purpose**: Model relationships between development indicators
- **Key Features**:
  - R² evaluation metric
  - Feature importance analysis
  - Residual analysis

#### 4. Logistic Regression
- **Purpose**: Classify countries into income levels
- **Key Features**:
  - Binary classification (high/low income)
  - L2 regularization
  - ROC-AUC evaluation

#### 5. Single Perceptron
- **Purpose**: Basic neural network for health status classification
- **Key Features**:
  - Binary classification
  - Linear decision boundary
  - Convergence analysis

#### 6. Multi-Layer Perceptron (MLP)
- **Purpose**: Capture non-linear patterns in development data
- **Key Features**:
  - Hidden layers: (100, 50)
  - ReLU activation
  - Dropout for regularization

#### 7. Decision Trees
- **Purpose**: Model hierarchical development decisions
- **Key Features**:
  - Max depth: 5
  - Gini impurity criterion
  - Feature importance visualization

#### 8. Random Forests
- **Purpose**: Predict GHG emissions from development indicators
- **Key Features**:
  - 100 estimators
  - Feature importance analysis
  - Out-of-bag error estimation

#### 9. Ensemble Learning
- **Purpose**: Combine multiple models for robust predictions
- **Key Features**:
  - Voting classifier
  - Model diversity analysis
  - Performance comparison

### Unsupervised Learning

#### 10. K-Means Clustering
- **Purpose**: Identify country clusters based on development patterns
- **Key Features**:
  - 3 clusters (elbow method)
  - Silhouette score evaluation
  - Cluster visualization

## Technical Implementation

### Data Collection
```python
import wbdata
import pandas as pd

   indicators = {
        'NY.GDP.PCAP.KD': 'GDP per capita',
        'SP.DYN.LE00.IN': 'Life Expectancy',
        'EN.GHG.ALL.MT.CE.AR5': 'CO2 Emissions per Capita',
        'SL.TLF.CACT.ZS': 'Labor Force Participation',
        'MS.MIL.XPND.GD.ZS': 'Military Expenditure',
        'DT.TDS.DPPF.XP.ZS': 'Debt Service'
    }
```

### Data Preprocessing
- Handle missing values
- Scale features
- Create derived indicators (e.g., GHG emissions per capita)
- Encode categorical variables

### Model Evaluation
- Cross-validation
- Performance metrics (accuracy, R², MSE)
- Visualization of results
- Feature importance analysis

## Dependencies
```requirements.txt
# Core Data Science Libraries
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=0.24.0

# World Bank Data Access
wbdata>=0.3.0

# Visualization
graphviz>=0.16
dtreeviz>=2.2.2

# Deep Learning
tensorflow>=2.8.0

# Development Tools
black>=21.5b2
flake8>=3.9.0
pytest>=6.2.5
```

## Setup Instructions
1. Create virtual environment:
   ```bash
   python setup_env.py
   ```

2. Activate environment:
   - Windows: `venv\Scripts\activate`
   - Unix/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## Project Structure
```
INDE577_WDI/
├── README.md
├── graph_theory/
│   ├── ZeroWeightEdges_GlobalDevNetworks.ipynb
├── images/
├── Income_Group_Project_package/
│   ├── Predicting_Income_Groups_Notebook.ipynb
│   ├── ml_package/
├── supervised_learning/
│   ├── knn/
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── random_forests/
│   ├── ensemble_learning/
├── unsupervised_learning/
│   ├── dbscan/
│   │   ├── dbscan_notebook.ipynb
│   ├── kmeans/
```


## References
1. World Bank World Development Indicators
2. Scikit-learn Documentation
3. Machine Learning Literature on Development Studies
