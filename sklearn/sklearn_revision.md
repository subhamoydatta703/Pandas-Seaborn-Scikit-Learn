# Dataset Splitting (train_test_split)

## What It Is
Splits dataset features and target labels into random training and testing sets.

## Why It Is Used
To evaluate how well our model generalizes to unseen data. It isolates a holdout test set (e.g. 20% of data) that is never seen during model training, protecting against overfitting.

## Common Syntax
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## Key Concepts
* **Test Size**: Determines the ratio allocated to the test set (usually 0.2 or 0.3).
* **Random State**: Seed value ensuring identical splits every time the script runs.
* **Stratified Splitting**: Keeps identical class label ratios in both splits, preventing bias when dealing with imbalanced categorical datasets.

## Important Notes
* Always split your dataset *before* applying transformers (like scaling or imputing) to avoid data leakage (information from the test set leaking into the training step).

## Interview / Revision Points
* Why stratify class labels? It ensures that minority classes are represented proportionally in both train and test splits.

---

# Data Imputation (SimpleImputer)

## What It Is
Fills missing numerical or categorical data values using statistical methods.

## Why It Is Used
Most scikit-learn models crash if inputs contain missing values. Imputation retains samples that would otherwise have to be discarded.

## Common Syntax
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

## Key Concepts
* **Strategies**: Fills missing values with `'mean'`, `'median'`, `'most_frequent'` (mode), or a `'constant'` value.
* **fit vs transform**: `fit` computes the imputation statistics (e.g. mean) on the training set. `transform` applies that pre-calculated statistic to fill null values in both train and test sets.

## Important Notes
* Never call `fit_transform` on the test set. This is a common data leakage mistake. The imputer should only ever learn replacement statistics from the training split.

## Interview / Revision Points
* Why not fit the imputer on test data? Fitting on the test set leaks future information (like test mean) into the training workflow.

---

# Categorical Encoding (OneHotEncoder)

## What It Is
Converts text category features into binary columns (columns of `1`s and `0`s).

## Why It Is Used
ML models require numerical input. Standard category names (e.g., Red, Blue) have no mathematical order; label encoding would imply Red (1) < Blue (2), which is incorrect. One-hot encoding creates independent binary features for each label category.

## Common Syntax
```python
from sklearn.preprocessing import OneHotEncoder

# Initialize encoder returning a dense array
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_encoded = ohe.fit_transform(X_train[['color']])
```

## Key Concepts
* **Sparse vs Dense Output**: `sparse_output=False` outputs a standard NumPy array; `True` outputs a sparse matrix to conserve memory on columns with massive categories.
* **handle_unknown**: Setting `'ignore'` prevents crashes if new categories appear in the test set.

## Important Notes
* Unlike Pandas `get_dummies`, scikit-learn's `OneHotEncoder` fits on training data and retains the exact same column structure when transforming test data, preventing column mismatches.

## Interview / Revision Points
* Why use `handle_unknown='ignore'`? It ensures the model does not crash if it encounters unknown categories during inference, encoding them as all zeros instead.

---

# Feature Scaling (StandardScaler)

## What It Is
Standardizes numerical variables by centering their values around a mean of 0 and scaling them to a standard deviation of 1 (Z-score normalization).

## Why It Is Used
Gradient descent optimization and distance-based algorithms (like regularized Logistic Regression or Support Vector Machines) are scale-sensitive. If features are on different scales (e.g. age 0-80 vs income 10k-100k), larger features will dominate model updates.

## Common Syntax
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Key Concepts
* **Standardization**: Formula: $z = \frac{x - \mu}{\sigma}$ (Z-score).
* **Scaling Contrast (MinMaxScaler)**: While `StandardScaler` standardizes unit variance, `MinMaxScaler` shifts values to fit a fixed range $[0, 1]$. `MinMaxScaler` is best for bounded ranges, but `StandardScaler` is generally preferred when features contain outliers.

## Important Notes
* Scaling does not change the distribution shape of your features (a skewed histogram will remain skewed); it only updates feature values to a uniform scale.

## Interview / Revision Points
* Why scale features? It prevents features with large magnitudes from dominating updates and helps models converge faster.

---

# Column Preprocessing (ColumnTransformer)

## What It Is
Applies different preprocessing pipelines (transformers) to specific subsets of columns in parallel.

## Why It Is Used
A typical dataset contains different column types (e.g., categorical text features and continuous numerical variables). `ColumnTransformer` lets you apply different steps (like `OneHotEncoder` to categories and `StandardScaler` to numbers) in a clean, unified preprocessing block.

## Common Syntax
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'fare']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['embarked'])
    ]
)
```

## Key Concepts
* **Parallel Pipelines**: Each transformation pathway executes independently on its respective columns.
* **Output Concatenation**: Automatically joins the scaled and encoded columns back together horizontally.

## Important Notes
* Unspecified columns are discarded by default. If you want to keep them unmodified, set `remainder='passthrough'`.

## Interview / Revision Points
* What is the purpose of `ColumnTransformer`? It enables target-specific transformations on different column types within a single preprocess step.

---

# End-to-End Workflows (Pipeline)

## What It Is
Chains multiple preprocessing transformers and a final estimator model together in sequential order.

## Why It Is Used
Keeps code modular and organized. It prevents data leakage by ensuring all preprocessing steps (scaling, encoding) are fit *only* on the training split, and simplifies predictions on new data.

## Common Syntax
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Construct pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# Fit pipeline on raw training splits
pipeline.fit(X_train, y_train)

# Evaluate pipeline directly on raw test splits
score = pipeline.score(X_test, y_test)
```

## Key Concepts
* **Pipeline Steps**: Declared as a list of tuples containing step names and transformer/model instances.
* **Pipeline Execution**: Calling `.fit()` on the pipeline sequentially calls `fit_transform` on transformers, and then calls `fit` on the final model estimator.

## Important Notes
* The final step of a pipeline must be an estimator (model), while all preceding steps must be data transformers.

## Interview / Revision Points
* How does a Pipeline prevent data leakage? It guarantees that transformers learn parameters (e.g. mean, variance) from the training set only, applying the same parameters to test data.

---

# Linear Regression (LinearRegression)

## What It Is
A model that predicts continuous continuous values by fitting a linear equation to the features.

## Why It Is Used
A simple baseline model for regression tasks. It is highly interpretable, fast to train, and works well when target relationships are linear.

## Common Syntax
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Key Concepts
* **coefficients & intercept**: Access coefficients using `model.coef_` and intercept using `model.intercept_`.
* **Ordinary Least Squares (OLS)**: Fits the line by minimizing the sum of squared differences (residuals) between predictions and actual values.

## Important Notes
* Linear Regression is sensitive to outliers, which can pull the line of best fit away from the majority of data points.

## Interview / Revision Points
* How does Linear Regression learn? It adjusts weights to minimize the mean squared error (residual sum of squares) of the predictions.

---

# Logistic Regression (LogisticRegression)

## What It Is
A linear model wrapped in a sigmoid function that predicts categorical outcomes (probabilities from 0 to 1).

## Why It Is Used
The classic baseline model for binary and multiclass classification tasks. It is fast, regularizable, and outputs prediction probabilities.

## Common Syntax
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict class labels
predictions = model.predict(X_test)

# Predict probabilities (outputs class probability array)
prob_predictions = model.predict_proba(X_test)
```

## Key Concepts
* **Sigmoid Function**: Transforms linear outputs $z$ into probabilities: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
* **Regularization Strength (C)**: Parameter tuning regularization. Smaller `C` values apply stronger regularization (preventing overfitting).

## Important Notes
* Requires feature scaling (`StandardScaler`) because it uses regularization, which penalizes coefficient sizes.

## Interview / Revision Points
* What is the difference between `predict` and `predict_proba`? `predict` outputs final class labels. `predict_proba` outputs probabilities for each target class.

---

# Decision Tree Classification (DecisionTreeClassifier)

## What It Is
A non-parametric classification model that partitions data into leaf nodes using sequential conditional splits.

## Why It Is Used
Capable of modeling complex non-linear relationships and feature interactions without requiring scaled inputs or category encoding.

## Common Syntax
```python
from sklearn.tree import DecisionTreeClassifier

# Initialize classifier with depth constraints
model = DecisionTreeClassifier(max_depth=3, min_samples_split=5)
model.fit(X_train, y_train)
```

## Key Concepts
* **Information Gain**: Splits are decided based on maximizing information gain (reducing Gini impurity or entropy).
* **Hyperparameter Control**: Features like `max_depth` and `min_samples_split` are tuned to prevent tree overfitting.

## Important Notes
* Unconstrained decision trees will split until leaves are pure, causing extreme overfitting. Always set depth constraints (like `max_depth`) or prune trees.

## Interview / Revision Points
* Why are Decision Trees prone to overfitting? If unconstrained, they will build deep trees that learn noise and outliers in training data.

---

# Regression Metrics (MAE, MSE, RMSE, R-squared)

## What It Is
Mathematical evaluations measuring predicted continuous deviations from actual labels.

## Why It Is Used
Continuous values require distance metrics to measure prediction accuracy.

## Common Syntax
```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
```

## Key Concepts
* **Mean Absolute Error (MAE)**: Average absolute distance. Interpretable, treats all errors equally.
* **Mean Squared Error (MSE)**: Average squared distance. Penalizes large errors heavily.
* **Root Mean Squared Error (RMSE)**: Square root of MSE. Standard error metric on the target scale.
* **R-squared ($R^2$)**: Proportion of variance explained by features. Closer to 1 is better.

## Important Notes
* MAE is robust to outliers, while MSE and RMSE are sensitive to outliers because errors are squared.

## Interview / Revision Points
* What is $R^2$? It is the coefficient of determination, indicating the percentage of target variance captured by the model features.

---

# Classification Metrics (Accuracy, Precision, Recall, F1)

## What It Is
Evaluation metrics measuring classification performance.

## Why It Is Used
Accuracy can be misleading on imbalanced datasets. We need multi-dimensional metrics (Precision, Recall) to evaluate models based on different requirements (e.g. minimizing false negatives in disease prediction).

## Common Syntax
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions)
rec = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
```

## Key Concepts
* **Accuracy**: Proportion of overall correct predictions.
* **Precision**: True Positives / (True Positives + False Positives). Best when false positives are costly.
* **Recall (Sensitivity)**: True Positives / (True Positives + False Negatives). Best when false negatives are costly (e.g. disease detection).
* **F1-Score**: Harmonic mean of Precision and Recall.

## Important Notes
* If your positive class is rare, a model that always predicts negative can have high accuracy. Use F1-score or Recall instead to check true performance.

## Interview / Revision Points
* When is Recall more important than Precision? In medical diagnostics, where missing a sick patient (false negative) is far more dangerous than flagging a healthy one (false positive).

---

# Cross Validation (cross_val_score)

## What It Is
An evaluation method that splits training datasets into $K$ folds, training and validating models iteratively.

## Why It Is Used
A single train/test split can result in a high-variance estimate of model performance. Cross-validation evaluates performance across different folds, giving a robust estimate of generalization error.

## Common Syntax
```python
from sklearn.model_selection import cross_val_score

# Evaluate pipeline across 5 folds
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
mean_score = scores.mean()
```

## Key Concepts
* **K-Fold**: Dataset is divided into $K$ equal parts. The model trains on $K-1$ folds and validates on the remaining fold, repeating the process $K$ times.
* **Variance Reduction**: Provides the mean performance score and standard deviation across splits.

## Important Notes
* Always pass your end-to-end `Pipeline` into `cross_val_score` rather than a raw model to prevent data leakage during fold splits.

## Interview / Revision Points
* Why cross-validate with a pipeline? If you scale or impute data before cross-validation, information from validation folds leaks into training folds. Passing the pipeline ensures proper split isolation.

---

# Hyperparameter Tuning (GridSearchCV)

## What It Is
Systematically fits models across a parameter grid, identifying the optimal hyperparameter combination using cross-validation.

## Why It Is Used
Manually tuning hyperparameters is slow and inefficient. `GridSearchCV` automates this search, ensuring selected parameters generalize well.

## Common Syntax
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid (use double underscore for pipeline step variables)
param_grid = {
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 5, 10]
}

# Initialize search
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

# Retrieve metrics
best_model = grid.best_estimator_
best_params = grid.best_params_
```

## Key Concepts
* **Parameter Naming**: For pipelines, parameters are declared using `stepName__parameterName` (e.g., `model__max_depth`).
* **Fit and Refit**: Automatically trains on parameter grids, then refits the best parameters on the full dataset.

## Important Notes
* Grid search fits the model $K$ (folds) * $P$ (combinations) times, which can be computationally expensive on large datasets.

## Interview / Revision Points
* How does `GridSearchCV` find the best hyperparameters? It searches through the cross-product of parameter grids and selects the combination that yields the highest average validation score across folds.
