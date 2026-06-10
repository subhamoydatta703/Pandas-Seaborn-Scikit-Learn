# Creating Arrays with np.array

## What It Is
An array is the fundamental N-dimensional data structure in NumPy. In the repository's practice exercises, it is primarily used to construct 2D input matrices (`X`) and 1D label vectors (`y`) for models.

## Why It Is Used
Scikit-learn estimators expect inputs to be in numerical matrix format (typically a 2D array of shape `(n_samples, n_features)` for features, and a 1D array of shape `(n_samples,)` for target labels). `np.array` provides a quick way to build these representations manually for training and testing.

## Common Syntax
```python
import numpy as np

# 2D feature matrix (shape N x 1)
X = np.array([[1], [2], [3], [4], [5]])

# 1D target label vector
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
```

## Key Concepts
* **Dimension Matching**: Features are represented as a column vector (shape `(n, 1)`) because models treat each column as a separate feature.
* **Vectorized Inputs**: Arrays enable fast mathematical computations under the hood without using slow Python `for` loops.

## Important Notes
* Passing a 1D array (e.g. `[1, 2, 3]`) as features to a scikit-learn estimator will throw a `ValueError` asking you to reshape your data using `.reshape(-1, 1)`. Always wrap features in a nested list to maintain a 2D structure.

## Interview / Revision Points
* Why does scikit-learn require a 2D array for features? It expects a matrix structure where rows represent samples and columns represent features, even if there is only a single feature.

---

# Representing Missing Data with np.nan

## What It Is
`np.nan` (Not a Number) is a special floating-point representation used in NumPy and Pandas to denote missing, null, or undefined data.

## Why It Is Used
In real-world machine learning, data often contains missing values. Python's default `None` is difficult to perform math on. `np.nan` allows numerical arrays and Pandas DataFrames to retain numerical data types while marking missing entries for statistical imputation.

## Common Syntax
```python
import numpy as np
import pandas as pd

# Creating a Series with missing scores
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Score': [85, np.nan, 78, 95, np.nan]
}
df = pd.DataFrame(data)
```

## Key Concepts
* **Data Imputation Pre-requisite**: Scikit-learn's `SimpleImputer` looks specifically for `np.nan` (by default) to locate and replace missing entries.
* **IEEE 754 Floating-Point Standard**: `np.nan` is classified as a float, allowing columns containing it to remain numeric.

## Important Notes
* `np.nan == np.nan` evaluates to `False`. To check for missing values, use `.isnull()`, `.isna()`, or `np.isnan(value)` rather than direct equality comparisons.

## Interview / Revision Points
* How do you identify `np.nan` in a Pandas DataFrame? Use `.isnull()` or `.isna()`. Never use `df['col'] == np.nan`.

---

# Calculating Roots with np.sqrt

## What It Is
A mathematical function that calculates the square root of an array elements or numeric scalar.

## Why It Is Used
Used in evaluating regression models to calculate the **Root Mean Squared Error (RMSE)**. Scikit-learn's metrics package provides `mean_squared_error`, which returns the squared error. Taking the square root of this value converts the metric back to the target feature's original scale, making error analysis intuitive.

## Common Syntax
```python
import numpy as np
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
```

## Key Concepts
* **Unit Alignment**: RMSE is in the exact same unit as the target variable `y` (e.g., dollars, degrees, ages), whereas MSE is in squared units.
* **Vectorized Processing**: `np.sqrt` can be applied to entire arrays of numbers at once.

## Important Notes
* In newer versions of scikit-learn, you can pass `root_mean_squared_error` directly, but taking `np.sqrt(mse)` is the classic and highly compatible method used across the repository.

## Interview / Revision Points
* Why use RMSE over MSE? RMSE aligns with the units of the target variable, making the error margin easier to interpret for business stakeholders.

---

# Bounding Predictions with np.clip

## What It Is
A utility function that limits (clips) values in an array/scalar within a specified interval. Values smaller than the minimum are set to the minimum, and values larger than the maximum are set to the maximum.

## Why It Is Used
Linear models can sometimes output predictions that are mathematically valid but logically impossible (e.g., predicting a student grade of 115% or -10% on a test). `np.clip` bounds these model predictions to realistic scales (e.g., 0 to 100).

## Common Syntax
```python
import numpy as np

# Constraining predictions to the range [0, 100]
predictions = np.array([-5, 45, 92, 108])
bounded_predictions = np.clip(predictions, 0, 100)
# Output: array([0, 45, 92, 100])
```

## Key Concepts
* **Sanity Checks**: Acts as a hard boundary layer at the end of the prediction pipeline.
* **Preserves Trends**: Binds outliers without altering the order or relationship of in-range predictions.

## Important Notes
* Clipping is a post-processing tool. If your model consistently predicts values far outside logical bounds, you should inspect your training features or model assumptions instead of relying solely on clipping.

## Interview / Revision Points
* What does `np.clip` do? It clamps array values to a user-defined minimum and maximum range.

---

# Generating Ranges for Custom Visualization with np.arange

## What It Is
Generates evenly spaced values within a given interval, returning a NumPy array.

## Why It Is Used
Used in data visualization workflows to define precise tick marks and intervals on graph axes (e.g., modifying the x-axis of a Seaborn histogram to display ticks at every 5-year interval instead of letting the library select arbitrary tick marks).

## Common Syntax
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=df, x="age")
# Set x-axis tick marks from 0 to 80, increments of 5
plt.xticks(np.arange(0, 81, 5))
plt.show()
```

## Key Concepts
* **Inclusive/Exclusive bounds**: `np.arange(start, stop, step)` is inclusive of `start` but exclusive of `stop`. To get an interval up to 80, you must set `stop` to at least 81.
* **Float Compatibility**: Unlike Python's standard `range()`, `np.arange` supports step increments that are floating-point values.

## Important Notes
* Ensure that the step size is non-zero, otherwise, NumPy will raise a `ZeroDivisionError` or create an infinite loop.

## Interview / Revision Points
* How does `np.arange(0, 81, 5)` behave compared to standard Python `range`? It returns a NumPy array and supports fractional/floating-point steps.
