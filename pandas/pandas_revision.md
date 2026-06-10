# Reading Data (pd.read_csv)

## What It Is
Loads tabular dataset contents from a CSV (Comma Separated Values) file into a Pandas DataFrame.

## Why It Is Used
Most structured ML datasets are stored in CSV files. `pd.read_csv` reads the files efficiently, automatically parses column headers, handles missing data representations, and structures the data for tabular manipulation.

## Common Syntax
```python
import pandas as pd

# Load CSV file into a DataFrame
df = pd.read_csv("data/data_pandas_practice.csv")
```

## Key Concepts
* **Path Resolution**: Relative or absolute file paths can be loaded.
* **Auto-Schema Inference**: Automatically detects header names and infers data types (e.g., integers, floats, strings).

## Important Notes
* Large CSVs can be memory-heavy. If needed, arguments like `usecols` (to load specific columns) can optimize performance.

## Interview / Revision Points
* How does Pandas handle missing values during read? By default, empty cells or strings like `NA`, `NaN`, `null` are parsed into `np.nan`.

---

# DataFrame and Series Basics

## What It Is
A `DataFrame` is a 2D labeled tabular data structure with columns of potentially different types (similar to a spreadsheet or SQL table). A `Series` is a 1D labeled array capable of holding any data type (equivalent to a single column or row in a DataFrame).

## Why It Is Used
Data must be organized, inspected, and formatted before feeding it into models. DataFrame and Series objects provide standard structures for operations like inspecting row/column metrics, mapping labels, and checking data types.

## Common Syntax
```python
import pandas as pd

# Creating a DataFrame from a dictionary
data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)

# Accessing a single column (returns a Series)
age_series = df['Age']

# Core inspection methods
df.head()       # View first 5 rows
df.info()       # View data types and non-null counts
df.describe()   # Summary statistics of numerical columns
```

## Key Concepts
* **Index**: Rows have a unique identifier index (default is 0-indexed integers).
* **Column Types**: Individual columns have a single data type (e.g., `int64`, `float64`, `object`).

## Important Notes
* Operations on columns usually return a Series, while operations on multiple columns (or operations that retain 2D structure) return a DataFrame.

## Interview / Revision Points
* Difference between Series and DataFrame? Series is a 1D array with an index. DataFrame is a 2D table with indexes for both rows and columns.

---

# Indexing with loc and iloc

## What It Is
Locators used to select specific rows and columns of data.
* `.loc`: Label-based indexing (selects rows/columns using names/labels).
* `.iloc`: Integer position-based indexing (selects rows/columns using numerical indices).

## Why It Is Used
Often, we need to extract specific rows or partition subsets of features (e.g., separating features `X` from target labels `y` using column indexing).

## Common Syntax
```python
# Label-based: Rows 0 to 3, column 'Age'
df.loc[0:3, 'Age']

# Position-based: Rows 0 to 2, columns 0 to 1
df.iloc[0:3, 0:2]
```

## Key Concepts
* **Inclusive vs Exclusive slicing**:
  * `.loc[start:stop]` is **inclusive** of the `stop` label.
  * `.iloc[start:stop]` is **exclusive** of the `stop` integer index (standard Python slicing behavior).
* **Row/Column partitioning**: Both indexers follow the `[row_slicing, column_slicing]` structure.

## Important Notes
* Be careful with index integer values. If your DataFrame index has non-sequential integer keys, `.loc` will match keys while `.iloc` will match numerical positions, causing unexpected selections.

## Interview / Revision Points
* Difference between `.loc` and `.iloc`? `.loc` is label-based (inclusive of bounds); `.iloc` is index position-based (exclusive of bounds).

---

# Data Filtering

## What It Is
Extracting a subset of rows from a DataFrame based on boolean conditions (boolean indexing).

## Why It Is Used
ML workflows require cleaning and filtering. For example, discarding anomalies, isolating specific target categories, or splitting datasets by custom features (e.g., filtering passengers who survived the Titanic).

## Common Syntax
```python
# Filter rows where age is greater than 30
filtered_df = df[df['age'] > 30]

# Multi-condition filtering (use bitwise operator & or |)
filtered_df = df[(df['age'] > 30) & (df['sex'] == 'female')]
```

## Key Concepts
* **Boolean Series**: Evaluating a condition like `df['age'] > 30` creates a Series of `True`/`False` flags. Passing this series to `df[...]` returns rows marked `True`.
* **Bitwise Operators**: Must use `&` (AND), `|` (OR), and `~` (NOT) instead of logical keywords `and`, `or`, `not`.

## Important Notes
* Always wrap individual condition expressions in parentheses `()` to ensure operator precedence rules are evaluated correctly.

## Interview / Revision Points
* Why do logical keywords fail in filtering? `and`/`or` evaluate truthiness of the entire Series, whereas `&`/`|` evaluate element-by-element boolean conditions.

---

# GroupBy and Aggregations

## What It Is
Partitions data into groups based on some categorical key, applies aggregation functions (sum, mean, count, min, max, etc.) to each group, and combines the results.

## Why It Is Used
Used in Exploratory Data Analysis (EDA) and feature aggregation. For instance, calculating average fare and survival rates grouped by passenger ticket class (`pclass`) to locate patterns.

## Common Syntax
```python
# Group by class and calculate average age
avg_age_by_class = df.groupby('pclass')['age'].mean()

# Multi-metric aggregation using .agg()
aggregated_df = df.groupby('pclass').agg({
    'age': 'mean',
    'fare': ['min', 'max']
})
```

## Key Concepts
* **Split-Apply-Combine**: The standard database paradigm for grouping operations.
* **Multi-Index Output**: Aggregating multiple columns with multiple functions produces multi-index headers, which can be flattened using `.columns = [...]` or `.reset_index()`.

## Important Notes
* Grouping by continuous numerical columns is rarely useful and will yield massive, uninformative tables. Stick to grouping by low-cardinality categorical variables.

## Interview / Revision Points
* How do you flatten a multi-index result after GroupBy? Use `.reset_index()` to convert grouping index labels back into normal columns.

---

# Handling Missing Values

## What It Is
Functions to detect, replace, or drop rows/columns containing missing values (`np.nan`).

## Why It Is Used
Many scikit-learn models crash if inputs contain missing values. Finding and handling missing values is a crucial first step in data preprocessing.

## Common Syntax
```python
# Check for null values in each column
df.isnull().sum()  # or df.isna().sum()

# Fill missing values with a placeholder or static value
df['age'] = df['age'].fillna(df['age'].median())

# Drop rows containing any missing value
clean_df = df.dropna()
```

## Key Concepts
* **Null Identification**: `.isnull()` and `.isna()` are aliases of the same underlying function.
* **Imputation**: Filling values (`fillna`) prevents the loss of valuable observation records.

## Important Notes
* When dropping columns/rows or filling values, Pandas creates a copy by default. Use assignments `df = df.fillna(...)` to preserve changes.

## Interview / Revision Points
* What are the primary ways to handle missing data? Dropping records (`dropna`) or imputing value estimates (`fillna`, or using `SimpleImputer`).

---

# Merging and Concatenation

## What It Is
* `pd.merge`: Combines DataFrames horizontally by matching column keys (similar to SQL Joins).
* `pd.concat`: Combines DataFrames vertically (appending rows) or horizontally (columns) along an axis.

## Why It Is Used
In real-world databases, information is normalized across separate tables. For example, matching email message bodies and spam labels using a shared transaction identifier.

## Common Syntax
```python
# Merge left and right tables along a shared key column
merged_df = pd.merge(left_df, right_df, on='email_id', how='inner')

# Concat datasets vertically (e.g. merging splits of training data)
appended_df = pd.concat([january_df, february_df], ignore_index=True)
```

## Key Concepts
* **Join Types**: `how` parameter supports `'inner'` (matches only), `'left'` (keep all left keys), `'right'` (keep all right keys), and `'outer'` (union of both).
* **Concatenation Alignment**: `pd.concat` aligns tables along column headers. Setting `ignore_index=True` prevents index duplication.

## Important Notes
* A mismatch in merge column names will cause errors. Ensure join key values contain matching data types, otherwise the merge will fail or return empty records.

## Interview / Revision Points
* What does `ignore_index=True` do in `concat`? It drops original indexes and constructs a sequential index range `0` to `N` for the combined DataFrame.

---

# One-Hot Encoding (pd.get_dummies)

## What It Is
Converts categorical string variables into binary columns (columns of `1`s and `0`s).

## Why It Is Used
Distance-based algorithms (SVM, KNN) and regression estimators cannot read text columns directly. Turning categories into binary dummy variables represents categories numerically without implying sequence hierarchy.

## Common Syntax
```python
# Encode category column
encoded_df = pd.get_dummies(df, columns=['color'])
```

## Key Concepts
* **Dummy Variable Trap**: Multicollinearity can happen if all dummy columns are included (e.g. Red, Green, Blue columns are fully dependent). Some workflows set `drop_first=True` to mitigate this.
* **Expansion**: Number of added columns is equal to the number of unique category labels in the original column.

## Important Notes
* `pd.get_dummies` is ideal for quick EDA and offline processing. For pipeline preprocessing in ML, use scikit-learn's `OneHotEncoder` to avoid feature structure mismatch on test data.

## Interview / Revision Points
* Why is `pd.get_dummies` risky for production model deployment? If the validation dataset contains categorical labels that the training set did not see, output shapes will mismatch.

---

# Binning Continuous Data (pd.cut & pd.qcut)

## What It Is
Segments continuous numerical data values into discrete intervals (bins).
* `pd.cut`: Equal-width binning (divides range into equal-size intervals).
* `pd.qcut`: Equal-frequency binning (divides range into quantiles containing equal number of data samples).

## Why It Is Used
Simplifies continuous variables to prevent noise or to transform continuous variables into categories for classification models (e.g., binning ages into 'Young', 'Adult', 'Senior').

## Common Syntax
```python
# Equal-width binning into 3 groups
df['age_group'] = pd.cut(df['age'], bins=3, labels=['Young', 'Adult', 'Old'])

# Equal-frequency binning (quantiles)
df['age_quantile'] = pd.qcut(df['age'], q=3)
```

## Key Concepts
* **pd.cut**: Good for linear intervals but sensitive to extreme outliers.
* **pd.qcut**: Ensures uniform sample density across bins.

## Important Notes
* The `labels` argument must match the number of bins. Equal-frequency binning might result in variable interval widths depending on numerical density.

## Interview / Revision Points
* Difference between `pd.cut` and `pd.qcut`? `pd.cut` divides continuous ranges into equal-length segments. `pd.qcut` divides ranges based on sample density (same sample count in each bin).

---

# Pivot Tables (pd.pivot_table)

## What It Is
Constructs a spreadsheet-style cross-tabulation of values, aggregating columns relative to custom rows and columns.

## Why It Is Used
Provides structured summaries of multidimensional metrics in EDA. For example, comparing sum or average sales across different cities and products in a single lookup grid.

## Common Syntax
```python
# Pivot table of average sales by city and product
pivot = pd.pivot_table(
    df, 
    values='sales', 
    index='city', 
    columns='product', 
    aggfunc='mean', 
    fill_value=0
)
```

## Key Concepts
* **Index & Columns**: Defines row and column dimensions of the output table.
* **fill_value**: Replaces `NaN` values where specific index-column intersection cells have no observations.

## Important Notes
* The aggregate function (`aggfunc`) defaults to `'mean'` if not explicitly declared.

## Interview / Revision Points
* What does `fill_value=0` do in `pd.pivot_table`? It replaces null values with 0 inside intersection cells that contain no data records.

---

# Transforming Data with apply and map

## What It Is
Apply mappings or user-defined functions across columns or rows.
* `.map`: Applies column-wise transformations mapping categorical keys or element-wise scalar variables.
* `.apply`: Applies a function (often a `lambda`) element-wise to columns/rows.

## Why It Is Used
Used for custom string manipulation, binary feature flags generation, or scaling values on tabular scales.

## Common Syntax
```python
# Mapping categorical categories to labels
gender_map = {'male': 0, 'female': 1}
df['sex_encoded'] = df['sex'].map(gender_map)

# Applying element-wise custom lambda modifications
df['double_age'] = df['age'].apply(lambda x: x * 2)
```

## Key Concepts
* **Element-Wise Operations**: Transformations apply to row records sequentially.
* **Vectorization Alternative**: While `.apply()` is flexible, vectorized NumPy/Pandas formulas are much faster for basic arithmetic.

## Important Notes
* Avoid using `.apply()` for simple arithmetic or standard scaling, as it is equivalent to a Python loop and runs slowly on large tables.

## Interview / Revision Points
* When should you use `.map` vs `.apply`? `.map` is best for series element-wise key-value replacement. `.apply` is best for applying arbitrary custom python function structures.
