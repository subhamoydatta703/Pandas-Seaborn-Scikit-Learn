# Scikit-Learn and Pandas Learning Repository

Beginner-friendly notebooks for practicing data manipulation with Pandas and machine learning workflows with Scikit-Learn.

## Project Structure

```text
.
|-- data/
|   `-- data_pandas_practice.csv
|-- pandas/
|   |-- pandas_beginner.ipynb
|   |-- pandas_intermediate.ipynb
|   `-- pd_bg_prac1.ipynb
|-- sklearn/
|   |-- day0.ipynb
|   |-- day1.ipynb
|   |-- day2.ipynb
|   `-- sklearn_day1.ipynb
|-- requirements.txt
`-- README.md
```

## Notebooks

### Pandas

- `pandas/pandas_beginner.ipynb`: DataFrame and Series basics, CSV loading, exploration, selection, and indexing.
- `pandas/pandas_intermediate.ipynb`: Grouping, aggregation, custom functions, joins, concatenation, encoding, binning, and pivot tables.
- `pandas/pd_bg_prac1.ipynb`: Extra practice with a larger email dataset, including inspection, column selection, and `loc`/`iloc`.

### Scikit-Learn

- `sklearn/day0.ipynb`: Introductory Scikit-Learn workflow covering models, train/test split, prediction, scoring, transformers, `StandardScaler`, `OneHotEncoder`, and `SimpleImputer`.
- `sklearn/sklearn_day1.ipynb`: Linear Regression, Decision Tree Classification, accuracy scoring, cross-validation, stratified splitting, and data leakage prevention.
- `sklearn/day1.ipynb`: Focused practice with Linear Regression predictions and regression evaluation metrics: MAE, MSE, RMSE, and R-squared.
- `sklearn/day2.ipynb`: Classification labels, Logistic Regression, prediction probabilities, model scoring, and when transformers are needed for feature preparation.

## Getting Started

### Prerequisites

- Python 3.12 or another recent Python 3 version
- JupyterLab or another notebook environment

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd skitlearn_pandas
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start JupyterLab:

   ```bash
   jupyter lab
   ```

## Suggested Learning Order

1. `pandas/pandas_beginner.ipynb`
2. `pandas/pandas_intermediate.ipynb`
3. `pandas/pd_bg_prac1.ipynb`
4. `sklearn/day0.ipynb`
5. `sklearn/sklearn_day1.ipynb`
6. `sklearn/day1.ipynb`
7. `sklearn/day2.ipynb`

## Data Notes

- `data/data_pandas_practice.csv` is included and used by the Pandas beginner notebook.
- `pandas/pandas_intermediate.ipynb`, `sklearn/day0.ipynb`, `sklearn/sklearn_day1.ipynb`, `sklearn/day1.ipynb`, and `sklearn/day2.ipynb` use small in-notebook examples.
- `pandas/pd_bg_prac1.ipynb` expects a larger `data/emails.csv` file. Add it locally before running that notebook if you have the dataset.

## Topics Covered

### Pandas

- Loading CSV files
- Exploring DataFrames and Series
- Indexing and slicing with `loc` and `iloc`
- Basic statistics and data inspection
- Grouping and aggregation with `groupby` and `agg`
- Applying functions with `lambda`, `apply`, and `map`
- Combining DataFrames with `merge`, `join`, and `concat`
- Encoding categories with `get_dummies`
- Binning values with `cut` and `qcut`
- Summarizing data with `pivot_table`

### Scikit-Learn

- Train/test splitting
- Linear Regression
- Logistic Regression
- Decision Tree Classification
- Classification labels
- Prediction workflows
- Prediction probabilities
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (`R2`) scoring
- Accuracy scoring
- Cross-validation
- Stratified sampling
- Transformers with `fit`, `transform`, and `fit_transform`
- Feature scaling with `StandardScaler`
- Categorical encoding with `OneHotEncoder`
- Missing value filling with `SimpleImputer`
- Overfitting and data leakage prevention
