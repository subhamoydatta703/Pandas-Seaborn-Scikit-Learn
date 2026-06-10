# Scikit-Learn and Pandas Practice

Beginner-friendly Jupyter notebooks for learning Pandas data manipulation and Scikit-Learn machine learning workflows.

The repository is organized as a hands-on study path: start with basic DataFrame operations, move into intermediate Pandas transformations, then practice model training, evaluation, preprocessing, and classification metrics.

## Learning Goals

By working through the notebooks, you will practice:

- Reading, inspecting, filtering, and transforming tabular data with Pandas.
- Preparing categorical, numerical, and missing-value features for machine learning.
- Training beginner-friendly regression and classification models with Scikit-Learn.
- Evaluating models with common regression and classification metrics.
- Building reusable preprocessing and modeling workflows with `ColumnTransformer` and `Pipeline`.

## Project Structure

```text
.
|-- data/
|   `-- data_pandas_practice.csv
|-- pandas/
|   |-- pandas_beginner.ipynb
|   |-- pandas_intermediate.ipynb
|   `-- pd_bg_prac1.ipynb
|-- seaborn/
|   `-- day1.ipynb
|-- sklearn/
|   |-- day0.ipynb
|   |-- day1.ipynb
|   |-- day2.ipynb
|   |-- day3.ipynb
|   |-- day4.ipynb
|   |-- day5.ipynb
|   |-- project1_disease_detection_model.ipynb
|   `-- sklearn_day1.ipynb
|-- requirements.txt
`-- README.md
```

## Notebooks

### Pandas

- `pandas/pandas_beginner.ipynb`: DataFrame and Series basics, CSV loading, data inspection, column selection, and `loc`/`iloc` indexing.
- `pandas/pandas_intermediate.ipynb`: Grouping, aggregation, custom functions, joins, concatenation, encoding, binning, pivot tables, `apply`, and `map`.
- `pandas/pd_bg_prac1.ipynb`: Extra Pandas practice with a larger email dataset, including inspection, column selection, and row/column indexing.

### Seaborn

- `seaborn/day1.ipynb`: Seaborn basics, loading data, creating distribution, count, and box plots (`countplot`, `histplot`, `boxplot`), adjusting bins/ticks, and analyzing outliers using the Titanic dataset.

### Scikit-Learn

- `sklearn/day0.ipynb`: Introductory Scikit-Learn workflow covering models, train/test splitting, prediction, scoring, transformers, `StandardScaler`, `OneHotEncoder`, and `SimpleImputer`.
- `sklearn/sklearn_day1.ipynb`: Linear Regression, Decision Tree Classification, accuracy scoring, cross-validation, stratified splitting, and data leakage prevention.
- `sklearn/day1.ipynb`: Focused Linear Regression practice with regression metrics: MAE, MSE, RMSE, and R-squared.
- `sklearn/day2.ipynb`: Classification labels, Logistic Regression, prediction probabilities, accuracy, precision, recall, and when transformers are needed for feature preparation.
- `sklearn/day3.ipynb`: ColumnTransformer practice for applying different preprocessing steps to categorical and numerical columns with `OneHotEncoder` and `StandardScaler`.
- `sklearn/day4.ipynb`: Pipeline practice combining `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, and `LogisticRegression`, then using `GridSearchCV` to tune model hyperparameters, inspect the best model, and make new-student predictions.
- `sklearn/day5.ipynb`: Decision Tree Classification practice, tuning model hyperparameters (such as `max_depth` and `min_samples_split`) using `GridSearchCV` in a complete ML workflow, and checking model evaluation metrics (accuracy, precision, recall, and F1-score).
- `sklearn/project1_disease_detection_model.ipynb`: Beginner classification project that trains a Logistic Regression model to predict disease status from age, blood pressure, and cholesterol values, then checks accuracy, precision, recall, prediction counts, and a new-patient prediction.

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

4. Optional: register the virtual environment as a notebook kernel:

   ```bash
   python -m ipykernel install --user --name skitlearn_pandas --display-name "Python (skitlearn_pandas)"
   ```

5. Start JupyterLab:

   ```bash
   jupyter lab
   ```

6. Open the notebooks in the suggested order below.

## Dependencies

The project dependencies are listed in `requirements.txt`:

- `numpy`
- `pandas`
- `scikit-learn`
- `ipykernel`
- `jupyterlab`
- `seaborn`

## Suggested Learning Order

1. `pandas/pandas_beginner.ipynb`
2. `pandas/pandas_intermediate.ipynb`
3. `pandas/pd_bg_prac1.ipynb`
4. `seaborn/day1.ipynb`
5. `sklearn/day0.ipynb`
6. `sklearn/sklearn_day1.ipynb`
7. `sklearn/day1.ipynb`
8. `sklearn/day2.ipynb`
9. `sklearn/day3.ipynb`
10. `sklearn/day4.ipynb`
11. `sklearn/day5.ipynb`
12. `sklearn/project1_disease_detection_model.ipynb`

## Data Notes

- `data/data_pandas_practice.csv` is included and used by the Pandas beginner notebook.
- `pandas/pandas_intermediate.ipynb`, `sklearn/day0.ipynb`, `sklearn/sklearn_day1.ipynb`, `sklearn/day1.ipynb`, `sklearn/day2.ipynb`, `sklearn/day3.ipynb`, `sklearn/day4.ipynb`, and `sklearn/day5.ipynb` use small in-notebook examples.
- `sklearn/project1_disease_detection_model.ipynb` uses a small in-notebook disease-status dataset for classification practice.
- `pandas/pd_bg_prac1.ipynb` expects a larger `data/emails.csv` file. Add it locally before running that notebook if you have the dataset.
- The repository intentionally keeps large local datasets out of version control.

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

### Seaborn

- Creating count plots with `countplot`
- Visualizing distributions with `histplot`
- Adjusting histogram resolution using bins and customizing axis ticks with `plt.xticks`
- Creating box plots with `boxplot` for data spread and outlier visualization
- Understanding outliers and implementing detection vs. handling strategies in ML workflows
- Using `hue` to split data by categorical subgroups
- Selecting color palettes and ordering axis categories

### Scikit-Learn

- Train/test splitting
- Linear Regression
- Logistic Regression
- Decision Tree Classification
- Disease status classification
- Classification labels
- Prediction workflows
- Prediction probabilities
- End-to-end pipelines with `Pipeline`
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (`R2`) scoring
- Accuracy scoring
- Precision scoring
- Recall scoring
- F1-score (`f1_score`)
- Cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Stratified sampling
- Transformers with `fit`, `transform`, and `fit_transform`
- Column-wise preprocessing with `ColumnTransformer`
- Feature scaling with `StandardScaler`
- Categorical encoding with `OneHotEncoder`
- Missing value filling with `SimpleImputer`
- Overfitting and data leakage prevention
