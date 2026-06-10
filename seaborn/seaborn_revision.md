# Count Plot (countplot)

## What It Is
A categorical bar plot that counts the number of observations in each category of a discrete variable.

## Why It Is Used
ML engineers use it during exploratory data analysis (EDA) to verify target class balance. Class imbalance (e.g. 95% non-disease cases vs 5% disease cases) requires standard adjustment techniques like stratification during splitting or customized scoring metrics (e.g. Recall instead of Accuracy).

## Common Syntax
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Vertical count plot of class labels
sns.countplot(data=df, x="survived")
plt.show()

# Horizontal count plot by swapping coordinates
sns.countplot(data=df, y="embarked")
plt.show()
```

## Key Concepts
* **What question it answers**: How are observations distributed across categorical bins?
* **Class Imbalance Identification**: Visually flags whether one category dominates the dataset.

## Important Notes
* Set `x` for vertical bars and `y` for horizontal bars. Swapping to `y` is useful when category names are long to prevent axis text overlap.

## Interview / Revision Points
* How does a count plot help ML modeling? It visualizes class distribution, indicating if class imbalance handling (e.g. SMOTE, class weighting) is necessary before modeling.

---

# Histogram (histplot)

## What It Is
A plot that divides continuous numerical columns into intervals (bins) and displays the count or density of values falling in each bin as vertical bars.

## Why It Is Used
ML engineers use histograms to check the distribution shape of continuous continuous values (e.g., age, fare). Knowing whether a feature has a normal distribution or is highly skewed tells us whether we need to apply scaling transformations (like standard scaling or log scaling) before fitting models.

## Common Syntax
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Basic histogram
sns.histplot(data=df, x="age")
plt.show()

# High-resolution histogram with custom bins
sns.histplot(data=df, x="age", bins=100)
plt.show()
```

## Key Concepts
* **What question it answers**: What is the shape of the continuous feature's distribution?
* **Distribution Properties**: Highlights statistical features like normality, variance, data gaps, or skewness.
* **Bin Count Sensitivity**: Higher bin counts reveal more local structure; lower bin counts provide general visual trends.

## Important Notes
* Changing the number of bins (`bins=100`) modifies visual group resolution, but does not alter the actual data values or the total range represented on the axes.

## Interview / Revision Points
* Why inspect histograms before scaling? Distance-based algorithms perform better when features are normally distributed. Highly skewed variables may require log transformation first.

---

# Box Plot (boxplot)

## What It Is
A visualization that summarizes numerical data distribution using 5 statistics: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum. It highlights values outside this range as outlier points.

## Why It Is Used
ML engineers use box plots to spot outliers and anomalies. If standard box plots identify values outside reasonable limits (e.g. Passenger Age = 800), they signify data errors that must be handled. Realistic outliers (e.g., Age = 80) are kept to preserve model robustness.

## Common Syntax
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create box plot for outlier detection
sns.boxplot(data=df, x="age")
plt.show()
```

## Key Concepts
* **What question it answers**: What is the spread of numerical data and are there outliers present?
* **Five-Number Summary**:
  * Box limits = Q1 (25th percentile) and Q3 (75th percentile).
  * Center line = Median (50th percentile).
  * Whiskers = Extend up to 1.5 * IQR (Interquartile Range) from the box edges.
  * Points beyond whiskers = Outliers.

## Important Notes
* Boxplots are outlier detection tools, not outlier removal tools. Keep outliers if they are realistic observations; remove them only if they represent errors.

## Interview / Revision Points
* How are outliers determined in a boxplot? Data points that lie beyond 1.5 times the Interquartile Range ($IQR = Q3 - Q1$) from the edges of the box.

---

# Scatter Plot (scatterplot)

## What It Is
A graph that displays individual values of two numerical features as coordinate points on a Cartesian grid.

## Why It Is Used
ML engineers use scatter plots to evaluate relationships between numerical columns. Linear patterns suggest a linear regression model will perform well, whereas clusters or non-linear curves indicate the need for more complex tree-based or polynomial models.

## Common Syntax
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot of study hours vs exam score
sns.scatterplot(data=study_df, x="study_hours", y="score")
plt.show()
```

## Key Concepts
* **What question it answers**: What is the visual relationship or correlation between two numerical features?
* **Correlation Identification**: Direct upward slopes indicate positive correlation, downward slopes indicate negative correlation.
* **Cluster Identification**: Points grouping in regions suggest hidden subsets.

## Important Notes
* Overplotting (too many points overlapping) can hide density. Adjust alpha transparency if necessary when working with large datasets.

## Interview / Revision Points
* How does a scatter plot guide model selection? A linear trend indicates simple linear models are appropriate. Curving or clustered trends require non-linear models.

---

# Regression Plot (regplot)

## What It Is
A plot that combines a scatter plot with a calculated linear regression line of best fit and a shaded 95% confidence interval band.

## Why It Is Used
ML engineers use it to quickly evaluate the validity of linear assumptions between variables. If scatter points lie very far from the regression line or have a non-linear shape, it signals that standard linear models may suffer from underfitting.

## Common Syntax
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Regplot with best-fit line
sns.regplot(data=study_df, x="study_hours", y="score")
plt.show()
```

## Key Concepts
* **What question it answers**: Is there a strong linear relationship between these two variables?
* **Residual Visualization**: Shows the distance of each data point from the predicted regression line.
* **Confidence Interval Band**: Shaded area representing the uncertainty margin around the regression estimate.

## Important Notes
* Unlike `lmplot`, `regplot` is a axes-level function and does not support facet grids directly. It is best used for quick single-axes trend evaluation.

## Interview / Revision Points
* What does the shaded area in a `regplot` represent? The 95% confidence interval of the regression line estimate.

---

# Heatmap (heatmap)

## What It Is
A 2D visualization where numerical matrix values (typically a correlation matrix) are represented as colors, often annotated with the exact values.

## Why It Is Used
ML engineers use correlation heatmaps to identify multicollinearity. If two features are highly correlated (correlation coefficients close to +1 or -1), they contain redundant information. Dropping one of the redundant features simplifies models, avoids unstable weights in linear models, and reduces training times.

## Common Syntax
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr = df.corr(numeric_only=True)

# Visualize matrix with color annotation
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
```

## Key Concepts
* **What question it answers**: What are the correlation coefficients across all numerical columns?
* **Multicollinearity Flag**: Dark red or dark blue cells indicate highly correlated variables.
* **Color Palettes**: Diverging colormaps (like `coolwarm`) clearly separate positive correlations (red) from negative correlations (blue).

## Important Notes
* Heatmaps expect a 2D matrix (like `.corr()`), not a raw DataFrame. Pass `annot=True` to print the exact correlation numbers inside the grid cells.

## Interview / Revision Points
* What is multicollinearity and how does a heatmap help? Multicollinearity occurs when features are highly correlated. Heatmaps locate these overlaps visually, allowing engineers to prune redundant features.

---

# Hue Parameter (hue)

## What It Is
An aesthetic parameter that groups plotted data categories using color encoding based on a separate categorical column.

## Why It Is Used
ML engineers use `hue` to compare distributions, relationships, or counts across groups (e.g. looking at age distributions split by survival status to see if passenger age impacted survival rate).

## Common Syntax
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Color code histogram by survival category
sns.histplot(data=df, x="age", hue="survived")
plt.show()

# Color code categorical count plots
sns.countplot(data=df, x="pclass", hue="survived")
plt.show()
```

## Key Concepts
* **What question it answers**: How does a third categorical dimension impact the relationship of two other variables?
* **Subgroup Analysis**: Allows side-by-side categorical comparisons inside distribution plots.

## Important Notes
* Using `hue` on columns with very high cardinality (too many unique string values) generates a chaotic plot with dozens of colors, which is unreadable. Stick to low-cardinality keys.

## Interview / Revision Points
* How does `hue` enhance visual plots? It introduces a categorical split, coloring data points to identify subgroup trends in distributions or counts.
