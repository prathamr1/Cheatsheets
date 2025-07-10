# Pandas Cheatsheet - Data Manipulation & Cleaning

## Import and Setup
```python
import pandas as pd
import numpy as np
```

## Creating DataFrames
```python
# From dictionary
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# From lists
df = pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=['A', 'B'])

# From CSV
df = pd.read_csv('file.csv')

# From Excel
df = pd.read_excel('file.xlsx')

# From JSON
df = pd.read_json('file.json')
```

## Basic Information
```python
df.head()          # First 5 rows
df.tail()          # Last 5 rows
df.shape           # (rows, columns)
df.info()          # Data types and memory usage
df.describe()      # Statistical summary
df.dtypes          # Data types of each column
df.columns         # Column names
df.index           # Row indices
df.memory_usage()  # Memory usage per column
```

## Selecting Data
```python
# Single column
df['column_name']
df.column_name

# Multiple columns
df[['col1', 'col2']]

# Rows by index
df.iloc[0]         # First row
df.iloc[0:3]       # First 3 rows
df.iloc[-1]        # Last row

# Rows by label
df.loc['row_name']
df.loc['row1':'row3']

# Conditional selection
df[df['column'] > 5]
df[df['column'].isin(['value1', 'value2'])]
df[(df['col1'] > 5) & (df['col2'] < 10)]
```

## Filtering and Querying
```python
# Query method
df.query('column > 5')
df.query('col1 > 5 and col2 < 10')

# Boolean indexing
df[df['column'].str.contains('pattern')]
df[df['column'].str.startswith('prefix')]
df[df['column'].str.endswith('suffix')]

# Between values
df[df['column'].between(1, 10)]

# Not null values
df[df['column'].notna()]
df[df['column'].notnull()]
```

## Data Cleaning
```python
# Handle missing values
df.isnull()                    # Check for null values
df.notnull()                   # Check for non-null values
df.isnull().sum()              # Count nulls per column
df.dropna()                    # Drop rows with any null
df.dropna(subset=['col1'])     # Drop rows with null in specific column
df.fillna(value)               # Fill nulls with value
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill
df.fillna(df.mean())           # Fill with mean

# Remove duplicates
df.duplicated()                # Check for duplicates
df.drop_duplicates()           # Remove duplicates
df.drop_duplicates(subset=['col1'])  # Remove duplicates based on column

# Replace values
df.replace('old_value', 'new_value')
df.replace({'col1': {'old': 'new'}})
```

## Data Transformation
```python
# Apply functions
df['new_col'] = df['col'].apply(lambda x: x * 2)
df.apply(lambda x: x.max() - x.min())  # Apply to each column
df.applymap(lambda x: x.upper())       # Apply to each element

# Map values
df['col'].map({'old': 'new', 'value': 'mapped'})

# String operations
df['col'].str.upper()
df['col'].str.lower()
df['col'].str.strip()
df['col'].str.replace('old', 'new')
df['col'].str.split(',')
df['col'].str.len()
```

## Sorting and Ranking
```python
# Sort by values
df.sort_values('column')
df.sort_values(['col1', 'col2'], ascending=[True, False])

# Sort by index
df.sort_index()

# Ranking
df['col'].rank()
df['col'].rank(method='dense')
```

## Grouping and Aggregation
```python
# Group by
df.groupby('column').sum()
df.groupby('column').mean()
df.groupby('column').count()
df.groupby('column').agg(['sum', 'mean', 'count'])
df.groupby(['col1', 'col2']).sum()

# Multiple aggregations
df.groupby('col').agg({
    'col1': 'sum',
    'col2': 'mean',
    'col3': ['min', 'max']
})

# Pivot tables
df.pivot_table(values='value_col', index='row_col', columns='col_col')
df.pivot_table(values='value_col', index='row_col', columns='col_col', aggfunc='sum')
```

## Merging and Joining
```python
# Concatenate
pd.concat([df1, df2])              # Vertical
pd.concat([df1, df2], axis=1)      # Horizontal

# Merge (SQL-like joins)
pd.merge(df1, df2, on='key')       # Inner join
pd.merge(df1, df2, on='key', how='left')    # Left join
pd.merge(df1, df2, on='key', how='right')   # Right join
pd.merge(df1, df2, on='key', how='outer')   # Full outer join

# Join (index-based)
df1.join(df2)
df1.join(df2, how='left')
```

## Reshaping Data
```python
# Pivot
df.pivot(index='row_col', columns='col_col', values='value_col')

# Melt (unpivot)
df.melt(id_vars=['col1'], value_vars=['col2', 'col3'])

# Stack/Unstack
df.stack()    # Columns to rows
df.unstack()  # Rows to columns

# Transpose
df.T
```

## Statistical Operations
```python
# Basic statistics
df.mean()
df.median()
df.mode()
df.std()
df.var()
df.min()
df.max()
df.sum()
df.count()
df.quantile(0.25)

# Correlation
df.corr()
df['col1'].corr(df['col2'])

# Cross-tabulation
pd.crosstab(df['col1'], df['col2'])
```

## Date and Time Operations
```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date_string'])

# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.dayofweek

# Date arithmetic
df['date'] + pd.Timedelta(days=30)
df['date'] - pd.DateOffset(months=1)

# Resample time series
df.set_index('date').resample('D').mean()  # Daily average
df.set_index('date').resample('M').sum()   # Monthly sum
```

## Adding and Modifying Columns
```python
# Add new column
df['new_col'] = df['col1'] + df['col2']
df['new_col'] = 'constant_value'

# Conditional column
df['new_col'] = np.where(df['col1'] > 5, 'high', 'low')
df['new_col'] = df['col1'].apply(lambda x: 'high' if x > 5 else 'low')

# Multiple conditions
conditions = [df['col1'] > 10, df['col1'] > 5]
choices = ['very_high', 'high']
df['new_col'] = np.select(conditions, choices, default='low')

# Rename columns
df.rename(columns={'old_name': 'new_name'})
df.columns = ['new_col1', 'new_col2']
```

## Dropping Data
```python
# Drop columns
df.drop('column_name', axis=1)
df.drop(['col1', 'col2'], axis=1)

# Drop rows
df.drop(0)  # Drop row with index 0
df.drop([0, 1, 2])  # Drop multiple rows

# Drop by condition
df.drop(df[df['col'] < 5].index)
```

## Index Operations
```python
# Set index
df.set_index('column')
df.set_index(['col1', 'col2'])  # MultiIndex

# Reset index
df.reset_index()
df.reset_index(drop=True)  # Don't keep old index as column

# Reindex
df.reindex(['new_index1', 'new_index2'])
```

## Copying and Sampling
```python
# Copy DataFrame
df_copy = df.copy()
df_copy = df.copy(deep=True)

# Sample data
df.sample(n=5)        # Random 5 rows
df.sample(frac=0.1)   # Random 10% of rows
df.sample(n=5, random_state=42)  # Reproducible sampling
```

## Export Data
```python
# To CSV
df.to_csv('output.csv', index=False)

# To Excel
df.to_excel('output.xlsx', index=False)

# To JSON
df.to_json('output.json')

# To HTML
df.to_html('output.html')

# To SQL
df.to_sql('table_name', connection, if_exists='replace')
```

## Performance Tips
```python
# Use categorical data for repeated strings
df['col'] = df['col'].astype('category')

# Use vectorized operations instead of loops
df['new_col'] = df['col1'] * df['col2']  # Good
# df['new_col'] = df.apply(lambda x: x['col1'] * x['col2'], axis=1)  # Slower

# Use loc/iloc for large datasets
df.loc[df['col'] > 5, 'new_col'] = 'value'  # Good
# df[df['col'] > 5]['new_col'] = 'value'  # Creates copy

# Chain operations
result = (df
          .dropna()
          .groupby('col1')
          .agg({'col2': 'mean'})
          .reset_index())
```

## Common Patterns
```python
# Find and replace outliers
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['col'] < (Q1 - 1.5 * IQR)) | (df['col'] > (Q3 + 1.5 * IQR)))]

# Create bins
df['binned'] = pd.cut(df['col'], bins=5, labels=['low', 'med_low', 'med', 'med_high', 'high'])

# Normalize data
df['normalized'] = (df['col'] - df['col'].min()) / (df['col'].max() - df['col'].min())

# One-hot encoding
pd.get_dummies(df['categorical_col'])
```