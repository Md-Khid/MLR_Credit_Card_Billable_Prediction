
# Project Overview

## Introduction
This project aims to analyse a credit facility dataset containing records of customers' demographics, outstanding amounts, repayment history/status, and other relevant variables. The goal is to extract meaningful insights from the data through exploratory data analysis and predictive modelling based on a mock-up credit facility dataset.

## Dataset Information

### Data Variables
The dataset comprises details about customers' credit facilities, including demographic information, outstanding amounts, and repayment history/status.
[Data.csv](https://github.com/Md-Khid/Linear-Regression-Modelling/blob/main/Data.csv)

### Data Dictionary

| Variable  | Description                                       |
|-----------|---------------------------------------------------|
| ID        | Customer unique identifier                         |
| LIMIT     | Customer total limit                              |
| BALANCE   | Customer current credit balance (snapshot in time)|
| INCOME    | Customer current income                           |
| GENDER    | Customer gender (0: Male, 1: Female)             |
| EDUCATION | Customer highest education attained (0: Others, 1: Postgraduate, 2: Tertiary, 3: High School)|
| MARITAL   | Customer marital status(0: Others, 1: Single, 2: Married)|
| AGE       | Customer age in years                             |
| S(n)      | Customer repayment reflected status in nth month (-1; Prompt payment, 0: Minimum sum payment, x = Delayed payment for x month(s))|
| B(n)      | Customer billable amount in nth month             |
| R(n)      | Customer previous repayment amount, paid in nth month|
| RATING    | Customer rating (0: Good, 1: Bad)                |

##### **Note**:
##### n=1 signifies the most recent month, while n=5 signifies the previous 4th month. 
##### If n=1 is the month of May, then n=5 is the month of January.


## Data Preparation

In this phase of data processing, we will refine the dataset for analysis by addressing missing values, handling special characters, and encoding variables. Additionally, we will import all necessary modules and libraries for the project and transform categorical variables into category columns for data visualization purposes.

### Data Pre-processing:

#### Import Data / Libraries/ Modules
```
# Import Libraries and Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Set a standard seaborn color palette
sns.set_palette("colorblind")

# Load data
df = pd.read_csv('Data.csv')

# Drop first column 
df = df.iloc[:, 1:]

# Define mapping dictionaries for categorical columns
mappings = {
    'GENDER': {0: 'Male', 1: 'Female'},
    'EDUCATION': {0: 'Others', 1: 'Postgraduate', 2: 'Tertiary', 3: 'High School'},
    'MARITAL': {0: 'Others', 1: 'Single', 2: 'Married'},
    'RATING': {0: 'Good', 1: 'Bad'},
    'S': {-1: 'Prompt', 0: 'Min Sum', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
}

# Convert columns to categorical and apply mappings
for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
    else:
        for s_col in df.columns[df.columns.str.startswith(col)]:
            df[s_col] = df[s_col].map(mapping)
            
df
```
![Screenshot 2024-03-10 at 12-50-47 Github-Linear Regression Modelling-VIF - Jupyter Notebook](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/1e2930f2-d4d2-4e02-a153-e9ba80dde565)

#### Check Missing Values

```
# Calculate number of missing values 
missing_values = df.isnull().sum()

# Filter the missing_values
columns_with_missing_values = missing_values[missing_values > 0]

# Display columns with missing values
columns_with_missing_values
```
<img width="214" alt="2" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/6c429f57-1fb6-48f1-a972-d052a93920b6">

Based on the output, it appears that the columns "Limit," "Balance," "Education," "Marital," and "Age" contain some missing values. To rectify this issue, we should first analyse the distribution of each column to determine the most suitable method for replacing the missing values, which could involve using the mean, median, or mode.

#### View Data Distribution
```
# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Define columns to plot
columns = ['LIMIT', 'BALANCE', 'AGE', 'MARITAL', 'EDUCATION']

# Iterate over columns and plot them
for i, column in enumerate(columns):
    row = i // 3
    col = i % 3
    sns.histplot(df[column], kde=False, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {column}')
    if column in ['LIMIT', 'BALANCE', 'AGE']:
        axes[row, col].axvline(df[column].median(), color='red', linestyle='--', label='Median')
        axes[row, col].legend()

# Remove gridlines from subplots
for ax in axes.flatten():
    ax.grid(False)
    
# Hide empty subplots
for i in range(len(columns), axes.size):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()
```
![3](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/e929e873-44bc-4c8b-b729-d1022b2e1624)

Given the positively skewed distribution of data in the "Limit," "Balance," and "Age" columns, we can replace the missing values with the median values. For the "Marital" and "Age" columns, we can replace the missing values with the mode. Additionally, upon inspecting the Age distribution, an anomalous age value is observed lying between -1 to 0, as well as 200. To address this anomaly, we will remove such values from the Age column.

#### Replace Missing Values and Remove Data Errors 

```
# Remove rows where 'Age' column has a value of 0, -1, or 100 and above
df = df[(df['AGE'] > 0) & (df['AGE'] < 100)].copy()

# Specify columns and corresponding fill methods
columns_to_fill = {
    'LIMIT': 'median',
    'BALANCE': 'median',
    'AGE': 'median',
    'MARITAL': 'mode',
    'EDUCATION': 'mode'
}

# Fill missing values in columns
for column, method in columns_to_fill.items():
    if method == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    elif method == 'mode':
        df[column].fillna(df[column].mode()[0], inplace=True)

# Display number of columns with missing values
count_missing_values = df.isnull().sum().sum()
count_missing_values
```
<img width="60" alt="4" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/c5a78a38-3298-46d3-8e58-eafc9bf212b0">

#### Removing Special Characters
```
# Define special characters
special_chars = "!@#$%^&"

# Iterate over each column 
for column in df.columns:
    # Iterate over each row in current column
    for index, value in df[column].items():
        # Check if value contains any special characters
        if any(char in special_chars for char in str(value)):
            print(f"Special characters found in column '{column}', row {index}: {value}")
```
<img width="510" alt="5" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/2ff5d9eb-552f-40e7-8a6c-53f43d732871">

```
# Remove special characters ('$' and ',') and spaces from column 'R3'
df['R3'] = df['R3'].str.replace("$", "").str.replace(",", "").str.replace(" ", "")
```
Based on the output, it seems that the R3 column contains special characters. To address this, we replace these characters with an empty string.


#### Encoding of Variables
```
# Identify categorical variables
categorical_variables = df.select_dtypes(include=['object', 'category']).columns

# Check for categorical variables that need encoding
if categorical_variables.empty:
    print("No categorical variables need encoding.")
else:
    print("The following categorical variables need encoding:\n" + "\n".join(categorical_variables))
```
<img width="414" alt="6" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/ab7f71fa-72a8-4ee9-95e5-e8ddbcbaf455">

```
# Convert 'R3' column to the same data type as 'R1', 'R2', 'R4', and 'R5'
df['R3'] = df['R3'].astype(df['R1'].dtype)
```
Based on the output, it appears that the 'R3' column may require encoding. However, according to the [data dictionary](#data-dictionary), 'R3' is expected to be numerical. To address this, we can adjust the data type of the 'R3' column to align with that of the 'R1', 'R2', 'R4', and 'R5' columns. For now, we will refrain from encoding the remaining categorical variables, as we intend to utilise them for generating other frequency tables, bar charts, or graphical methods to comprehend their distribution and relationships with other variables.

## Exploratary Data Analysis 
In this section, we will delve into comprehending the dataset. This encompasses tasks such as examining data distributions, identifying outliers, visualising correlations between variables, and detecting any irregularities or trends, then transforming the insights obtained into valuable information.

#### Descriptive Statisitcs
```
# Create Descriptive Stats table 
Descriptive_Stats = df.describe(include='all').round(2)

# Separate columns into categorical and numerical groups
categorical_columns = Descriptive_Stats.select_dtypes(include=['object']).columns
numeric_columns = Descriptive_Stats.select_dtypes(exclude=['object']).columns

# Order columns (categorical followed by numerical)
ordered_columns = list(categorical_columns) + list(numeric_columns)
Descriptive_Stats = Descriptive_Stats.reindex(ordered_columns, axis=1)

# Transpose Descriptive Stats table 
Descriptive_Stats = Descriptive_Stats.transpose()

Descriptive_Stats
```
![7](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/a06672c1-24c8-4f24-805b-3913474e7df1)

Based on the statistical table, the credit card bank typically offers a uniform credit limit based on customers' income. However, a significant portion of its customers struggle to meet their credit card bill payments once they have utilised approximately 97% of their credit limit. They can only afford to make a small payment towards their monthly bills, around 10%. This serves as a clear signal for the credit card bank to issue reminder notices and make phone calls or impose late fees and additional interest charges on unpaid balances, leading to an increase in the outstanding amount over time.

## Insight Articulation

#### Scaling Numerical Features
```
# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
df
```
![8](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/1b10cd65-f4f9-497e-ba94-0595147f0a67)

Scaling numerical variables in a dataset helps interpret relationships between variables, especially in scatterplots and correlation analysis. It helps to ensure they are on a similar scale.

#### Heatmap
```
def plot_corr_and_print_highly_correlated(df):
    # Create a correlation matrix
    corr_matrix = df.select_dtypes(include='number').corr()

    # Plot heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5)
    plt.show()

    # Create a mask for correlations greater than 0.7
    high_corr = corr_matrix[corr_matrix > 0.7]

    # Get pairs of highly correlated variables
    high_corr_pairs = [(i, j) for i in high_corr.columns for j in high_corr.index if (high_corr[i][j] > 0.7) & (i != j)]

    # Sort each pair and remove duplicates
    high_corr_pairs = list(set([tuple(sorted(pair)) for pair in high_corr_pairs]))

    # Sort the pairs alphabetically
    high_corr_pairs = sorted(high_corr_pairs)

    print("Pairs of variables with correlation greater than 0.7:")
    for pair in high_corr_pairs:
        print(pair)
        
# Call the function
plot_corr_and_print_highly_correlated(df)
```
![9 1](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/61434dc1-e690-4aee-a0e2-bd0896d082d4)
![9 2](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/0bfaf7c1-8cd3-4152-9524-dbffb58f62b9)

Based on the correlation heatmap, it is clear that there is a strong correlation between variables such as 'INCOME' and 'LIMIT', as well as 'B(n)' and 'BALANCE'.


#### Density Plot
```
# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define variables
plot_data = [('EDUCATION', 'Education Level', ['Others', 'Postgraduate', 'Tertiary', 'High School']),
             ('GENDER', 'Gender', ['Male', 'Female']),
             ('MARITAL', 'Marital Status', ['Others', 'Single', 'Married'])]

# Plot each density plot in a subplot
for ax, (variable, title, labels) in zip(axes, plot_data):
    sns.kdeplot(data=df, x='LIMIT', hue=variable, fill=True, ax=ax)
    ax.set(xlabel='LIMIT', ylabel='Density')
    ax.legend(title=variable, labels=labels)

plt.tight_layout()
plt.show()
```
![10](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/baf1814d-0dda-416e-85c8-d77912ea21d0)

Based on the density plots, it is clear that the credit card bank tends to offer higher credit limits to customers who are 1. Male, 2. Married, and 3. Have a postgraduate education.


#### Boxplot
```
# Define variables for iteration
plot_data = [
    {'column_x': 'INCOME', 'column_y': 'EDUCATION', 'data': 'INCOME', 'order': ['Others', 'Postgraduate', 'Tertiary', 'High School'], 'title':'.'},
    {'column_x': 'BALANCE', 'column_y': 'EDUCATION', 'data': 'BALANCE', 'order': ['Others', 'Postgraduate', 'Tertiary', 'High School'], 'title':'.'}
]

# Create figure with subplots
fig, axes = plt.subplots(1, len(plot_data), figsize=(15, 6))

# Iterate plot_data and create subplots
for i, plot_info in enumerate(plot_data):
    # Calculate the mean for the current group and sort the order accordingly
    mean_values = df.groupby(plot_info['column_y'])[plot_info['column_x']].mean().sort_values(ascending=False).index
    sns.boxplot(ax=axes[i], x=plot_info['column_x'], y=plot_info['column_y'], data=df, order=mean_values)
    axes[i].set_xlabel(plot_info['data'].capitalize())
    axes[i].set_ylabel('.')
    axes[i].set_title(plot_info['title'])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)  # Rotate x-axis labels
    axes[i].grid(False)

plt.tight_layout()
plt.show()
```
![11](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/4a4bd9ad-9bed-4db3-becb-53a3d1246f55)

Based on the Income box plot, it is clear that individuals with high incomes mainly belong to the Tertiary Education group. However, this group demonstrates lower credit balances in comparison to the Postgraduate and High School groups.

#### Scatterplot
```
# Define education levels and corresponding colors
unique_education = df['EDUCATION'].unique()
colors = plt.cm.tab10.colors[:len(unique_education)]

# Define variables to plot
variables = ['B1', 'B2', 'B3', 'B4', 'B5']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Plotting
for ax, variable in zip(axes, variables):
    for color, education in zip(colors, unique_education):
        data = df[df['EDUCATION'] == education]
        ax.scatter(data[variable], data['LIMIT'], color=color, label=education, alpha=0.6)
    ax.set_xlabel(variable)
    ax.set_ylabel('LIMIT')
    ax.legend()

# Hide empty subplots
for ax in axes[len(variables):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
```
![12](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/b4335cd8-984c-45ae-96ca-123c737b08b8)

Spell check: The bill amount owned by customers kept increasing as the month increasses. And most of the high sizeabbe bill amoout is concerntrated on the Postgraduates and Tertiary education group as they mosntly obtained a higer credit limit

```
# Define variables to plot
variables = ['S1', 'S2', 'S3', 'S4', 'S5']

# Define the order of categories for each variable
variable_order = {
    'S1': ['Prompt', 'Min Sum', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight'],
    'S2': ['Prompt', 'Min Sum', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight'],
    'S3': ['Prompt', 'Min Sum', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight'],
    'S4': ['Prompt', 'Min Sum', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight'],
    'S5': ['Prompt', 'Min Sum', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']
}

# Iterate through each hue
for hue in ['GENDER', 'EDUCATION', 'MARITAL']:
    # Get the colorblind palette
    hue_colors = sns.color_palette("colorblind", len(df[hue].unique()))

    # Create subplots for the current hue
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Plotting
    for ax, variable in zip(axes, variables):
        # Ensure df contains the current hue column
        if hue in df.columns:
            sns.stripplot(x=df[variable], y=df['B1'], hue=df[hue], ax=ax, palette=hue_colors, order=variable_order[variable], jitter=True, dodge=True)
            ax.set_xlabel(variable if variable not in variables[3:] else '')
            ax.set_ylabel('B1')
            ax.grid(True)

            # Create a legend for the current hue
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in hue_colors]
            ax.legend(handles, df[hue].unique(), loc='upper right')
        else:
            print(f"Warning: DataFrame does not contain column '{hue.upper()}'")

    # Hide empty subplots
    for ax in axes[len(variables):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


```
![13 1](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/e2dcf531-99a6-4ef9-a17d-4abc34f5a948)
![13 2](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/618be632-4091-4c49-b7c4-bfe210ab4e05)
![13 3](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/63490a5c-de34-4503-a457-70381a6e17fe)

```
# Define variables 
variables = ['BALANCE', 'AGE']

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plotting
for i, var in enumerate(variables):
    ax = axes[i]
    sns.scatterplot(data=df, x=var, y='INCOME', hue='RATING', palette={'Good': sns.color_palette()[0], 'Bad': sns.color_palette()[3]}, alpha=0.6, ax=ax)
    ax.set_xlabel(var)
    ax.set_ylabel('INCOME')
    ax.legend(title='RATING')

plt.tight_layout()
plt.show()
```
![14](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/3d56b2c5-a95e-4940-8352-b259c6c5b896)


#### Barplot
```
# Calculate number of occurrences of each category in 'RATING' column
rating_counts = df['RATING'].value_counts()

# Display counts 
print(f"Number of GOOD ratings: {rating_counts['Good']}")
print(f"Number of BAD ratings: {rating_counts['Bad']}")
```
<img width="315" alt="15" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/fbedfff7-e497-47ef-8769-16a1f470cec0">


```
# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Calculate frequencies for each rating category for GENDER, MARITAL, and EDUCATION
for ax, category in zip(axes, ['GENDER', 'MARITAL', 'EDUCATION']):
    # Count the occurrences of each category in the RATING column
    education_counts = df[df['RATING'].isin(['Good', 'Bad'])].groupby([category, 'RATING']).size().unstack(fill_value=0)

    # Plotting with sorted frequencies
    education_counts.sort_values(by='Good', ascending=False).plot(kind='bar', stacked=True, ax=ax, color={'Good': sns.color_palette()[0], 'Bad': sns.color_palette()[3]}, alpha=0.6)
    ax.set_xlabel(category)
    ax.set_ylabel('Frequency')
    ax.legend(title='RATING')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate x-axis labels

plt.tight_layout()
plt.show()
```
![16](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/ae756b3b-b921-4c81-8b68-0e740a715b0f)

Based on the barplots, it is clear that the credit card bank tends to provide Good ratings to customers who are 1. Female, 2. Married, and 3. Have a Tertiary education.


## Linear Regression Modelling
Build a linear regression model to predict the variable B1, explaining the approach taken and any necessary data pre-processing.

#### Reverting
```
# Revert to original scale
df[numeric_columns] = scaler.inverse_transform(df[numeric_columns])
df
```
![17](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/fc1ab4c8-4ae0-49e4-a677-ae5932807504)


#### Dummy Variables
```
# Dummy columns
df = pd.get_dummies(df, columns=['RATING','GENDER','EDUCATION','MARITAL','S1','S2','S3','S4','S5'])

# Convert boolean columns to integers (1 and 0)
df[df.select_dtypes(include=bool).columns] = df.select_dtypes(include=bool).astype(int)
df
```
<img width="492" alt="18" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/e60f9263-472d-466f-9fb7-a14b6f221099">

#### Remove Outliers
```
# Identify numerical columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# IQR function to identify outliers
def identify_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers

# Apply the function
outliers = df[numeric_columns].apply(identify_outliers_iqr)

# Keep a copy of original DataFrame
df_with_outliers = df.copy()

# Remove outliers
df = df[~outliers.any(axis=1)]

# Get the rows of outliers that have been removed
removed_outliers = df_with_outliers[outliers.any(axis=1)]

# Print the number of rows of outliers that have been removed
print(f"Number of rows of outliers removed: {len(removed_outliers)}")
```
<img width="351" alt="19" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/bea70915-381d-40a0-bfa5-022c6d4684cf">

```
# Get list of numerical predictor variables
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
predictors = df[numeric_columns].drop(columns='B1')

# Add a constant to predictor variables
predictors = sm.add_constant(predictors)

# Calculate VIF for each predictor variable
vif = pd.DataFrame()
vif["Variable"] = predictors.columns
vif["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

# Remove variable with the highest VIF
while vif['VIF'].max() > 5:
    # Identify variable with the highest VIF
    max_vif_variable = vif.loc[vif['VIF'].idxmax(), 'Variable']
    
    # If variable with the highest VIF is 'const', skip it
    if max_vif_variable == 'const':
        vif = vif.drop(vif['VIF'].idxmax())
        continue
    
    # Drop variable with the highest VIF
    predictors = predictors.drop(columns=max_vif_variable)
    
    # Recalculate VIF
    vif = pd.DataFrame()
    vif["Variable"] = predictors.columns
    vif["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

# Print VIF values
vif
```
![20](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/0de6b1a4-40df-46f8-ad28-0f0048606340)

### Modelling
```

# Define independent variables (X)
X = sm.add_constant(df.drop(columns=['B1']))

# Define dependent variable (Y)
y = df['B1']

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the OLS (Ordinary Least Squares) model on training data
model = sm.OLS(y_train, X_train).fit()

# Iterate through the p-values and remove variables with p-value > 0.05, excluding the constant term
while model.pvalues.drop('const').max() > 0.05:
    # Find the variable with the highest p-value (excluding the constant term)
    max_pvalue_index = model.pvalues.drop('const').idxmax()
    # Remove the variable from X_train and X_test
    X_train = X_train.drop(columns=[max_pvalue_index])
    X_test = X_test.drop(columns=[max_pvalue_index])
    # Fit the model again with the updated X_train
    model = sm.OLS(y_train, X_train).fit()

# Make predictions on training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate Root Mean Squared Error (RMSE) for training and testing sets
rmse_train = np.sqrt(np.mean((y_train - y_train_pred)**2))
rmse_test = np.sqrt(np.mean((y_test - y_test_pred)**2))

# Print the RMSE 
print("Train RMSE:", round(rmse_train, 2))
print("Test RMSE:", round(rmse_train, 2))

# Print model summary
model.summary()
```
![21 2](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/4a93c81f-4443-4464-a531-a9fd7b34c073)
![21 1](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/cfb5fad1-b314-48cd-b27b-a4d0e701c502)

Given the [descriptive statistics of B1](#descriptive-statistics), the mean is around 49985.76 and the standard deviation is approximately 71927.41. The RMSE of 7638.74 is much smaller than both the mean and the standard deviation, which suggests that the model’s predictions are relatively close to the actual values.

## Evaluate Model Performance

```
# Define columns for dummy encoding
dummy_cols = ['RATING','GENDER','EDUCATION','MARITAL','S1','S2','S3','S4','S5']

# Load and preprocess test data
df_test = (pd.read_csv('Test Data.csv')
             .iloc[:, 1:]
             .pipe(pd.get_dummies, columns=dummy_cols)
             .assign(**{col: lambda df: df[col].astype(int) for col in df.select_dtypes(include=bool).columns})
             .pipe(lambda df: (sm.add_constant(df.drop(columns=['B1'])), df['B1']))
          )

# Align the test data with training data
X_test, y_test = df_test
X_test = X_test.reindex(columns = X_train.columns, fill_value = 0)

# Make predictions 
y_test_pred = model.predict(X_test).round(0)

# Compare actual and predicted B1 values
df_compare = pd.DataFrame({'Actual B1': y_test, 'Predicted B1': y_test_pred})
df_compare
```
![22](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/c479e0d5-0cba-4d72-ba62-352f92e4679d)





