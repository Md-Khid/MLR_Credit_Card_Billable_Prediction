
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
##### If n=1 is the month of May 2022, then n=5 is the month of January 2024.


## Data Preparation

In this part of data processing, we will prepare the dataset for analysis by handling missing values, special characters and encoding of variables.


### Data Pre-processing:

#### Import Data
```
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Covert columns to categorical variables
columns_to_convert = ['RATING', 'GENDER', 'EDUCATION', 'MARITAL', 'S1', 'S2', 'S3', 'S4', 'S5']

# Read data file and drop the 1st column
df = pd.read_csv('Data.csv', usecols=lambda column: column != 'SERIAL NUMBER', dtype={col: 'category' for col in columns_to_convert})

# Display data table
df
```
<img width="748" alt="1" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/43cf29ab-7f6a-4e69-83cf-2540420cf8b6">

#### Check Missing Values

```
# Calculate number of missing values in each column 
missing_values = df.isnull().sum()

# Filter the missing_values i.e. only columns with missing values
columns_with_missing_values = missing_values[missing_values > 0]

# Display columns with missing values
columns_with_missing_values
```
<img width="198" alt="2" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/bef1ceb8-640c-46f3-a87e-568210ca0797">


Based on the output, the columns "Limit," "Balance," "Education," "Marital," and "Age" contain some missing values. To address this, we need to understand the distribution of each column so that we can appropriately replace the missing values, such as using the mean, median, or mode.

#### View Data Distribution
```
# Set the plot style
sns.set_style("white")

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Define columns to plot
columns = ['LIMIT', 'BALANCE', 'AGE', 'MARITAL', 'EDUCATION']

# Iterate over the columns and plot them
for i, column in enumerate(columns):
    row = i // 3
    col = i % 3
    sns.histplot(df[column], kde=False, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {column}')
    if column in ['LIMIT', 'BALANCE', 'AGE']:
        axes[row, col].axvline(df[column].median(), color='red', linestyle='--', label='Median')
        axes[row, col].legend()

# Remove gridlines from all subplots
for ax in axes.flatten():
    ax.grid(False)

plt.tight_layout()
plt.show()
```
<img width="910" alt="2" src="https://github.com/Md-Khid/Linear-Regression-Modelling/assets/160820522/ed483cbc-7fe2-4798-b975-39dd52083ad0">

Given the positively skewed distribution of data in the "Limit," "Balance," and "Age" columns, we can replace the missing values with the median values. For the "Marital" and "Age" columns, we can replace the missing values with the mode. Additionally, upon inspecting the Age distribution, an anomalous age value is observed lying between -1 to 0, as well as 200. To address this anomaly, we will remove such values from the Age column, as they may represent system or human entry errors.

#### Replace Missing Values and Remove Data Errors 

```
# Remove rows where 'Age' column has a value of 0, -1, or 100 and above
df = df[(df['AGE'] > 0) & (df['AGE'] < 100)]

# Specify columns and their corresponding fill methods
columns_to_fill = {
    'LIMIT': 'median',
    'BALANCE': 'median',
    'MARITAL': 'mode',
    'EDUCATION': 'mode'
}

# Fill missing values in specified columns
for column, method in columns_to_fill.items():
    if method == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    elif method == 'mode':
        df[column].fillna(df[column].mode()[0], inplace=True)

# Check for missing values in columns 
missing_values = df.isnull().any()

# Calculate the total count of columns with missing values 
count_missing_values = missing_values.sum()

# Display number of columns with missing values
count_missing_values
```

#### Removing Special Characters
```
# Iterate over each column in the DataFrame
for column in df.columns:
    # Iterate over each row in the current column
    for index, value in df[column].items():
        # Check if the value contains any special characters
        if any(char in "!@#$%^&" for char in str(value)):
            print(f"Special characters found in column '{column}', row {index}: {value}")
```
<img width="400" alt="4" src="https://github.com/Md-Khid/Linear-Regression-Modelling/assets/160820522/e7266526-7830-4492-b36d-31d818e8f01e">

```
# Remove special characters ('$' and ',') from column 'R3'
df['R3'] = df['R3'].str.replace('[\$,]', '', regex=True)
```
Based on the output, it seems that the R3 column contains special characters. To address this, we replace these characters with an empty string.

#### Encoding of Variables
```
# Identify categorical variables
categorical_variables = df.select_dtypes(include=['object', 'category'])

# Check for categorical variables that need encoding
if not categorical_variables.empty:
    print("The following categorical variables need encoding:")
    for column in categorical_variables.columns:
        print(column)
else:
    print("No categorical variables need encoding.")
```
<img width="318" alt="4" src="https://github.com/Md-Khid/Linear-Regression-Modelling/assets/160820522/0514b001-4715-4c13-bf05-31612441be09">

```
# Convert 'R3' column to the same data type as 'R1', 'R2', 'R4', and 'R5'
df['R3'] = df['R3'].astype(df['R1'].dtype)
```
Based on the output, it seems that the 'R3' column needs encoding. However, based on the [data dictionary](#data-dictionary), 'R3' is expected to be numerical, similar to 'R1', 'R2', 'R4', and 'R5'. To resolve this, we can change the data type of the 'R3' column to match that of the 'R1', 'R2', 'R4', and 'R5' columns.  For now, we will refrain from encoding the remaining categorical variables as they are typically analysed using frequency tables, bar charts, or other graphical methods to understand their distribution and relationships with other variables.

## Exploratary Data Analysis 
In this section, we will dive into understanding the dataset. This involves tasks like exploring data distributions, spotting outliers, visualising relationships between variables, and identifying any anomalies. 

#### Descriptive Statisitcs
```
# Select numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Create the statistical description table
statistical_description = df[numeric_columns].describe()

# Round the statistical description table to 2 decimal places and transpose it
statistical_description = np.round(statistical_description, 2).T

# Display statistical table
statistical_description
```


#### Scaling Numerical Features
```
# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Display data table
df
```

Scaling numerical variables in a dataset helps interpret relationships between variables, especially in scatterplots and correlation analysis. It helps to ensure they are on a similar scale.

## Insight Articulation
```
# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define the variables for plotting
plot_data = [('EDUCATION', 'Education Level', ['Others', 'Postgraduate', 'Tertiary', 'High School']),
             ('GENDER', 'Gender', ['Male', 'Female']),
             ('MARITAL', 'Marital Status', ['Others', 'Single', 'Married'])]

# Plot each density plot in a subplot
for idx, (variable, title, labels) in enumerate(plot_data):
    sns.kdeplot(data=df, x='LIMIT', hue=variable, fill=True, ax=axes[idx])
    axes[idx].set_title(f'Density Plot of LIMIT by {title}')
    axes[idx].set_xlabel('LIMIT')
    axes[idx].set_ylabel('Density')
    axes[idx].legend(title=variable, labels=labels)

plt.tight_layout()
plt.show()
```
<img width="895" alt="7" src="https://github.com/Md-Khid/Linear-Regression-Modelling/assets/160820522/1329de34-22b6-4cf0-8aa3-0a475604dcb7">

## Linear Regression Modelling
Build a linear regression model to predict the variable B1, explaining the approach taken and any necessary data pre-processing.

## Evaluate Model Performance





- Excel - Data Cleaning
  - [Download here](https://microsoft.com)
- SQL Server - Data Analysis
- PowerBI - Creating reports





