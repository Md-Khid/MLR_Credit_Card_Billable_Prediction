
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
| EDUCATION | Customer highest education attained               |
| MARITAL   | Customer marital status                           |
| AGE       | Customer age in years                             |
| S(n)      | Customer repayment reflected status in nth month  |
| B(n)      | Customer billable amount in nth month             |
| R(n)      | Customer previous repayment amount, paid in nth month|
| RATING    | Customer rating (0: Good, 1: Bad)                |

##### **Note**:
##### n=1 signifies the most recent month, while n=5 signifies the previous 4th month. 
##### If n=1 is the month of May 2022, then n=5 is the month of January 2024.


## Data Preparation

In this part of data pipeline, we will prepare the dataset for analysis by handling missing values, encoding categorical variables, and scaling numerical features.


1.1 **Data Pre-processing:**

__Missing Values__<br>

```
# Import the Pandas library and assign it the alias pd
import pandas as pd

# Read the CSV file 'Data.csv' into a Pandas DataFrame called df
df = pd.read_csv('Data.csv')

# Calculate the number of missing values in each column of the DataFrame df
missing_values = df.isnull().sum()

# Filter the missing_values Series to include only columns with missing values
columns_with_missing_values = missing_values[missing_values > 0]

# Print the columns with missing values
print("Columns with missing values:\n", columns_with_missing_values)
```
<img width="200" alt="1" src="https://github.com/Md-Khid/Linear-Regression-Modelling/assets/160820522/f7974594-bfe8-4191-891a-ebcb1dbbc951">


#### Data Distribution
Based on the output, the columns "Limit," "Balance," "Education," "Marital," and "Age" contain some missing values. To address this, we need to understand the distribution of each column so that we can appropriately replace the missing values, such as using the mean, median, or mode.

```
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of the plot
sns.set_style("white")

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Define the columns to plot
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
![2](https://github.com/Md-Khid/Linear-Regression-Modelling/assets/160820522/6b75ff6c-564a-4005-9bce-95bf120be48f)











3. **Exploratory Data Analysis (EDA):**
   Explore the dataset to identify patterns, trends, and relationships within the data using descriptive statistics and visualisations.

4. **Insight Articulation:**
   Articulate relevant insights derived from the data analysis process, supported by visualisations.

5. **Linear Regression Modelling:**
   Build a linear regression model to predict the variable B1, explaining the approach taken and any necessary data pre-processing.

## Methodology
1. **Data Pre-processing:**
   - Handle missing data through appropriate imputation techniques.
   - Encode categorical variables.
   - Scale numerical features.

2. **Exploratory Data Analysis:**
   - Calculate descriptive statistics.
   - Generate visualisations such as histograms and scatter plots.

3. **Insight Articulation:**
   - Summarise key findings from the data analysis process.

4. **Linear Regression Modelling:**
   - Split the dataset into training and testing sets.
   - Apply linear regression algorithm to predict B1.
   - Evaluate model performance.


- Excel - Data Cleaning
  - [Download here](https://microsoft.com)
- SQL Server - Data Analysis
- PowerBI - Creating reports


### Data Cleaning/Preparation

In the initial data preparation phase, we performed the following tasks:
1. Data loading and inspection.
2. Handling missing values.
3. Data cleaning and formating.



