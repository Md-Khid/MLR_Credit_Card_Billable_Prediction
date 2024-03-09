
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
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set a standard seaborn color palette
sns.set_palette("colorblind")


# Load the data
df = pd.read_csv('Data.csv')

# Drop the first column 
df = df.iloc[:, 1:]

# Define mapping dictionaries for categorical columns for visual plotting
gender_map = {0: 'Male', 1: 'Female'}
education_map = {0: 'Others', 1: 'Postgraduate', 2: 'Tertiary', 3: 'High School'}
marital_map = {0: 'Others', 1: 'Single', 2: 'Married'}
rating_map = {0: 'Good', 1: 'Bad'}
s_map = {-1: 'Prompt', 0: 'Min Sum', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

# Convert columns to categorical and apply mappings
df['GENDER'] = df['GENDER'].map(gender_map)
df['EDUCATION'] = df['EDUCATION'].map(education_map)
df['MARITAL'] = df['MARITAL'].map(marital_map)
df['RATING'] = df['RATING'].map(rating_map)
for col in ['S1', 'S2', 'S3', 'S4', 'S5']:
   df[col] = df[col].map(s_map)

# Display the modified dataframe
df
```
<img width="499" alt="1" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/3f12cb0d-d601-4335-a7bc-6be4a5507d42">

#### Check Missing Values

```
# Calculate number of missing values in each column 
missing_values = df.isnull().sum()

# Filter the missing_values i.e. only columns with missing values
columns_with_missing_values = missing_values[missing_values > 0]

# Display columns with missing values
columns_with_missing_values
```
<img width="199" alt="2" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/6e0781b1-309b-450b-b42b-be82a56e32a6">

Based on the output, it appears that the columns "Limit," "Balance," "Education," "Marital," and "Age" contain some missing values. To rectify this issue, we should first analyse the distribution of each column to determine the most suitable method for replacing the missing values, which could involve using the mean, median, or mode.

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
    
# Hide empty subplots
for i in range(len(columns), axes.size):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()
```
<img width="497" alt="3" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/49515e8a-f3e7-442a-bf33-513e5e10dd16">

Given the positively skewed distribution of data in the "Limit," "Balance," and "Age" columns, we can replace the missing values with the median values. For the "Marital" and "Age" columns, we can replace the missing values with the mode. Additionally, upon inspecting the Age distribution, an anomalous age value is observed lying between -1 to 0, as well as 200. To address this anomaly, we will remove such values from the Age column.

#### Replace Missing Values and Remove Data Errors 

```
# Remove rows where 'Age' column has a value of 0, -1, or 100 and above
df = df[(df['AGE'] > 0) & (df['AGE'] < 100)].copy()

# Specify columns and their corresponding fill methods
columns_to_fill = {
    'LIMIT': 'median',
    'BALANCE': 'median',
    'AGE': 'median',
    'MARITAL': 'mode',
    'EDUCATION': 'mode'
}

# Fill missing values in specified columns
for column, method in columns_to_fill.items():
    if method == 'median':
        df[column] = df[column].fillna(df[column].median())
    elif method == 'mode':
        df[column] = df[column].fillna(df[column].mode()[0])

# Check for missing values in columns 
missing_values = df.isnull().any()

# Calculate the total count of columns with missing values 
count_missing_values = missing_values.sum()

# Display number of columns with missing values
count_missing_values
```
<img width="45" alt="4" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/aaf41384-2017-466b-b592-bd9dc31e4638">

#### Removing Special Characters
```
# Iterate over each column 
for column in df.columns:
    # Iterate over each row in the current column
    for index, value in df[column].items():
        # Check if value contains any special characters
        if any(char in "!@#$%^&" for char in str(value)):
            print(f"Special characters found in column '{column}', row {index}: {value}")
```
<img width="243" alt="5" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/4dc47bf2-f75e-4368-9f4d-23b146507ab2">

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
<img width="237" alt="6" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/22070eb7-cdce-40b0-b5b7-666ad7d8206f">

```
# Convert 'R3' column to the same data type as 'R1', 'R2', 'R4', and 'R5'
df['R3'] = df['R3'].astype(df['R1'].dtype)
```
Based on the output, it appears that the 'R3' column may require encoding. However, according to the [data dictionary](#data-dictionary), 'R3' is expected to be numerical. To address this, we can adjust the data type of the 'R3' column to align with that of the 'R1', 'R2', 'R4', and 'R5' columns. For now, we will refrain from encoding the remaining categorical variables, as we intend to utilise them for generating other frequency tables, bar charts, or graphical methods to comprehend their distribution and relationships with other variables.

## Exploratary Data Analysis 
In this section, we will delve into comprehending the dataset. This encompasses tasks such as examining data distributions, identifying outliers, visualising correlations between variables, and detecting any irregularities or trends, then transforming the insights obtained into valuable information.

#### Descriptive Statisitcs
```
# Create statistical description table 
statistical_description = df.describe(include='all')

# Round statistical description table to 2 decimal places
statistical_description = np.round(statistical_description, 2)

# Separate columns into categorical and numerical groups
categorical_columns = statistical_description.columns[statistical_description.dtypes == 'object']
numeric_columns = [col for col in statistical_description.columns if col not in categorical_columns]

# Concatenate columns in order (categorical followed by numerical)
statistical_description = pd.concat([statistical_description[categorical_columns], statistical_description[numeric_columns]], axis=1)

# Transpose statistical description table
statistical_description = statistical_description.T

# Display statistical table
statistical_description
```
<img width="383" alt="7" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/74809df8-880f-4b92-9265-5d89bed6472d">

Based on the statistical table, the credit card bank typically offers a uniform credit limit based on customers' income. However, a significant portion of its customers struggle to meet their credit card bill payments once they have utilised approximately 97% of their credit limit. They can only afford to make a small payment towards their monthly bills, around 10%. This serves as a clear signal for the credit card bank to issue reminder notices and make phone calls or impose late fees and additional interest charges on unpaid balances, leading to an increase in the outstanding amount over time.

## Insight Articulation

#### Scaling Numerical Features
```
# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Display data table
df
```
<img width="751" alt="8" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/c4ae83ec-81cf-43c5-baa2-490be2674578">

Scaling numerical variables in a dataset helps interpret relationships between variables, especially in scatterplots and correlation analysis. It helps to ensure they are on a similar scale.

#### Heatmap
```
# Select numerical columns
numerical_columns = df.select_dtypes(include='number')

# Plotting heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numerical_columns.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.show()
```
![9](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/cbc3a167-f442-4c44-9bb2-5770654c0b53)

Based on the correlation heatmap, it is clear that there is a strong correlation between variables such as 'INCOME' and 'LIMIT', as well as 'B(n)' and 'BALANCE'.


#### Density Plot
```
# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define variables for plotting
plot_data = [('EDUCATION', 'Education Level', ['Others', 'Postgraduate', 'Tertiary', 'High School']),
             ('GENDER', 'Gender', ['Male', 'Female']),
             ('MARITAL', 'Marital Status', ['Others', 'Single', 'Married'])]

# Plot each density plot in a subplot
for idx, (variable, title, labels) in enumerate(plot_data):
    sns.kdeplot(data=df, x='LIMIT', hue=variable, fill=True, ax=axes[idx])
    axes[idx].set_xlabel('LIMIT')
    axes[idx].set_ylabel('Density')
    axes[idx].legend(title=variable, labels=labels)

plt.tight_layout()
plt.show()
```
<img width="493" alt="8" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/3b10bb4e-7f10-4e04-ad84-9533fbe20ceb">

Based on the density plots, it is clear that the credit card bank tends to offer higher credit limits to customers who are 1. Male, 2. Married, and 3. Have a postgraduate education.


#### Boxplot
```
# Define the variables for iteration
plot_data = [
    {'column_x': 'INCOME', 'column_y': 'EDUCATION', 'data': 'INCOME', 'order': ['Others', 'Postgraduate', 'Tertiary', 'High School'], 'title':'.'},
    {'column_x': 'BALANCE', 'column_y': 'EDUCATION', 'data': 'BALANCE', 'order': ['Others', 'Postgraduate', 'Tertiary', 'High School'], 'title':'.'}
]

# Define a fixed color dictionary for specific education levels
education_colors = {'Tertiary': 'blue', 'Postgraduate': 'green', 'High School': 'orange', 'Others': 'red'}

# Create a figure with subplots
fig, axes = plt.subplots(1, len(plot_data), figsize=(15, 6))

# Iterate through the plot_data and create subplots
for i, plot_info in enumerate(plot_data):
    # Calculate the mean for the current group and sort the order accordingly
    mean_values = df.groupby(plot_info['column_y'])[plot_info['column_x']].mean().sort_values(ascending=False).index
    sns.boxplot(ax=axes[i], x=plot_info['column_x'], y=plot_info['column_y'], data=df, order=mean_values, palette=education_colors)
    axes[i].set_xlabel(plot_info['data'].capitalize())
    axes[i].set_ylabel('.')
    axes[i].set_title(plot_info['title'])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)  # Rotate x-axis labels
    axes[i].grid(False)

plt.tight_layout()
plt.show()

```
![11](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/175eca19-66da-4dd2-9fe0-4e54d42165c3)

Based on the Income box plot, it is clear that individuals with high incomes mainly belong to the Tertiary Education group. However, this group demonstrates lower credit balances in comparison to the Postgraduate and High School groups.

#### Scatterplot
```
# Define colors for each education category
colors = {'Postgraduate': 'blue', 'Tertiary': 'green', 'High School': 'orange', 'Others': 'purple'}

# Define the variables to plot
variables = ['B1', 'B2', 'B3', 'B4', 'B5']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Plotting
for i, variable in enumerate(variables):
    ax = axes[i]
    for education, color in colors.items():
        ax.scatter(df[df['EDUCATION'] == education][variable], df[df['EDUCATION'] == education]['LIMIT'], color=color, label=education, alpha=0.6)
    ax.set_xlabel(variable)
    ax.set_ylabel('LIMIT')
    ax.legend()

# Hide empty subplots
for j in range(len(variables), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```
![11](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/471a66e0-584a-47f1-adbb-3c67f47133b6)

Spell check: The bill amount owned by customers kept increasing as the month increasses. And most of the high sizeabbe bill amoout is concerntrated on the Postgraduates and Tertiary education group as they mosntly obtained a higer credit limit

```
# Define the variables to plot
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
    for i, variable in enumerate(variables):
        ax = axes[i]

        # Ensure df contains the current hue column
        if hue in df.columns:
            sns.stripplot(x=df[variable], y=df['B1'], hue=df[hue], ax=ax, palette=hue_colors, order=variable_order[variable], jitter=True, dodge=True)
            ax.set_xlabel(variable)
            ax.set_ylabel('B1')
            ax.grid(True)

            # Create a legend for the current hue
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in hue_colors]
            ax.legend(handles, df[hue].unique(), loc='upper right')

            if i >= 3:
                ax.set_xlabel('')
                ax.set_xticklabels([])
        else:
            print(f"Warning: DataFrame does not contain column '{hue.upper()}'")

    # Hide empty subplots
    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
```

![12](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/cfb062c7-38dd-400f-a11b-efd23e9e07dc)

```

# Define the variables for the scatter plots
variables = ['BALANCE', 'AGE']

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plotting
for i, var in enumerate(variables):
    ax = axes[i]
    # Use colorblind palette for scatter plot
    sns.scatterplot(data=df, x=var, y='INCOME', hue='RATING', palette={'Good': sns.color_palette()[0], 'Bad': sns.color_palette()[3]}, alpha=0.6, ax=ax)
    ax.set_xlabel(var)
    ax.set_ylabel('INCOME')
    ax.legend(title='RATING')

plt.tight_layout()
plt.show()
```
![13](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/60cab0e2-19b8-467b-b5c1-d4297e6f9a9b)


```
#### Barplot
```
# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Calculate frequencies for each rating category for GENDER, MARITAL, and EDUCATION
for i, category in enumerate(['GENDER', 'MARITAL', 'EDUCATION']):
    # Count the occurrences of each category in the RATING column
    education_counts = df[df['RATING'].isin(['Good', 'Bad'])].groupby([category, 'RATING']).size().unstack(fill_value=0)

    # Plotting
    # Plotting with sorted frequencies
    education_counts.sort_values(by='Good', ascending=False).plot(kind='bar', stacked=True, ax=axes[i], color={'Good': sns.color_palette()[0], 'Bad': sns.color_palette()[3]}, alpha=0.6)
    axes[i].set_xlabel(category)
    axes[i].set_ylabel('Frequency')
    axes[i].legend(title='RATING')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)  # Rotate x-axis labels

plt.tight_layout()
plt.show()
```
![14](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/924d50b9-a72e-48c9-8e0e-2b8045b6721d)

Based on the barplots, it is clear that the credit card bank tends to provide Good ratings to customers who are 1. Female, 2. Married, and 3. Have a Tertiary education.


## Linear Regression Modelling
Build a linear regression model to predict the variable B1, explaining the approach taken and any necessary data pre-processing.

## Evaluate Model Performance





- Excel - Data Cleaning
  - [Download here](https://microsoft.com)
- SQL Server - Data Analysis
- PowerBI - Creating reports





