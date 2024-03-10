
# Project Overview

## Introduction
The objective of this project is to conduct an in-depth analysis of a credit facility dataset, primarily focusing on predicting the billable amount (B1) for credit card customers. Through the utilisation of data analytics techniques such as exploratory data analysis (EDA) and predictive modelling, the aim is to derive valuable insights that can assist financial institutions in accurately forecasting future billable amounts.

## Dataset Information

### Data Variables
The dataset comprises various attributes pertaining to customers' credit facilities, including demographic details, outstanding balances, repayment histories, and socio-economic indicators. These variables serve as the cornerstone for our analysis and predictive modelling endeavours.
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

In this phase of data processing, we will refine the dataset for analysis by addressing missing values, handling special characters, and encoding variables. Additionally, we will import all necessary modules and libraries for the project and transform categorical variables into category columns for data visualisation purposes.

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
<img width="498" alt="1" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/f11a32b9-4a8f-4951-ab65-eb580a723ae2">

#### Check Missing Values

```
# Calculate number of missing values 
missing_values = df.isnull().sum()

# Filter the missing_values
columns_with_missing_values = missing_values[missing_values > 0]

# Display columns with missing values
columns_with_missing_values
```
<img width="204" alt="2" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/50421406-5b9c-4957-8a2d-31fd37c091e2">

Based on the output, it seems that the columns "Limit," "Balance," "Education," "Marital," and "Age" contain some missing values. To address this issue, we need to analyse the distribution of each column to decide on the most appropriate method for replacing the missing values. Possible approaches include using the mean, median, or mode depending on the data distribution.

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
![3](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/07663d5d-b3d7-4f77-8d55-d495c6d82b05)

Given the positively skewed distribution of data in the "Limit," "Balance," and "Age" columns, we can replace the missing values with the median values. For the "Marital" and "Age" columns, we can replace the missing values with the mode. Additionally, upon inspecting the Age distribution, an anomalous age value is observed lying between -1 to 0, as well as 200. To address this anomaly, we will remove such values from the Age column.

As for the overall data distribution plot

- **Credit Limits:**
  - Most customers have a credit limit below 200,000, with a significant number having limits around 50,000.
  - This suggests that the bank is cautious in extending high credit limits.

- **Balance:**
  - A large number of customers have low balances, indicating they are not utilising their full credit limit.
  - This could suggest that most customers are financially responsible and avoid maxing out their credit.

- **Age Distribution:**
  - The majority of customers are between 25 and 40 years old.
  - This could be the bank’s target demographic for credit card products.

- **Marital Status:**
  - There are more married customers than single or others.
  - This could suggest that married couples are more likely to apply for credit cards, reflecting the demographic profile of the bank’s customer base.

- **Education:**
  - Most customers have attained tertiary education followed by high school.
  - This could indicate that individuals with higher education levels are more likely to have credit cards.

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
![5](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/4d33be7e-10a6-45f9-a8ca-c3815c337f72)

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
![6](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/c2304d19-e302-472e-9d7d-5a803abce688)

```
# Convert 'R3' column to the same data type as 'R1', 'R2', 'R4', and 'R5'
df['R3'] = df['R3'].astype(df['R1'].dtype)
```
Based on the output, it appears that the 'R3' column may require encoding. However, according to the [data dictionary](#data-dictionary), 'R3' is expected to be numerical. To address this, we can adjust the data type of the 'R3' column to align with that of the 'R1', 'R2', 'R4', and 'R5' columns. For now, we will refrain from encoding the remaining categorical variables, as we intend to utilise them for generating other frequency tables, bar charts, or graphical methods to comprehend their distribution and relationships with other variables.

## Exploratary Data Analysis 
In this section, we will delve into comprehending the dataset. This encompasses tasks such as examining data distributions, identifying outliers, visualising correlations between variables, and detecting any irregularities or trends, then transforming the insights obtained into valuable information.

#### Descriptive Statistics
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
![7](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/566c5a7f-fc1a-453d-bc91-67146d0f241a)

Insights from Descriptive Statistics
- **Gender and Marital Status:**
  - The majority of customers are female (62%) and married.
  - This might indicate that married individuals, particularly females, are more likely to have credit cards or engage in banking activities.

- **Education Level:**
  - The most common education level among customers is tertiary education.
  - This suggests that individuals with higher education levels are more prevalent in the dataset.

- **Age Distribution:**
  - The average age of customers is approximately 35.55 years, with a standard deviation of 9.16.
  - This indicates that the customer base is relatively young, with a considerable spread in ages.

- **Credit Limits and Balances:**
  - The mean credit limit is $168,359.54, while the mean balance is $9,110.24.
  - On average, customers have significant credit limits compared to their current balances.

- **Income:**
  - The mean income of customers is $177,858.19, with a wide standard deviation of $143,137.41.
  - This indicates a diverse range of income levels among customers.

- **Repayment Amounts:**
  - The mean repayment amounts (R1, R2, R3, R4, R5) are relatively low compared to the credit limits and balances.
  - This could indicate that customers are not fully utilising their credit or are making minimum payments.


#### Scaling Numerical Features
```
# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
df
```
<img width="500" alt="8" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/532ad846-952a-437c-aeaa-313499149aeb">

Scaling numerical variables in a dataset helps in understanding the relationships between variables, particularly when creating scatterplots and conducting correlation analysis. It ensures that variables are comparable on a similar scale, facilitating more accurate interpretation of their relationships.

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

![9 (1)](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/d39dac92-ecf0-4df2-a5e3-e8cb1b2dadcb)
![9 (2)](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/3e114827-515b-48cc-95e3-e480df0fd309)

Based on the heatmap, the correlations between the variables suggest the following insights:

- **('B1', 'B2'), ('B1', 'B3'), ('B1', 'B4'), ('B1', 'B5'), ('B1', 'BALANCE')**: These correlations may indicate how the billable amounts in the first month relate to each other and to the customer's balance, potentially reflecting consistent spending patterns or repayment behaviours.

- **('B2', 'B3'), ('B2', 'B4'), ('B2', 'B5'), ('B2', 'BALANCE')**: Similarly, these correlations may reveal relationships between billable amounts in the second month and the customer's balance.

- **('B3', 'B4'), ('B3', 'B5'), ('B3', 'BALANCE')**: These correlations may provide insights into how billable amounts in the third month correlate with each other and with the customer's balance.

- **('B4', 'B5'), ('B4', 'BALANCE')**: These correlations may indicate the relationship between billable amounts in the fourth month and the customer's balance.

- **('B5', 'BALANCE')**: This correlation may illustrate how the billable amount in the fifth month relates to the customer's balance.

- **('INCOME', 'LIMIT')**: This correlation may offer insights into the relationship between a customer's income and their credit limit. A strong positive correlation could imply that customers with higher incomes tend to have higher credit limits, while a weak or negative correlation may suggest otherwise.


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
![10](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/237d246a-405f-4555-8d5d-3dcf66497e82)

Based on the density plots, they offer insights into how credit limits are distributed among different groups of customers, which can inform the bank's customer preferences.

- **Education:** Customers with tertiary education tend to have similar credit limits, suggesting that the bank may prefer to offer similar credit options to individuals with higher education levels.

- **Gender:** While both males and females exhibit similar credit limit distribution patterns, females tend to have slightly lower credit limits.

- **Marital Status:** The density plot for marital status reveals that singles have a higher peak density at lower credit limits compared to married individuals. This suggests that single individuals tend to have lower credit limits.


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
![11](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/4414a949-9d63-4ab9-920d-9d3956312a23)

From the box plots, we observed that postgraduates, despite having a wider range of income, tend to have the lowest median balance among all education groups. This could indicate a potential relationship between education level and financial behavior.

- **Financial Management Skills:** Postgraduates could be more adept at managing finances due to their higher level of education. This could lead to greater awareness of the implications of maintaining high credit balances, resulting in more diligent payment of balances.

- **Income Stability:** Postgraduates might benefit from more stable incomes, allowing them to pay off their credit balances regularly, hence leading to lower balances.

- **Credit Behavior:** Postgraduates may exhibit more cautious credit usage habits, preferring to reserve credit card usage for emergencies or specific purchases, which could result in lower balances.

- **Debt Aversion:** Postgraduates might be more averse to accumulating debt.


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
![12](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/d15f03a0-cd00-4180-bef6-a52bea8a3d64)

Based on the scatter plots

- **Concentration of Data Points:** There is a concentration of data points towards the lower end of the LIMIT axis across all plots. This suggests that a majority of customers have lower limits.

- **Education Levels:** Customers with “Others” as their highest education attained are less frequent in these datasets. This could indicate that most customers have at least completed high school education.

- **No Clear Pattern Based on Education:** There isn’t a clear pattern or segregation based on education levels; data points for all education categories are mixed throughout. This suggests that the customer’s education level may not have a significant impact on their limit or billable amount.

- **Similar Distribution Across All Plots:** The distribution and concentration of data points are similar across all five scatter plots (B1 to B5). This could imply that the billable amount in each month (B1 to B5) has a similar relationship with the total limit.


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

![13 1](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/ac878d31-38cf-4d0e-a0ed-8f1f5320c500)

Based on the GENDER groups

- **Prompt and Minimum Sum Payments:** For both genders across all five scatter plots (B1 vs. each of the five repayment statuses), there is a concentration of data points at the “Prompt Sum” and “Min Sum” categories. This suggests that most customers tend to make at least minimum payments promptly.

- **Delayed Payments:** There are fewer data points in the delayed payment categories (“One” to “Eight”), indicating that most customers avoid delayed payments. This could be due to various reasons such as financial discipline, avoidance of penalties, or maintaining a good credit score.

- **Gender Distribution:** The distribution of male and female customers is fairly similar across all categories of repayment status. This suggests that gender does not significantly influence the repayment status.



![13 2](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/dba771d7-2382-48fc-81be-17b5e7f47d94)

Based on the EDUCATION groups

- **Prompt and Minimum Sum Payments:** Across all five scatter plots (B1 vs. each of the five repayment statuses), there is a concentration of data points at the “Prompt Sum” and “Min Sum” categories for all education levels. This suggests that most customers, regardless of their education level, tend to make at least minimum payments promptly.

- **Delayed Payments:** There are fewer data points in the delayed payment categories (“One” to “Eight”), indicating that most customers avoid delayed payments. This pattern is consistent across all education levels.

- **Education Level Distribution:** The distribution of customers across different education levels is fairly similar across all categories of repayment status. This suggests that the customer’s education level does not significantly influence the repayment status.


![13 3](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/0f36a92f-02f1-4a24-847c-f557c97b13cc)


Based on the MARITAL groups

- **Prompt and Minimum Sum Payments:** Across all five scatter plots (B1 vs. each of the five repayment statuses), there is a concentration of data points at the “Prompt Sum” and “Min Sum” categories for all marital statuses. This suggests that most customers, regardless of their marital status, tend to make at least minimum payments promptly.

- **Delayed Payments:** There are fewer data points in the delayed payment categories (“One” to “Eight”), indicating that most customers avoid delayed payments. This pattern is consistent across all marital statuses.

- **Marital Status Distribution:** The distribution of customers across different marital statuses is fairly similar across all categories of repayment status. This suggests that the customer’s marital status does not significantly influence the repayment status.


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
![14](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/f72ce5e9-ecf3-4422-b3c2-12337e724e70)


Based on the INCOME scatterplots

**Balance vs Income:** The left plot shows a correlation between BALANCE and INCOME. Data points are densely populated towards the lower INCOME range and spread out as BALANCE increases. This could suggest that customers with lower incomes tend to have lower balances, while those with higher balances have a wider range of incomes.

**Age vs Income:** The right plot illustrates the distribution of AGE against INCOME. Data points are densely packed across all ages but slightly more concentrated at lower incomes. This could indicate that income does not significantly increase with age in this customer dataset.

**Customer Rating:** In both plots, blue dots (GOOD RATING) are more prevalent. This suggests that there are more customers with a good rating in these datasets, regardless of their balance, income, or age.


#### Barplot

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
![16](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/fcf95cd9-9a60-457b-8f28-943c88d2ce8d)


Based on the bar graphs

**Gender:** The graph shows that there are more female customers than male customers. Both genders predominantly have a good rating, but the proportion of good ratings is slightly higher for females.

**Marital Status:** The graph shows that the number of married and single customers is almost equal, and both are significantly higher than the number of customers in the ‘Others’ category. All marital statuses predominantly consist of customers with good ratings.

**Education:** The graph shows that the majority of customers have a tertiary education, followed by postgraduate, high school, and others. Across all education levels, the majority of customers have good ratings, with slightly higher proportions observed among tertiary and postgraduate customers.



## Linear Regression Modelling
In this phase, we will construct a linear regression model to forecast the variable B1

#### Reverting
```
# Revert to original scale
df[numeric_columns] = scaler.inverse_transform(df[numeric_columns])
df
```
![17](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/98dccfd2-3362-4911-b0c2-84cd10ba7d03)

Reverting the scaled predictions back to their original scale is essential for accurate evaluation. This process enables us to compare the model's predictions directly with the original target variable values. Without reverting the predictions, the evaluation metrics would be calculated on a different scale, leading to inaccurate assessments of the model's performance.


#### Dummy Variables
```
# Dummy columns
df = pd.get_dummies(df, columns=['RATING','GENDER','EDUCATION','MARITAL','S1','S2','S3','S4','S5'])

# Convert boolean columns to integers (1 and 0)
df[df.select_dtypes(include=bool).columns] = df.select_dtypes(include=bool).astype(int)
df
```
<img width="492" alt="18" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/82588fa7-c399-4d80-b91d-d2bcfa4db616">

Dummy variables are utilised to represent categorical data in a numerical format, primarily to meet the requirements of various machine learning algorithms and statistical models. These models typically necessitate numerical input for processing. By encoding categorical variables as dummy variables, we can effectively convert them into a format that can be directly inputted into these models. This ensures compatibility and seamless integration of categorical data into the modelling process, facilitating the analysis and prediction tasks performed by the algorithms.

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
<img width="362" alt="19" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/0a238edd-dadb-408d-bd0a-06f3af3140cf">

Outliers can significantly influence the parameters and performance of statistical models. Removing outliers can help in achieving more accurate and stable model estimates, leading to better predictive performance.

```
# Multicollinearity 

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
![20](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/e59f6607-4ebe-4d5a-b089-78cb8ac689d2)

Multicollinearity refers to the phenomenon where two or more independent variables in a regression model are highly correlated with each other. This can inflate the standard errors of the coefficients, making them unstable and difficult to interpret. To address this issue, it is essential to check for multicollinearity using the Variance Inflation Factor (VIF). A VIF value greater than 5 indicates a high degree of multicollinearity, and removing variables with high VIF values helps alleviate multicollinearity-related issues in the regression model.

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
![21](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/6fdf40e3-a4bd-4d89-9282-e443639a21ae)

The model underwent training and testing on sets divided into 70% for training and 30% for testing. Through iterations, variables with p-values greater than 0.05 were removed. It achieved an R-squared (R²) value of 0.949 and an F-statistic of 1.155e+04. The Root Mean Square Error (RMSE) for both the training and testing models is 7638.74. Considering the [descriptive statistics of B1](#descriptive-statistics), with a mean of approximately 49985.76 and a standard deviation of around 71927.41, the Test RMSE of 7638.74 is notably smaller than both the mean and the standard deviation. This suggests that the model's predictions are relatively close to the actual values.

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
![22](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/af9e451d-f470-403b-9dd6-2c03d7a330f9)


Using the test dataset allows for further evaluation of the model's performance on unseen data. Based on the model's performance, the predictions are relatively close to the actual values, indicating that the model generalises well to new data.


