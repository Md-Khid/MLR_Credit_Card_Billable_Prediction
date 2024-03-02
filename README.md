
# Project Overview

## Introduction
This project aims to analyse a credit facility dataset containing records of customers' demographics, outstanding amounts, repayment history/status, and other relevant variables. The goal is to extract meaningful insights from the data through exploratory data analysis and predictive modelling based on a mock-up credit facility dataset.

## Dataset Description

### Data Variables
The dataset comprises information about customers' credit facilities, including demographic details, outstanding amounts, and repayment history/status.
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


## Objectives
1. **Data Pre-processing:**
   Prepare the dataset for analysis by handling missing values, encoding categorical variables, and scaling numerical features.

2. **Exploratory Data Analysis (EDA):**
   Explore the dataset to identify patterns, trends, and relationships within the data using descriptive statistics and visualisations.

3. **Insight Articulation:**
   Articulate relevant insights derived from the data analysis process, supported by visualisations.

4. **Linear Regression Modelling:**
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



