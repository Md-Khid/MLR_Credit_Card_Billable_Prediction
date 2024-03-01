# E-Commerce Sales Analysis

### Project Overview

This data analysis project aim to provide insights into the sales perfromance of an e-commerce company over the past year. By analysing various aspects of the sales data, we seek to identify trends, make data-driven recommnendations, and gain a deeper understanding of the company performance.

### Data Sources

Sales Data: The primary dataset used for this analysis is the "sales_data.csv" file, containing detailed information about each sales made by the company.

### Tools

- Excel - Data Cleaning
  - [Download here](https://microsoft.com)
- SQL Server - Data Analysis
- PowerBI - Creating reports


### Data Cleaning/Preparation

In the initial data preparation phase, we performed the following tasks:
1. Data loading and inspection.
2. Handling missing values.
3. Data cleaning and formating.

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

**Note**:

n=1 signifies the most recent month, while n=5 signifies the previous 4th month. 

If n=1 is the month of May 2022, then n=5 is the month of January 2024.


