
# Project Overview

## Introduction
The objective of this project is to conduct an in-depth analysis of a credit facility dataset, primarily focusing on predicting the billable amount (B1) for credit card customers. Through the utilisation of data analytics techniques such as exploratory data analysis (EDA) and predictive modelling, the aim is to derive valuable insights that can assist financial institutions in accurately forecasting future billable amounts.

## Dataset Information

### Data Variables
The dataset comprises various attributes pertaining to customers' credit facilities, including demographic details, outstanding balances, repayment histories, and socio-economic indicators. These variables serve as the foundation for our analysis and predictive modelling process.
[Dataset](https://github.com/Md-Khid/Linear-Regression-Modelling/blob/main/Data.csv)


## Data Preparation

In this phase of data processing, we will refine the dataset for analysis by addressing missing values, handling special characters, and encoding variables. Additionally, we will import all necessary modules and libraries for the project and transform categorical variables into category columns for data visualisation purposes.

### Data Pre-processing:

<img width="498" alt="1" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/f11a32b9-4a8f-4951-ab65-eb580a723ae2">

#### Check Missing Values

<img width="204" alt="2" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/50421406-5b9c-4957-8a2d-31fd37c091e2">

Based on the output, it seems that the columns "Limit," "Balance," "Education," "Marital," and "Age" contain some missing values. To address this issue, we need to analyse the distribution of each column to decide on the most appropriate method for replacing the missing values. Possible approaches include using the mean, median, or mode depending on the data distribution.

#### View Data Distribution

![3](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/07663d5d-b3d7-4f77-8d55-d495c6d82b05)

Given the positively skewed distribution of data in the "Limit," "Balance," and "Age" columns, we can replace the missing values with the median values. For the "Marital" and "Education" columns, we can replace the missing values with the mode. Additionally, upon inspecting the Age distribution, we notice an anomalous age value present between -1 and 0, as well as in the range of 200. To address this anomaly, we will remove such values from the Age column.

#### Replace Missing Values and Remove Data Errors 

<img width="60" alt="4" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/c5a78a38-3298-46d3-8e58-eafc9bf212b0">

#### Removing Special Characters

![5](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/4d33be7e-10a6-45f9-a8ca-c3815c337f72)

Based on the output, it seems that the R3 column contains special characters. To address this, we replace these characters with an empty string.

#### Encoding of Variables
![6](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/c2304d19-e302-472e-9d7d-5a803abce688)

Based on the output, it appears that the 'R3' column may require encoding. However, based on the data structure, the  'R3' column is expected to be numerical. To address this, we can adjust the data type of the 'R3' column to align with that of the 'R1', 'R2', 'R4', and 'R5' columns. For now, we will refrain from encoding the remaining categorical variables, as we intend to utilise them for generating other frequency tables, bar charts, or graphical methods to comprehend their distribution and relationships with other variables.

## Exploratory Data Analysis 
In this section, we will delve into comprehending the dataset. This encompasses tasks such as examining data distributions, identifying outliers, visualising correlations between variables, and detecting any irregularities or trends, then transforming the insights obtained into valuable information.

#### Descriptive Statistics

![7](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/566c5a7f-fc1a-453d-bc91-67146d0f241a)

#### Scaling Numerical Features

<img width="500" alt="8" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/532ad846-952a-437c-aeaa-313499149aeb">

#### Heatmap

![9 (1)](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/d39dac92-ecf0-4df2-a5e3-e8cb1b2dadcb)
![9 (2)](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/3e114827-515b-48cc-95e3-e480df0fd309)

Based on the heatmap, the correlations between the variables suggest

- **('B1', 'B2'), ('B1', 'B3'), ('B1', 'B4'), ('B1', 'B5'), ('B1', 'BALANCE')**: These correlations may indicate how the billable amounts in the first month relate to each other and to the customer's balance, potentially reflecting consistent spending patterns or repayment behaviours.

- **('B2', 'B3'), ('B2', 'B4'), ('B2', 'B5'), ('B2', 'BALANCE')**: Similarly, these correlations may reveal relationships between billable amounts in the second month and the customer's balance.

- **('B3', 'B4'), ('B3', 'B5'), ('B3', 'BALANCE')**: These correlations may provide insights into how billable amounts in the third month correlate with each other and with the customer's balance.

- **('B4', 'B5'), ('B4', 'BALANCE')**: These correlations may indicate the relationship between billable amounts in the fourth month and the customer's balance.

- **('B5', 'BALANCE')**: This correlation may illustrate how the billable amount in the fifth month relates to the customer's balance.

- **('INCOME', 'LIMIT')**: This correlation may offer insights into the relationship between a customer's income and their credit limit. A strong positive correlation could imply that customers with higher incomes tend to have higher credit limits, while a weak or negative correlation may suggest otherwise.

#### Density Plot

![10](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/237d246a-405f-4555-8d5d-3dcf66497e82)

Based on the density plots:

- **Education:** Customers with tertiary education tend to have similar credit limits, suggesting that the bank may prefer to offer similar credit options to individuals with higher education levels.

- **Gender:** While both males and females exhibit similar credit limit distribution patterns, females tend to have slightly lower credit limits.

- **Marital Status:** The density plot for marital status indicates that married individuals have a higher peak density at lower credit limits compared to singles. This suggests that married individuals tend to have lower credit limits.

#### Boxplot

![11](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/c2ec2fd9-3f03-4826-b20a-c1f5d9cce414)

From the box plots, we observed that Postgraduates, despite having a wider range of income, tend to exhibit the lowest median balance among the Others and Tertiary education groups. This observation suggests a potential relationship between education level and financial behavior.”

- **Financial Management Skills:** Postgraduates could be more adept at managing finances due to their higher level of education. This could lead to greater awareness of the implications of maintaining high credit balances, resulting in more diligent payment of balances.

- **Income Stability:** Postgraduates might benefit from more stable incomes, allowing them to pay off their credit balances regularly, hence leading to lower balances.

- **Credit Behaviour:** Postgraduates may exhibit more cautious credit usage habits, preferring to reserve credit card usage for emergencies or specific purchases, which could result in lower balances.

- **Debt Aversion:** Postgraduates might be more averse to accumulating debt.

#### Scatterplot

![12](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/d15f03a0-cd00-4180-bef6-a52bea8a3d64)

Based on the scatter plots

- **Concentration of Data Points:** There is a concentration of data points towards the lower end of the LIMIT axis across all plots. This suggests that a majority of customers have lower limits.

- **Education Levels:** Customers with “Others” as their highest education attained are less frequent in these datasets. This could indicate that most customers have at least completed high school education.

- **No Clear Pattern Based on Education:** There is not a clear pattern or segregation based on education levels; data points for all education categories are mixed throughout. This suggests that the customer’s education level may not have a significant impact on their limit or billable amount.

- **Similar Distribution Across All Plots:** The distribution and concentration of data points are similar across all five scatter plots (B1 to B5). This could imply that the billable amount in each month (B1 to B5) has a similar relationship with the total limit.

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


![14](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/f72ce5e9-ecf3-4422-b3c2-12337e724e70)


Based on the INCOME scatterplots

**Balance vs Income:** The left plot shows a correlation between BALANCE and INCOME. Data points are densely populated towards the lower INCOME range and spread out as BALANCE increases. This could suggest that customers with lower incomes tend to have lower balances, while those with higher balances have a wider range of incomes.

**Age vs Income:** The right plot illustrates the distribution of AGE against INCOME. Data points are densely packed across all ages but slightly more concentrated at lower incomes. This could indicate that income does not significantly increase with age in this customer dataset.

**Customer Rating:** In both plots, blue dots (GOOD RATING) are more prevalent. This suggests that there are more customers with a good rating in these datasets, regardless of their balance, income, or age.


#### Barplot

![16](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/fcf95cd9-9a60-457b-8f28-943c88d2ce8d)

Based on the bar graphs

**Gender:** The graph shows that there are more female customers than male customers. Both genders predominantly have a good rating, but the proportion of good ratings is slightly higher for females.

**Marital Status:** The graph shows that the number of married and single customers is almost equal, and both are significantly higher than the number of customers in the ‘Others’ category. All marital statuses predominantly consist of customers with good ratings.

**Education:** The graph shows that the majority of customers have a tertiary education, followed by postgraduate, high school, and others. Across all education levels, the majority of customers have good ratings, with slightly higher proportions observed among tertiary and postgraduate customers.


## Linear Regression Modelling
In this phase, we will construct a linear regression model to forecast the variable B1

#### Reverting

![17](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/98dccfd2-3362-4911-b0c2-84cd10ba7d03)

Reverting the scaled predictions back to their original scale is essential for accurate evaluation. 


#### Dummy Variables

<img width="492" alt="18" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/82588fa7-c399-4d80-b91d-d2bcfa4db616">

Dummy variables are utilised to represent categorical data in a numerical format, primarily to meet the requirements of various machine learning algorithms and statistical models. These models typically necessitate numerical input for processing. 

#### Remove Outliers

<img width="362" alt="19" src="https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/0a238edd-dadb-408d-bd0a-06f3af3140cf">

Outliers can significantly influence the parameters and performance of statistical models. Removing outliers can help in achieving more accurate and stable model estimates, leading to better predictive performance.

### Modelling

The model underwent training and testing on datasets divided into 70% for training and 30% for testing. During this process, variables with p-values greater than 0.05 were iteratively removed using forward variable selection. Despite this refinement, the model maintained its predictive capability, as evidenced by an R-squared (R²) value that remained high, shifting marginally from 0.949 to 0.948. Additionally, while the F-statistic increased substantially from 1.614e+04 to 5.319e+04, both iterations of the model continued to provide statistically significant predictions of the dependent variable based on the independent variables.

The Root Mean Square Error (RMSE) difference for the training and testing models is significantly small for both the 10 and 3 selection features. Considering the descriptive statistics of [B1](#Descriptive-Statistics), with a mean of approximately 49961.41 and a standard deviation of around 71888.94, the Test RMSE for both the 10 and 3 selection features (i.e., 7478.54 and 7492.75) is notably smaller than both the mean and the standard deviation. This suggests that the model's predictions are relatively close to the actual values.

##### 10 Predictor Variables
![20 1](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/12139f33-8538-4abd-a204-c2f1daece375)
##### 3 Predictor Variables
![20 2](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/0954a735-d4e4-40bf-935f-5b02a82f399a)

## Evaluate Model Performance

![21](https://github.com/Md-Khid/Multiple-Linear-Regression/assets/160820522/97673859-8ad3-4ce7-b943-e3b9baa791b1)

Using the [test](https://github.com/Md-Khid/Linear-Regression-Modelling/blob/main/Test.Data.csv) dataset allows for further evaluation of the model's performance on unseen data. Based on the model's performance, the predictions are relatively close to the actual values indicating that the model generalises well to new data.

### Conclusion

In conclusion, this project has effectively addressed the objective of conducting a comprehensive analysis of the credit facility dataset, with a specific focus on predicting the billable amount (B1) for credit card customers. By employing data analytics techniques such as exploratory data analysis (EDA) and predictive modelling, valuable insights can be derived to assist financial institutions in accurately forecasting future billable amounts.

Through iterative processes, a linear regression model was developed and refined, achieving a commendable R-squared (R²) value of 0.948 and an F-statistic of 5.319e+04 with the Root Mean Square Error (RMSE) of 7492.75 indicating the model's capability to make predictions with reasonable accuracy.

Furthermore, evaluation using the test dataset revealed that the model's predictions closely aligned with the actual values, demonstrating its robustness in generalising to new data. This underscores the significance of rigorous model evaluation and refinement in ensuring the reliability of predictive analytics in the financial domain.

Overall, this project contributes valuable insights towards the use of data-driven tools to enhance decision-making processes and optimise resource allocation in managing credit facilities effectively.


