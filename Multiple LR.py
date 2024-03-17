import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# Set a standard seaborn colour palette
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

# Display the DataFrame in the Variable Explorer or the console
df
#%%
# Calculate number of missing values 
missing_values = df.isnull().sum()

# Filter the missing_values
columns_with_missing_values = missing_values[missing_values > 0]

# Display columns with missing values
columns_with_missing_values
#%%
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
#%%
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
#%%
# Define special characters
special_chars = "!@#$%^&"

# Iterate over each column 
for column in df.columns:
    # Iterate over each row in current column
    for index, value in df[column].items():
        # Check if value contains any special characters
        if any(char in special_chars for char in str(value)):
            print(f"Special characters found in column '{column}', row {index}: {value}")
#%%
# Remove special characters ('$' and ',') and spaces from column 'R3'
df['R3'] = df['R3'].str.replace("$", "").str.replace(",", "").str.replace(" ", "")
#%%
# Identify categorical variables
categorical_variables = df.select_dtypes(include=['object', 'category']).columns

# Check for categorical variables that need encoding
if categorical_variables.empty:
    print("No categorical variables need encoding.")
else:
    print("The following categorical variables need encoding:")
    for var in categorical_variables:
        print(var)
#%%
# Convert 'R3' column to the same data type as 'R1', 'R2', 'R4', and 'R5'
df['R3'] = df['R3'].astype(df['R1'].dtype)
#%%
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
#%%
# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
df
#%%
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
#%%
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
#%%
# Define plot data
plot_data = [
    {'column_x': 'INCOME', 'column_y': 'EDUCATION', 'data': 'INCOME', 'title': '.'},
    {'column_x': 'BALANCE', 'column_y': 'EDUCATION', 'data': 'BALANCE', 'title': '.'}
]

# Custom colour palette for education levels 
custom_palette = {
    'Others': '#1f78b4',        
    'Postgraduate': '#33a02c',  
    'Tertiary': '#fdbf6f',     
    'High School': '#ff7f00'   
}

# Create subplots
fig, axes = plt.subplots(1, len(plot_data), figsize=(15, 6))

for i, plot_info in enumerate(plot_data):
    # Calculate mean values and sort by order
    mean_values = df.groupby(plot_info['column_y'])[plot_info['column_x']].mean().sort_values(ascending=False).index

    sns.boxplot(ax=axes[i], x=plot_info['column_x'], y=plot_info['column_y'],
                data=df, order=mean_values, palette=custom_palette)

    axes[i].set_xlabel(plot_info['data'].capitalize())
    axes[i].set_ylabel('.')
    axes[i].set_title(plot_info['title'])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
    axes[i].grid(False)

plt.tight_layout()
plt.show()
#%%
# Define education levels and corresponding colours
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
#%%
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
    # Get the colourblind palette
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
#%%
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
#%%
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
#%%
# Revert to original scale
df[numeric_columns] = scaler.inverse_transform(df[numeric_columns])
df
#%%
# Dummy columns
df = pd.get_dummies(df, columns=['RATING', 'GENDER', 'EDUCATION', 'MARITAL', 'S1', 'S2', 'S3', 'S4', 'S5'])

# Convert boolean columns to integers (1 and 0)
df[df.select_dtypes(include=bool).columns] = df.select_dtypes(include=bool).astype(int)
df
#%%
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
#%%
# Define predictors and target variable 
X = df.drop(columns=['B1'])  # Features (exclude the target variable)
y = df['B1']  # Target variable

# Split data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LinearRegression()

# Build step forward feature selection
sfs1 = sfs(clf, k_features=3, forward=True, floating=False, scoring='r2', cv=5)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)

# Get selected features
selected_features = list(sfs1.k_feature_names_)

# Fit the model using selected features
X_train_selected = sm.add_constant(X_train[selected_features])
model = sm.OLS(y_train, X_train_selected)
results = model.fit()

# Predict target variable for the training and testing set
y_train_pred = results.predict(X_train_selected)
X_test_selected = sm.add_constant(X_test[selected_features])
y_test_pred = results.predict(X_test_selected)

# Calculate the RMSE for training and testing set
rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = sqrt(mean_squared_error(y_test, y_test_pred))

print(f'Training RMSE: {round(rmse_train, 2)}')
print(f'Testing RMSE: {round(rmse_test, 2)}')

# Print the model summary
print(results.summary())
#%%
# Define columns for dummy encoding
dummy_cols = ['RATING', 'GENDER', 'EDUCATION', 'MARITAL', 'S1', 'S2', 'S3', 'S4', 'S5']

# Load and preprocess test data
df_test = (pd.read_csv('Test.Data.csv')
             .iloc[:, 1:]
             .pipe(pd.get_dummies, columns=dummy_cols)
             .assign(**{col: lambda df: df[col].astype(int) for col in df.select_dtypes(include=bool).columns})
             .pipe(lambda df: (sm.add_constant(df.drop(columns=['B1'])), df['B1']))
          )

# Align the test data with training data
X_test, y_test = df_test

# Ensure X_test only includes the selected features
X_test_selected = sm.add_constant(X_test[selected_features])

# Make predictions 
y_test_pred = results.predict(X_test_selected).round(0)

# Compare actual and predicted B1 values
df_compare = pd.DataFrame({'Actual B1': y_test, 'Predicted B1': y_test_pred})
df_compare

