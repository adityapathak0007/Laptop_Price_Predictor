import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,  mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
import pickle

df = pd.read_csv("D:\\Aditya's Notes\\All Projects\\Laptop Price Predictor\\laptop_data.csv")
print(df.head())
print(df.shape)
print(df.info())

print(df.duplicated().sum())
print(df.isnull().sum())

df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.info())

df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')
print(df.head())
print(df.info())

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')
print(df.head())
print(df.info())

'''
sns.displot(df['Price'])
df['Company'].value_counts().plot(kind='bar')
plt.show()

sns.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df['TypeName'].value_counts().plot(kind='bar')
plt.show()

sns.barplot(x=df['TypeName'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.displot(df['Inches'])
plt.show()

sns.scatterplot(x=df['Inches'], y=df['Price'])
plt.show()

'''

'''

# 1. Distribution Plot for 'Price'
sns.displot(df['Price'], kde=True, color='blue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Density')
plt.grid(visible=True, linestyle='--', alpha=0.6)
# plt.show()

# 2. Bar Chart for 'Company' Value Counts
df['Company'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Products by Company')
plt.xlabel('Company')
plt.ylabel('Count')
plt.grid(visible=True, linestyle='--', alpha=0.6)
# plt.show()

# 3. Bar Plot for 'Company' vs 'Price'
sns.barplot(x=df['Company'], y=df['Price'], palette='viridis')
plt.xticks(rotation='vertical')
plt.title('Average Price by Company')
plt.xlabel('Company')
plt.ylabel('Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)
# plt.show()

# 4. Bar Chart for 'TypeName' Value Counts
df['TypeName'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Number of Products by TypeName')
plt.xlabel('TypeName')
plt.ylabel('Count')
plt.grid(visible=True, linestyle='--', alpha=0.6)
# plt.show()

# 5. Bar Plot for 'TypeName' vs 'Price'
sns.barplot(x=df['TypeName'], y=df['Price'], palette='coolwarm')
plt.xticks(rotation='vertical')
plt.title('Average Price by TypeName')
plt.xlabel('TypeName')
plt.ylabel('Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)
# plt.show()

# 6. Distribution Plot for 'Inches'
sns.displot(df['Inches'], kde=True, color='purple')
plt.title('Inches Distribution')
plt.xlabel('Inches')
plt.ylabel('Density')
plt.grid(visible=True, linestyle='--', alpha=0.6)
# plt.show()

# 7. Scatter Plot for 'Inches' vs 'Price'
sns.scatterplot(x=df['Inches'], y=df['Price'], hue=df['Company'], palette='Set1')
plt.title('Price vs Inches by Company')
plt.xlabel('Inches')
plt.ylabel('Price')
plt.legend(title='Company')
plt.grid(visible=True, linestyle='--', alpha=0.6)
# plt.show()
'''

print(df['ScreenResolution'].value_counts())

df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
print(df.head())
print(df.info())
print(df.sample())

'''

# Bar Chart for 'Touchscreen' Value Counts
df['Touchscreen'].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Count of Touchscreen Devices')
plt.xlabel('Touchscreen (Yes/No)')
plt.ylabel('Count')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for clarity
plt.show()

# Bar Plot for 'Touchscreen' vs 'Price'
sns.barplot(x=df['Touchscreen'], y=df['Price'], palette='muted')  # Use a muted color palette
plt.title('Average Price by Touchscreen Availability')
plt.xlabel('Touchscreen (Yes/No)')
plt.ylabel('Average Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()

'''

df['IPS'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
print(df.head())
print(df.info())
print(df.sample())

'''
# Bar Chart for 'IPS' Value Counts
df['IPS'].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Count of IPS Display Devices')
plt.xlabel('IPS Display (Yes/No)')
plt.ylabel('Count')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for clarity
plt.show()

# Bar Plot for 'IPS' vs 'Price'
sns.barplot(x=df['IPS'], y=df['Price'], palette='muted')  # Use a muted color palette
plt.title('Average Price by IPS Display Availability')
plt.xlabel('IPS Display (Yes/No)')
plt.ylabel('Average Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()
'''


new = df['ScreenResolution'].str.split('x', n=1, expand=True)
df['X_res'] = new[0]
df['Y_res'] = new[1]
print(df.head())
print(df.info())
print(df.sample(5))

df['X_res'] = df['X_res'].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')
print(df.head())
print(df.info())
print(df.sample(5))

numeric_df = df.select_dtypes(include=[float, int])
print(numeric_df.corr()['Price'])

df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

numeric_df = df.select_dtypes(include=[float, int])
print(numeric_df.corr()['Price'])

df.drop(columns=['ScreenResolution'], inplace=True)
df.drop(columns=['Inches', 'X_res', 'Y_res'], inplace=True)

print(df.head())

print(df['Cpu'].value_counts())

df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
print(df.head())


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
print(df.head())

'''
df['Cpu brand'].value_counts().plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')
plt.title('CPU Brand Distribution', fontsize=16)
plt.xlabel('CPU Brand', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()

# Bar Plot for 'Processor' vs 'Price'
sns.barplot(x=df['Cpu brand'], y=df['Price'], palette='muted')  # Use a muted color palette
plt.title('Average Price by Processor')
plt.xlabel('Processor')
plt.ylabel('Average Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
#plt.show()
'''

df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)
print(df.head())

#print(df['Ram'].value_counts())

'''
# Plotting the RAM distribution
df['Ram'].value_counts().plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')
plt.title('RAM Distribution', fontsize=16)
plt.xlabel('RAM Size (GB)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Bar Plot for 'RAM Size' vs 'Price'
sns.barplot(x=df['Ram'], y=df['Price'], palette='muted')  # Use a muted color palette
plt.title('Average Price by RAM Size')
plt.xlabel('RAM Size')
plt.ylabel('Average Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()
'''

# print(df['Memory'].value_counts())


# Fill missing values in the 'Memory' column
df['Memory'] = df['Memory'].fillna('0')

# Preprocess the 'Memory' column
df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)  # Remove '.0' if present
df['Memory'] = df['Memory'].str.replace('GB', '')  # Remove 'GB'
df['Memory'] = df['Memory'].str.replace('TB', '000')  # Convert 'TB' to equivalent GB

# Split the Memory column into 'first' and 'second' components
new = df["Memory"].str.split("+", n=1, expand=True)
df["first"] = new[0].str.strip()  # Strip whitespace
df["second"] = new[1].fillna("0").str.strip()  # Replace NaN in 'second' with "0"

# Identify storage types in the 'first' component
df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

# Remove non-numeric characters from 'first'
df['first'] = df['first'].str.replace(r'\D', '', regex=True).replace('', '0').astype(int)

# Identify storage types in the 'second' component
df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

# Remove non-numeric characters from 'second'
df['second'] = df['second'].str.replace(r'\D', '', regex=True).replace('', '0').astype(int)

# Calculate total storage by type
df["HDD"] = (df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
df["SSD"] = (df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])
df["Hybrid"] = (df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"])
df["Flash_Storage"] = (df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"])

# Drop unnecessary columns
df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                 'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                 'Layer2Flash_Storage'], inplace=True)

# Output the final DataFrame
print(df.head())
print(df.info())
print(df.sample(5))

df.drop(columns=['Memory'], inplace=True)

numeric_df = df.select_dtypes(include='number')
print(numeric_df.corr()['Price'])

df.drop(columns=['Hybrid', 'Flash_Storage'], inplace=True)
print(df.head())

print(df['Gpu'].value_counts())

df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
print(df.head())
df = df[df['Gpu brand'] != 'ARM']
print(df['Gpu brand'].value_counts())

'''
# Bar Plot for 'GPU Brand' vs 'Price'
sns.barplot(x=df['Gpu brand'], y=df['Price'], palette='muted')  # Use a muted color palette
plt.title('Average Price by GPU Brand')
plt.xlabel('GPU Brand')
plt.ylabel('Average Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()
'''

df.drop(columns=['Gpu'], inplace=True)
print(df.head())

print(df['OpSys'].value_counts())

'''
# Bar Plot for 'Operating Systems' vs 'Price'
sns.barplot(x=df['OpSys'], y=df['Price'], palette='muted')  # Use a muted color palette
plt.title('Average Price by OS')
plt.xlabel('OS')
plt.ylabel('Average Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()
'''


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


df['os'] = df['OpSys'].apply(cat_os)
print(df.head())

df.drop(columns=['OpSys'], inplace=True)

'''
# Bar Plot for 'Operating Systems' vs 'Price'
sns.barplot(x=df['os'], y=df['Price'], palette='muted')  # Use a muted color palette
plt.title('Average Price by OS')
plt.xlabel('OS')
plt.ylabel('Average Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()

# Create the distribution plot for 'Operating Systems' vs 'Price'
sns.displot(df, x='Price', hue='os', kind='kde', palette='muted', common_norm=False)
plt.title('Price Distribution by Operating System')
plt.xlabel('Price')
plt.ylabel('Density')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()
'''

'''
# Create the distribution plot for 'Weight'
sns.displot(df['Weight'])
plt.title('Weight Distribution')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()

# Create the scatter plot for 'Weight' vs 'Price'
sns.scatterplot(x=df['Weight'], y=df['Price'], color='blue')
plt.title('Price vs Weight')
plt.xlabel('Weight')
plt.ylabel('Price')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()
'''

numeric_df = df.select_dtypes(include='number')
print(numeric_df.corr()['Price'])

'''
# Create a heatmap of the correlation matrix
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Create the distribution plot for the logarithm of 'Price'
sns.displot(np.log(df['Price']), kde=True, color='purple')  # Use a color of your choice
plt.title('Logarithm of Price Distribution')
plt.xlabel('Log(Price)')
plt.ylabel('Density')
plt.grid(visible=True, linestyle='--', alpha=0.6)  # Add gridlines for readability
plt.show()
'''

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

'''
# print("Algorithm: Linear Regression")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0,1,7,10,11])
], remainder='passthrough')

# Step 2: Ridge Regression
step2 = LinearRegression()
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("***********************************")
print("Algorithm: Linear Regression")
print('R2 Score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# print("Algorithm: Ridge Regression")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: Ridge Regression
step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Ridge Regression")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))


# print("Algorithm: Lasso Regression")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: Lasso Regression
step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Lasso Regression")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))


# print("Algorithm: KNeighbors Regressor")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: KNeighbors Regressor
step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: KNeighbors Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
'''

# print("Algorithm: Decision Tree Regressor")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: Decision Tree Regressor
step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Decision Tree Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))


# print("Algorithm: Random Forest Regressor")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: Random Forest Regressor
step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Random Forest Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

'''
# print("Algorithm: Gradient Boosting Regressor")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: Gradient Boosting Regressor
step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Gradient Boosting Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))


# print("Algorithm: AdaBoost Regressor")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: AdaBoost Regressor
step2 = AdaBoostRegressor(n_estimators=15, learning_rate=1.0)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: AdaBoost Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))


# print("Algorithm: Extra Trees Regressor")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: Extra Trees Regressor
step2 = ExtraTreesRegressor(bootstrap=True,
                            n_estimators=100,
                            random_state=3,
                            max_samples=0.5,
                            max_features=0.75,
                            max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Extra Trees Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))


# print("Algorithm: Support Vector Regressor (SVR)")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: SVR
step2 = SVR(kernel='rbf', C=10000, epsilon=0.1)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Support Vector Regressor (SVR)")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))


# print("Algorithm: XGB Regressor")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: XGB Regressor
step2 = XGBRegressor(n_estimators=45, max_depth=5, learning_rate=0.5)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: XGB Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

# print("Algorithm: Voting Regressor")

# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: Define base regressors
rf = RandomForestRegressor(n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)
gbdt = GradientBoostingRegressor(n_estimators=100, max_features=0.5)
xgb = XGBRegressor(n_estimators=25, max_depth=5, learning_rate=0.3)
lr = LinearRegression()
et = ExtraTreesRegressor(bootstrap=True, n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=10)

# Step 3: Voting Regressor
step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb', xgb), ('et', et)] , weights=[5,1,1,1])

# Step 4: Create Pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the model and make predictions
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Voting Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))


# print("Algorithm: Stacking Regressor")
# Step 1: Preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Step 2: Define base regressors and final estimator
estimators = [
    ('rf', RandomForestRegressor(n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)),
    ('gbdt', GradientBoostingRegressor(n_estimators=100, max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25, max_depth=5, learning_rate=0.3))
]

# Step 3: Stacking Regressor
step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

# Step 4: Create Pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the model and make predictions
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Output
print("***********************************")
print("Algorithm: Stacking Regressor")
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
'''

'''
# Exporting the Model
pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))
'''

print(df.info())


