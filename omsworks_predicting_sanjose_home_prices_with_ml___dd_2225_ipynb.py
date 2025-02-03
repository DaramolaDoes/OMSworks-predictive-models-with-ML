#!/usr/bin/env python
# coding: utf-8

# In[1]:


#San Jose Real Estate (single family), n=24; 2023-2024
#invest now or later (1 month or 6 months)
#linear regression models

#required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# San Jose Single Family 2023-2024 Dataset
data = {
    'Date': pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS'),
    'Price': [940930, 929451, 916338, 907631, 905241, 908244, 916925, 928365, 941508, 953024, 962739, 968395,
              967916, 964526, 963946, 967725, 975414, 982480, 993214, 1002958, 1009910, 1013149, 1017499, 1023242]
}
# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Add numeric index for regression purposes
df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)


# In[3]:


#view dataset
df.info()


# In[4]:


# Check if there are any missing values in the dataset
missing_values = df.isnull().sum()

# Print missing values count per column
print(missing_values)

# Check if there are any missing values in the entire dataset
if missing_values.sum() > 0:
    print("There are missing values in the dataset.")
else:
    print("There are no missing values in the dataset.")
    


# In[5]:


#view first four rows
df.head()


# In[6]:


#view last four rows
df.tail()


# In[7]:


#view complete, past 24 months
df


# In[8]:


# Generate summary statistics
summary = df.describe()

print(summary)


# In[9]:


# 1. Average price for the last 24 months

average_price = df['Price'].mean()
print(f'Average price over the last 24 months: ${average_price:,.2f}')

# 2. Highest price for the last 22 months

highest_price = df['Price'].max()
print(f'Highest price over the last 24 months: ${highest_price:,.2f}')

# 3. Lowest price for the last 22 months

lowest_price = df['Price'].min()
print(f'Lowest price over the last 24 months: ${lowest_price:,.2f}')


# In[9]:


# Generate mean, max, min
print(9.608654e+05, 1.023242e+06, 9.052410e+05)


# In[10]:


# plotting the historical 24 months (2023-2024) San Jose single family housing prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Historical Prices', marker='o', color='black')
plt.title('Historical Price Trends of San Jose Single-Family Homes: 2023-2024')
plt.xlabel('Date')
plt.ylabel('Price in Millions $')


# In[11]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data preparation
df = pd.DataFrame(data)

# Convert Date to ordinal (for simplicity in linear regression)
df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())

# Splitting the data into training and test sets (80/20)
X = df[['Date_Ordinal']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression model training
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions on test data
y_pred = lin_reg.predict(X_test)

# Model coefficients
intercept = lin_reg.intercept_
slope = lin_reg.coef_[0]

# Calculating accuracy metrics
mae_test = mean_absolute_error(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

# Returning results
slope, intercept, mae_test, rmse_test


# In[12]:


#linear regression model with historical San Jose 2023-2024 data points

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm

# Prepare data for linear regression
df['time_numeric'] = (df['Date'] - df['Date'].min()).dt.days  # Convert dates to numeric for regression

# Define X (independent variable: time in days) and y (dependent variable: price)
X = df['time_numeric'].values.reshape(-1, 1)
y = df['Price'].values

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot the linear regression line with the data points
plt.figure(figsize=(10,6))
plt.scatter(df['Date'], df['Price'], color='black', label="Actual Prices")
plt.plot(df['Date'], model.predict(X), color='blue', label="Linear Regression Line")
plt.title("Linear Regression Line with Historical Price Trends of San Jose Single Family Real Estate (2023-2024)")
plt.xlabel("Date")
plt.ylabel("Price in Millions $")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[13]:


# Import necessary libraries for OLS
import statsmodels.api as sm

# Prepare the data
df_real_estate = pd.DataFrame(data)
df_real_estate['Date_Ordinal'] = df_real_estate['Date'].apply(lambda x: x.toordinal())  # Convert dates to ordinal

# Prepare the independent (X) and dependent variable (y)
X = df_real_estate[['Date_Ordinal']]
y = df_real_estate['Price']

# Add a constant for OLS regression (intercept)
X_const = sm.add_constant(X)

# Perform OLS regression
ols_model = sm.OLS(y, X_const).fit()

# Linear regression equation
slope_ols = ols_model.params['Date_Ordinal']
intercept_ols = ols_model.params['const']
equation_ols = f"Price = {slope_ols:.4f} * Date_Ordinal + {intercept_ols:.2f}"

# Predict future prices for the next 12 months
future_dates = pd.date_range(start='2025-01-01', periods=12, freq='MS').to_frame(index=False, name='Date')
future_dates['Date_Ordinal'] = future_dates['Date'].apply(lambda x: x.toordinal())

# Add constant for future predictions
future_dates_const = sm.add_constant(future_dates[['Date_Ordinal']])

# Predict future prices
future_prices_ols = ols_model.predict(future_dates_const)

# Output OLS summary, equation, and future predictions
{
    "ols_summary": ols_model.summary(),
    "equation_ols": equation_ols,
    "future_prices_ols": future_prices_ols
}


# In[14]:


# reimporting libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# San Jose 2023-2024 single family real estate dataset
data = {
    'Date': pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS'),
    'Price': [940930, 929451, 916338, 907631, 905241, 908244, 916925, 928365, 941508, 953024, 962739, 968395,
              967916, 964526, 963946, 967725, 975414, 982480, 993214, 1002958, 1009910, 1013149, 1017499, 1023242]
}
# Co

df = pd.DataFrame(data)
# Creating the DataFrame
df = pd.DataFrame(data)

# Preparing the data for linear regression
df['date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)  # Convert dates to ordinal for regression

# Linear regression model
X = df['date_ordinal'].values.reshape(-1, 1)  # Dates as independent variable
y = df['Price'].values  # Prices as dependent variable

model = LinearRegression()
model.fit(X, y)

# Predicted values for plotting
y_pred = model.predict(X)

# Adding the predicted price for 1/1/2025 to the dataset for visualization
predicted_date = pd.to_datetime('2025-1-01')
predicted_price = model.predict([[predicted_date.toordinal()]])[0]

# Plotting the actual prices, regression line, and predicted value
plt.figure(figsize=(12,8))
plt.scatter(df['Date'], df['Price'], color='black', label='Actual prices')
plt.plot(df['Date'], y_pred, color='blue', label='Linear regression line')

# Highlighting the predicted value for 1/1/2025
plt.scatter(predicted_date, predicted_price, color='green', label=f'Predicted for {predicted_date.strftime("%m/%d/%Y")}', s=100)

# Annotating the predicted value
plt.text(predicted_date, predicted_price, f"${predicted_price:.2f}", fontsize=10, color='green', ha='left')

# Plot settings
plt.title('Linear Regression with Predicted Price for 1/1/25')
plt.xlabel('Date')
plt.ylabel('Price in Millions $')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()


# In[15]:


# reimporting libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# San Jose 2023-2024 single family real estate dataset
data = {
    'Date': pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS'),
    'Price': [940930, 929451, 916338, 907631, 905241, 908244, 916925, 928365, 941508, 953024, 962739, 968395,
              967916, 964526, 963946, 967725, 975414, 982480, 993214, 1002958, 1009910, 1013149, 1017499, 1023242]
}
# Co

df = pd.DataFrame(data)
# Creating the DataFrame
df = pd.DataFrame(data)

# Preparing the data for linear regression
df['date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)  # Convert dates to ordinal for regression

# Linear regression model
X = df['date_ordinal'].values.reshape(-1, 1)  # Dates as independent variable
y = df['Price'].values  # Prices as dependent variable

model = LinearRegression()
model.fit(X, y)

# Predicted values for plotting
y_pred = model.predict(X)

# Adding the predicted price for 3/1/2025 to the dataset for visualization
predicted_date = pd.to_datetime('2025-3-01')
predicted_price = model.predict([[predicted_date.toordinal()]])[0]

# Plotting the actual prices, regression line, and predicted value
plt.figure(figsize=(12,8))
plt.scatter(df['Date'], df['Price'], color='black', label='Actual prices')
plt.plot(df['Date'], y_pred, color='blue', label='Linear regression line')

# Highlighting the predicted value for 3/1/2025
plt.scatter(predicted_date, predicted_price, color='green', label=f'Predicted for {predicted_date.strftime("%m/%d/%Y")}', s=100)

# Annotating the predicted value
plt.text(predicted_date, predicted_price, f"${predicted_price:.2f}", fontsize=10, color='green', ha='right')

# Plot settings
plt.title('Linear Regression with Predicted Price for 3/1/25')
plt.xlabel('Date')
plt.ylabel('Price in Millions $')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()


# In[37]:


# reimporting libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# San Jose 2023-2024 single family real estate dataset
data = {
    'Date': pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS'),
    'Price': [940930, 929451, 916338, 907631, 905241, 908244, 916925, 928365, 941508, 953024, 962739, 968395,
              967916, 964526, 963946, 967725, 975414, 982480, 993214, 1002958, 1009910, 1013149, 1017499, 1023242]
}
# Co

df = pd.DataFrame(data)
# Creating the DataFrame
df = pd.DataFrame(data)

# Preparing the data for linear regression
df['date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)  # Convert dates to ordinal for regression

# Linear regression model
X = df['date_ordinal'].values.reshape(-1, 1)  # Dates as independent variable
y = df['Price'].values  # Prices as dependent variable

model = LinearRegression()
model.fit(X, y)

# Predicted values for plotting
y_pred = model.predict(X)

# Adding the predicted price for 6/1/2025 to the dataset for visualization
predicted_date = pd.to_datetime('2025-6-01')
predicted_price = model.predict([[predicted_date.toordinal()]])[0]

# Plotting the actual prices, regression line, and predicted value
plt.figure(figsize=(12,8))
plt.scatter(df['Date'], df['Price'], color='black', label='Actual prices')
plt.plot(df['Date'], y_pred, color='blue', label='Linear regression line')

# Highlighting the predicted value for 6/1/2025
plt.scatter(predicted_date, predicted_price, color='green', label=f'Predicted for {predicted_date.strftime("%m/%d/%Y")}', s=100)

# Annotating the predicted value
plt.text(predicted_date, predicted_price, f"${predicted_price:.2f}", fontsize=10, color='green', ha='right')

# Plot settings
plt.title('Linear Regression with Predicted Price for 6/1/25')
plt.xlabel('Date')
plt.ylabel('Price in Millions $')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()


# In[16]:


# reimporting libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# San Jose 2023-2024 single family real estate dataset
data = {
    'Date': pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS'),
    'Price': [940930, 929451, 916338, 907631, 905241, 908244, 916925, 928365, 941508, 953024, 962739, 968395,
              967916, 964526, 963946, 967725, 975414, 982480, 993214, 1002958, 1009910, 1013149, 1017499, 1023242]
}
# Co

df = pd.DataFrame(data)
# Creating the DataFrame
df = pd.DataFrame(data)

# Preparing the data for linear regression
df['date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)  # Convert dates to ordinal for regression

# Linear regression model
X = df['date_ordinal'].values.reshape(-1, 1)  # Dates as independent variable
y = df['Price'].values  # Prices as dependent variable

model = LinearRegression()
model.fit(X, y)

# Predicted values for plotting
y_pred = model.predict(X)

# Adding the predicted price for 1/1/2026 to the dataset for visualization
predicted_date = pd.to_datetime('2026-1-01')
predicted_price = model.predict([[predicted_date.toordinal()]])[0]

# Plotting the actual prices, regression line, and predicted value
plt.figure(figsize=(12,8))
plt.scatter(df['Date'], df['Price'], color='black', label='Actual prices')
plt.plot(df['Date'], y_pred, color='blue', label='Linear regression line')

# Highlighting the predicted value for 1/1/2026
plt.scatter(predicted_date, predicted_price, color='green', label=f'Predicted for {predicted_date.strftime("%m/%d/%Y")}', s=100)

# Annotating the predicted value
plt.text(predicted_date, predicted_price, f"${predicted_price:.2f}", fontsize=10, color='green', ha='right')

# Plot settings
plt.title('Linear Regression Model for San Jose Price Prediction: Forcast for January 1, 2026')
plt.xlabel('Date')
plt.ylabel('Price in Millions $')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()


# In[17]:


# 1. Average price for the last 24 months

average_price = df['Price'].mean()
print(f'Average price over the last 24 months: ${average_price:,.2f}')

# 2. Highest price for the last 24 months

highest_price = df['Price'].max()
print(f'Highest price over the last 24 months: ${highest_price:,.2f}')

# 3. Lowest price for the last 24 months

lowest_price = df['Price'].min()
print(f'Lowest price over the last 24 months: ${lowest_price:,.2f}')


# In[19]:


print(1.046615e+06)


# In[18]:


def calculate_projected_savings(purchase_price, predicted_price):
    # Calculate the difference between predicted and purchase price
    price_difference = predicted_price - purchase_price

    # Calculate the percentage increase
    percentage_increase = (price_difference / purchase_price) * 100

    return percentage_increase

# Inputs
purchase_price = 960865  # Purchase price of the home
predicted_price = 1046615 # Predicted price of the home in June 2025

# Calculate the projected savings
projected_savings = calculate_projected_savings(purchase_price, predicted_price)

# Output the result
print(f"The projected percentage saved on the investment is {projected_savings:.2f}%")


# In[19]:


def calculate_projected_savings(purchase_price, predicted_price):
    # Calculate the difference between predicted and purchase price
    dollar_saved = predicted_price - purchase_price

    return dollar_saved

# Inputs
purchase_price = 960865  # Purchase price of the home
predicted_price = 1046615  # Predicted price of the home in June 2025

# Calculate the projected savings
projected_savings = calculate_projected_savings(purchase_price, predicted_price)

# Output the result
print(f"The projected dollar saved on the investment is ${projected_savings:,}")


# In[ ]:




