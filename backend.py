# Importing necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import joblib

# Loading the Dataset

data = Path("dataset/sales_data.csv")
df = pd.read_csv(data)

# Performing EDA Operations

print("\n No of Rows and Columns:")
print(df.shape)                         # Displaying no of Rows and Columns

print("\n Structure of Dataset:")
print(df.info)                          # Displaying the Structure of Dataset

print("\n Statistical Summary of Dataset:")
print(df.describe)                      # Displaying the Statistical Summary

print("\n No of Missing Values:")
print(df.isnull().sum)                  # Count no of Missing Values

print("\n No of Duplicate Values:")
print(df.duplicated().sum)              # Count no of Duplicated Values

# Split and Train data

df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y") # Convert order_date to proper datetime format

df = df.groupby("order_date")["sales"].sum().reset_index()  # Calculate overall daily sales

df["day"] = df["order_date"].map(pd.Timestamp.toordinal)   # Convert dates into numeric values from string

x = df[["day"]]
y = df["sales"]

# Training and Testing

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size= 0.2)

# Training Linear Regression Model

model = LinearRegression()
model.fit(x_train,y_train)

# Testing and Evaluating the Model

y_test_pred = model.predict((x_test))

# Calculate Evaluation Metrics

mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rs = r2_score(y_test, y_test_pred)

# Display the Evaluation metrics

print(f"Mean_Absolute_Error:{mae:.2f}")
print(f"Mean_Squared_Error:{mse:.2f}")
print(f"r2_Score:{rs:.2f}")

'''
# Forescating Future Values

n = int(input("How many days do you want to forecast? "))

last_day = df["day"].max()              # Returns the last day

future_days = pd.DataFrame({
    "day": range(last_day + 1, last_day + n + 1)    # Creating forecast range according to input
})

predictions = model.predict(future_days)

for i, sales in enumerate(predictions, 1):
    print(f"Day {i}: {sales:.2f}")

# Plotting Graph

plt.plot(predictions, marker="o")
plt.xlabel("Future Days")
plt.ylabel("Predicted Sales")
plt.title("Future Sales Forecast")
plt.grid(True)
plt.show()
'''
# Dumping the model

joblib.dump(model, "model.joblib")
joblib.dump(df["order_date"].max(), "last_date.joblib")
