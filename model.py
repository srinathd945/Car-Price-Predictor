import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load dataset and drop index if needed
df = pd.read_csv("data.csv", encoding="ISO-8859-1")

# Map categorical variables to numeric
df["Fuel_Type"] = df["Fuel_Type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
df["Seller_Type"] = df["Seller_Type"].map({"Dealer": 0, "Individual": 1})
df["Transmission"] = df["Transmission"].map({"Manual": 0, "Automatic": 1})

# Features and target
X = df[["Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"]]
y = df["Selling_Price"]

# Feature scaling (optional but included for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model trained and saved as model.pkl!")
