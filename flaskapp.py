import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    year = int(request.form["year"])
    present_price = float(request.form["present_price"])
    kms_driven = int(request.form["kms_driven"])
    fuel_type = request.form["fuel_type"]  # "Petrol", "Diesel", or "CNG"
    seller_type = request.form["seller_type"]  # "Dealer" or "Individual"
    transmission = request.form["transmission"]  # "Manual" or "Automatic"
    owner = int(request.form["owner"])

    # Manual encoding
    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 0, "Automatic": 1}

    fuel_type = fuel_map.get(fuel_type, 0)
    seller_type = seller_map.get(seller_type, 0)
    transmission = trans_map.get(transmission, 0)

    # Create DataFrame for prediction
    input_data = pd.DataFrame([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]],
                              columns=["Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"])

    # Apply scaling
    input_scaled = scaler.transform(input_data)

    # Predict price
    prediction = model.predict(input_scaled)
    formatted_prediction = f"The predicted selling price is ₹{round(float(prediction[0]), 2)}"

    return render_template("result.html", prediction=formatted_prediction)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
