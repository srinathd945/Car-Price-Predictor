import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import os

template_folder = os.getenv("FLASK_TEMPLATE_FOLDER", "templates")
app = Flask(__name__, template_folder=template_folder)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    if app.config.get("TESTING"):
        return "<form></form>"  # simple dummy response
    return render_template("Index.html")

@app.route("/predict", methods=["POST"])
def predict():
    year = int(request.form["year"])
    present_price = float(request.form["present_price"])
    kms_driven = int(request.form["kms_driven"])
    fuel_type = request.form["fuel_type"]
    seller_type = request.form["seller_type"]
    transmission = request.form["transmission"]
    owner = int(request.form["owner"])

    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 0, "Automatic": 1}

    fuel_type = fuel_map.get(fuel_type, 0)
    seller_type = seller_map.get(seller_type, 0)
    transmission = trans_map.get(transmission, 0)

    input_data = pd.DataFrame([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]],
                              columns=["Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    formatted_prediction = f"The predicted selling price (in lacs) is {round(float(prediction[0]), 2)}"

    if app.config.get("TESTING"):
        return formatted_prediction  # simple response for testing

    return render_template("result.html", prediction=formatted_prediction)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')