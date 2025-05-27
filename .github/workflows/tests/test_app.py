import pytest
import pickle
import numpy as np
from flask import Flask
from app.main import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Test if the home page renders successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"<form" in response.data  # Assumes form exists in template

def test_predict_valid_input(client):
    """Test prediction with valid form input."""
    response = client.post('/predict', data={
        'year': 2015,
        'present_price': 6.5,
        'kms_driven': 50000,
        'fuel_type': 'Petrol',
        'seller_type': 'Dealer',
        'transmission': 'Manual',
        'owner': 0
    })
    assert response.status_code == 200
    assert b"predicted selling price" in response.data.lower()

def test_predict_invalid_fuel_type(client):
    """Test fallback encoding on unknown fuel type."""
    response = client.post('/predict', data={
        'year': 2018,
        'present_price': 4.0,
        'kms_driven': 30000,
        'fuel_type': 'UnknownFuel',  # Should default to 0
        'seller_type': 'Individual',
        'transmission': 'Manual',
        'owner': 0
    })
    assert response.status_code == 200
    assert b"predicted selling price" in response.data.lower()

def test_missing_form_field(client):
    """Test form submission with a missing field."""
    response = client.post('/predict', data={
        # 'year' is missing
        'present_price': 6.5,
        'kms_driven': 50000,
        'fuel_type': 'Petrol',
        'seller_type': 'Dealer',
        'transmission': 'Manual',
        'owner': 0
    })
    # Flask will throw a 400 or 500 on bad form parse
    assert response.status_code in [400, 500]
