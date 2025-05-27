import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test if home page loads successfully"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'form' in response.data.lower()  # crude check for form presence

def test_prediction(client, monkeypatch):
    """Test prediction route with mock form data"""

    # Monkeypatch the model and scaler
    class MockModel:
        def predict(self, X):
            return [2.5]

    class MockScaler:
        def transform(self, X):
            return X

    monkeypatch.setattr("app.model", MockModel())
    monkeypatch.setattr("app.scaler", MockScaler())

    response = client.post('/predict', data={
        'year': 2015,
        'present_price': 5.5,
        'kms_driven': 40000,
        'fuel_type': 'Petrol',
        'seller_type': 'Dealer',
        'transmission': 'Manual',
        'owner': 0
    })

    assert response.status_code == 200
    assert b"The predicted selling price" in response.data
