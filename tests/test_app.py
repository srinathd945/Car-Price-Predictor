import pytest

@pytest.fixture
def client(monkeypatch):
    # Mock model and scaler before importing flaskapp
    class MockModel:
        def predict(self, X):
            return [2.5]

    class MockScaler:
        def transform(self, X):
            return X

    monkeypatch.setattr("flaskapp.model", MockModel())
    monkeypatch.setattr("flaskapp.scaler", MockScaler())

    # Now import flaskapp after mocking
    from flaskapp import app

    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'form' in response.data.lower()

def test_prediction(client):
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