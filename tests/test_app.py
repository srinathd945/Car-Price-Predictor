import pytest

@pytest.fixture
def client(monkeypatch):
    # Dummy classes to simulate your model and scaler
    class DummyModel:
        def predict(self, X):
            return [2.5]

    class DummyScaler:
        def transform(self, X):
            return X

    # This list helps us return DummyModel first, then DummyScaler
    dummy_objects = [DummyModel(), DummyScaler()]

    def fake_pickle_load(file):
        return dummy_objects.pop(0)

    # Patch pickle.load before importing flaskapp
    monkeypatch.setattr("pickle.load", fake_pickle_load)

    # Now import flaskapp after patching pickle.load
    import flaskapp
    flaskapp.app.config['TESTING'] = True

    with flaskapp.app.test_client() as client:
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