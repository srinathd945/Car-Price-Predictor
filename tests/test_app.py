import pytest
import io
import pickle

@pytest.fixture
def client(monkeypatch):
    # Dummy classes to simulate model and scaler
    class DummyModel:
        def predict(self, X):
            return [2.5]

    class DummyScaler:
        def transform(self, X):
            return X

    dummy_objects = [DummyModel(), DummyScaler()]

    # Mock pickle.load
    def fake_pickle_load(file):
        return dummy_objects.pop(0)

    monkeypatch.setattr(pickle, "load", fake_pickle_load)

    # Mock open() to return a dummy binary file
    def fake_open(file, mode='rb'):
        return io.BytesIO(b"fake pickle data")

    monkeypatch.setattr("builtins.open", fake_open)

    # Now import flaskapp after patching
    import flaskapp
    flaskapp.app.config["TESTING"] = True

    with flaskapp.app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"form" in response.data.lower()

def test_prediction(client):
    response = client.post("/predict", data={
        "year": 2015,
        "present_price": 5.5,
        "kms_driven": 40000,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": 0
    })
    assert response.status_code == 200
    assert b"predicted selling price" in response.data.lower()