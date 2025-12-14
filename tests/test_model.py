import joblib

def test_model_load():
    model = joblib.load("model.pkl")
    assert model is not None