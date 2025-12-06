# tests/test_model_utils.py
from src.model_utils import predict_label

def test_predict_low():
    result = predict_label(3.0)
    assert result["label"] == "low"

def test_predict_high():
    result = predict_label(15.0)
    assert result["label"] == "high"
