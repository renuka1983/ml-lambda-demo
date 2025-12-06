# src/model_utils.py
import json
import math
from pathlib import Path

PARAMS_PATH = Path(__file__).parent / "model_params.json"

def load_params():
    with open(PARAMS_PATH) as f:
        return json.load(f)

def predict_label(x_value: float) -> dict:
    """
    Returns a dict with probability and label: 'low' or 'high'
    """
    params = load_params()
    coef = params["coef"]
    intercept = params["intercept"]

    # logistic regression: p = 1 / (1 + exp(-(coef * x + intercept)))
    z = coef * x_value + intercept
    p_high = 1.0 / (1.0 + math.exp(-z))

    label = "high" if p_high >= 0.5 else "low"
    return {"prob_high": p_high, "label": label}
