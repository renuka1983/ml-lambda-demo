# src/train_model.py
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    # 1. Create a toy dataset
    # x < 10 -> low, x >= 10 -> high
    X = np.array([[1], [2], [3], [5], [8], [9], [10], [12], [15], [20]], dtype=float)
    y = np.array([0,   0,   0,   0,   0,   0,   1,    1,    1,    1])  # 0=low, 1=high

    # 2. Train a simple logistic regression
    model = LogisticRegression()
    model.fit(X, y)

    # 3. Get parameters (coef and intercept)
    coef = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])

    params = {
        "coef": coef,
        "intercept": intercept
    }

    # 4. Save to JSON file
    with open("src/model_params.json", "w") as f:
        json.dump(params, f)

    print("Saved model parameters to src/model_params.json")
    print(params)

if __name__ == "__main__":
    main()
