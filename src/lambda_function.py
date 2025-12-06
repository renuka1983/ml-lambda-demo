# src/lambda_function.py
import json
from model_utils import predict_label

def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    Expect API Gateway HTTP event with JSON body: {"x": 12.5}
    """

    try:
        # 1. Parse body
        if "body" in event:
            body = event["body"]
            if isinstance(body, str):
                body = json.loads(body)
        else:
            body = event  # for direct invocation / tests

        x = float(body.get("x"))

        # 2. Run prediction
        result = predict_label(x)

        # 3. Build HTTP response (for API Gateway)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "input": x,
                "prediction": result["label"],
                "prob_high": result["prob_high"]
            })
        }

    except Exception as e:
        # Log the error string (will show in CloudWatch)
        print(f"Error: {e}")

        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    # Quick local test
    test_event = {
        "body": json.dumps({"x": 11})
    }
    print(lambda_handler(test_event, None))
