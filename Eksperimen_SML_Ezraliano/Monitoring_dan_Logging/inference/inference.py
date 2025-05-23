from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import time

app = Flask(__name__)
model = mlflow.sklearn.load_model("./best_logistic_regression_model")

# Metrik latency
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    latency = (time.time() - request.start_time) * 1000  # ms
    print(f"Latency: {latency:.2f} ms")
    return response

# Endpoint prediksi
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    prediction = model.predict(df).tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)