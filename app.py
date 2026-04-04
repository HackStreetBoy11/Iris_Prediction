from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Global variables (initially None)
model = None
scaler = None

# Class labels
classes = ['setosa', 'versicolor', 'virginica']


def load_resources():
    global model, scaler
    if model is None or scaler is None:
        from tensorflow.keras.models import load_model

        model_path = os.path.join("models", "model.keras")
        scaler_path = os.path.join("models", "scaler.pkl")

        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        print("✅ Model and scaler loaded successfully.")


# ✅ Warm up on first request (not on user's first prediction)
@app.before_request
def warmup():
    load_resources()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Convert to array
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale data
        data = scaler.transform(data)

        # Prediction
        prediction = model.predict(data)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        result = classes[class_index]

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result} ({confidence:.2f})"
        )

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)