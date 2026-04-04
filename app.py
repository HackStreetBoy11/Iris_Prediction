from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model + scaler
model = load_model("models/model.keras", compile=False)
scaler = joblib.load("models/scaler.pkl")

# Class labels
classes = ['setosa', 'versicolor', 'virginica']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get individual inputs
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Convert to array
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # 🔥 Apply scaling (VERY IMPORTANT)
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