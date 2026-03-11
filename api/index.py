from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)

# permitir CORS correctamente
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

modelo = joblib.load(os.path.join(BASE_DIR, "modelo_iris.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

clases = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


@app.route("/")
def home():
    return "API Iris funcionando"


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():

    # responder preflight
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.json

    sepal_length = float(data["sepal_length"])
    sepal_width = float(data["sepal_width"])
    petal_length = float(data["petal_length"])
    petal_width = float(data["petal_width"])

    entrada = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    entrada = scaler.transform(entrada)

    pred = modelo.predict(entrada)

    clase = np.argmax(pred)

    resultado = clases[clase]

    return jsonify({"prediccion": resultado})