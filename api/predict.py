from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# ruta absoluta del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

modelo_path = os.path.join(BASE_DIR, "modelo_iris.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

clases = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


@app.route("/")
def home():
    return "API Iris funcionando"


@app.route("/predict", methods=["POST"])
def predict():

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