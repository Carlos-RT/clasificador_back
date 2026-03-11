from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# cargar modelo
modelo = joblib.load("modelo_iris.pkl")
scaler = joblib.load("scaler.pkl")

clases = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

@app.route("/api/predict", methods=["POST"])
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

# handler requerido por vercel
def handler(request, context):
    return app(request.environ, lambda status, headers: None)