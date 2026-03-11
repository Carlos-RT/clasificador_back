from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# ==========================
# CARGAR MODELO Y SCALER
# ==========================

modelo = joblib.load("modelo_iris.pkl")
scaler = joblib.load("scaler.pkl")

clases = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# ==========================
# RUTA DE PREDICCIÓN
# ==========================

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

# ==========================
# INICIAR SERVIDOR
# ==========================

if __name__ == "__main__":
    app.run(debug=True)