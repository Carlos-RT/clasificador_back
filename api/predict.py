import json
import numpy as np
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

modelo = joblib.load(os.path.join(BASE_DIR, "modelo_iris.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

clases = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def handler(request):

    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Method not allowed"})
        }

    data = request.get_json()

    sepal_length = float(data["sepal_length"])
    sepal_width = float(data["sepal_width"])
    petal_length = float(data["petal_length"])
    petal_width = float(data["petal_width"])

    entrada = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    entrada = scaler.transform(entrada)

    pred = modelo.predict(entrada)

    clase = int(np.argmax(pred))

    resultado = clases[clase]

    return {
        "statusCode": 200,
        "body": json.dumps({"prediccion": resultado})
    }