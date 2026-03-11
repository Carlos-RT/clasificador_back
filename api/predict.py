import json
import numpy as np
import joblib
import os

# cargar modelo y scaler
modelo = joblib.load(os.path.join(os.path.dirname(__file__), "../modelo_iris.pkl"))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), "../scaler.pkl"))

clases = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

def handler(request):

    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Método no permitido"})
        }

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

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "prediccion": resultado
        })
    }