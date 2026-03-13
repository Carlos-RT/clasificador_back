from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import math
from collections import Counter
import base64
import codecs

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

modelo = joblib.load(os.path.join(BASE_DIR, "modelo_cifrado.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler_cifrado.pkl"))

clases = [
    "Texto plano",
    "Caesar",
    "ROT13",
    "Base64",
    "XOR"
]


# ===============================
# ENTROPÍA
# ===============================

def shannon_entropy(text):

    counter = Counter(text)
    length = len(text)

    entropy = 0

    for count in counter.values():

        p = count / length

        entropy -= p * math.log2(p)

    return entropy


# ===============================
# EXTRAER FEATURES
# ===============================

def extraer_features(text):

    length = len(text)

    if length == 0:
        return [0]*20

    mayus = sum(1 for c in text if c.isupper())
    minus = sum(1 for c in text if c.islower())
    nums = sum(1 for c in text if c.isdigit())
    spaces = sum(1 for c in text if c.isspace())
    especiales = sum(1 for c in text if not c.isalnum() and not c.isspace())

    ascii_vals = [ord(c) for c in text]

    rango_min = min(ascii_vals)
    rango_max = max(ascii_vals)

    media_ascii = np.mean(ascii_vals)
    var_ascii = np.var(ascii_vals)

    counter = Counter(text)

    freq = sorted(counter.values(), reverse=True)

    top1 = freq[0]/length
    top2 = freq[1]/length if len(freq) > 1 else 0

    padding = 1 if text.endswith("=") else 0

    alpha = sum(1 for c in text if c.isalpha())

    unique_chars = len(set(text))

    ratio_unique = unique_chars/length

    entropy = shannon_entropy(text)


    # ===============================
    # FEATURES NUEVAS
    # ===============================

    base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="

    freq_base64 = sum(1 for c in text if c in base64_chars) / length

    hex_chars = "0123456789abcdefABCDEF"

    ratio_hex = sum(1 for c in text if c in hex_chars) / length

    try:
        base64.b64decode(text)
        es_base64 = 1
    except:
        es_base64 = 0


    return [

        length,
        entropy,
        mayus/length,
        minus/length,
        nums/length,
        spaces/length,
        especiales/length,
        rango_min,
        rango_max,
        media_ascii,
        var_ascii,
        top1,
        top2,
        padding,
        alpha/length,
        unique_chars,
        ratio_unique,
        freq_base64,
        ratio_hex,
        es_base64
    ]


# ===============================
# DESCIFRADORES
# ===============================

def rot13_decode(text):

    return codecs.decode(text, 'rot_13')


def caesar_bruteforce(text):

    resultados = []

    for shift in range(26):

        decoded = ""

        for c in text:

            if c.isalpha():

                base = ord('A') if c.isupper() else ord('a')

                decoded += chr((ord(c) - base - shift) % 26 + base)

            else:

                decoded += c

        resultados.append(decoded)

    return resultados[0]


def base64_decode(text):

    try:
        return base64.b64decode(text).decode("utf-8")
    except:
        return "No se pudo decodificar Base64"


# ===============================
# API
# ===============================

@app.route("/", methods=["GET", "POST"])

def predict():

    if request.method == "GET":

        return "API Detector de Cifrados funcionando"


    data = request.json

    texto = data["texto"]


    features = extraer_features(texto)

    entrada = np.array([features])

    entrada = scaler.transform(entrada)

    pred = modelo.predict(entrada)[0]

    tipo = clases[int(pred)]


    # =========================
    # DESCIFRADO
    # =========================

    descifrado = ""

    if tipo == "Texto plano":

        descifrado = texto

    elif tipo == "ROT13":

        descifrado = rot13_decode(texto)

    elif tipo == "Caesar":

        descifrado = caesar_bruteforce(texto)

    elif tipo == "Base64":

        descifrado = base64_decode(texto)

    elif tipo == "XOR":

        descifrado = "No se puede descifrar XOR sin la clave."


    return jsonify({
        "prediccion": tipo,
        "descifrado": descifrado
    })