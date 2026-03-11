import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================
# CARGAR DATASET
# ==========================

Data = pd.read_csv("iris.csv")

print("Primeras filas del dataset:")
print(Data.head())

# ==========================
# SEPARAR VARIABLES Y CLASE
# ==========================

X = Data.iloc[:,1:5].values
Y = Data.iloc[:,5].values

print("Shape X:", X.shape)
print("Shape Y:", Y.shape)

# ==========================
# NORMALIZAR DATOS (IMPORTANTE)
# ==========================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================
# CONVERTIR CLASES A NUMÉRICO
# ==========================

le = preprocessing.LabelEncoder()
Y_encoded = le.fit_transform(Y)

# ==========================
# ONE HOT ENCODING
# ==========================

lb = preprocessing.LabelBinarizer()
Y2 = lb.fit_transform(Y_encoded)

# ==========================
# DIVISIÓN 80% / 20%
# ==========================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y2,
    test_size=0.20,
    random_state=42
)

print("Datos entrenamiento:", X_train.shape)
print("Datos prueba:", X_test.shape)

# ==========================
# CREAR RED NEURONAL
# ==========================

clf = MLPClassifier(
    hidden_layer_sizes=(10,5),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=1,
    verbose=True
)

# ==========================
# ENTRENAMIENTO
# ==========================

clf.fit(X_train, Y_train)

print("Entrenamiento finalizado")

# ==========================
# PRUEBA DEL MODELO
# ==========================

Y_pred = clf.predict(X_test)

# Convertir resultados para comparar
Y_test_labels = np.argmax(Y_test, axis=1)
Y_pred_labels = np.argmax(Y_pred, axis=1)

# Accuracy
accuracy = accuracy_score(Y_test_labels, Y_pred_labels)

print("Precisión del modelo:", accuracy)

# Matriz de confusión
cm = confusion_matrix(Y_test_labels, Y_pred_labels)

print("Matriz de confusión:")
print(cm)

# ==========================
# GUARDAR MODELO ENTRENADO
# ==========================

joblib.dump(clf,"modelo_iris.pkl")

# guardar también el scaler
joblib.dump(scaler,"scaler.pkl")

print("Modelo guardado como modelo_iris.pkl")
print("Scaler guardado como scaler.pkl")