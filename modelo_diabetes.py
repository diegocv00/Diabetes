

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes_dataset_limpio.csv')

x = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']

# División de datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Escalado
scaler = StandardScaler()
x_train_escalado = scaler.fit_transform(x_train)
x_test_escalado = scaler.transform(x_test)

smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train_escalado, y_train)

# Regresión logistica
log_model = LogisticRegressionCV(cv=5, penalty='l2', max_iter=1000)
log_model.fit(x_train_res, y_train_res)

# Obtener las probabilidades de clase positiva (Diabetes)
y_probs = log_model.predict_proba(x_test_escalado)

# Se extrae solo la columna de probabilidad de Diabetes
prob_diabetes = y_probs[:, 1]

# Se define un nuevo umbral basado en tu gráfica
umbral_personalizado = 0.6

# Se clasificam manualmente
y_pred_personalizado = (prob_diabetes >= umbral_personalizado).astype(int)

joblib.dump(log_model, 'modelo_regresion_logistica_diabetes.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(classification_report(y_test, y_pred_personalizado))

