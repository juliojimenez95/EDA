# Regresión Lineal y Polinómica

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-latest-013243.svg)

Investigación sobre los fundamentos del modelado predictivo utilizando técnicas de regresión lineal y polinómica.

---

## 📑 Tabla de Contenidos

- [1. Regresión Lineal](#1-regresión-lineal)
  - [1.1 Concepto General](#11-concepto-general)
  - [1.2 Análisis Práctico](#12-análisis-práctico)
  - [1.3 Supuestos y Limitaciones](#13-supuestos-y-limitaciones)
- [2. Regresión Polinómica](#2-regresión-polinómica)
  - [2.1 Concepto General](#21-concepto-general)
  - [2.2 Análisis Práctico](#22-análisis-práctico)
  - [2.3 Supuestos y Limitaciones](#23-supuestos-y-limitaciones)
- [Comparación](#comparación)
- [Instalación](#instalación)
- [Referencias](#referencias)

---

## 1. Regresión Lineal

### 1.1 Concepto General

La **regresión lineal** es un método estadístico que modela la relación entre una variable dependiente **Y** y una o más variables independientes **X₁, X₂, ..., Xₚ** suponiendo una relación lineal.

**Modelo general:**
Yᵢ = β₀ + β₁X₁ᵢ + β₂X₂ᵢ + ... + βₚXₚᵢ + εᵢ

**Donde:**
- **β₀**: Intercepto o término independiente
- **βⱼ**: Pendiente o coeficiente de regresión
- **εᵢ**: Error aleatorio del modelo (media cero, varianza constante σ²)

**Objetivo:** Estimar los parámetros **β** minimizando la suma de errores cuadráticos (Mínimos Cuadrados Ordinarios, MCO):
min β: Σ(Yᵢ - Ŷᵢ)² = min β: (Y - Xβ)ᵀ(Y - Xβ)

**Solución matricial:**
β̂ = (XᵀX)⁻¹XᵀY

donde **X** es la matriz de diseño que incluye una columna de unos para el intercepto.

---

### 1.2 Análisis Práctico

**Ejemplo:** Predecir la nota de un estudiante en función de las horas de estudio.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Datos de ejemplo
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2.3, 2.8, 3.2, 3.9, 4.2, 4.8, 5.3, 5.7, 6.3, 6.8])

# Crear y entrenar modelo
model = LinearRegression()
model.fit(X, y)

# Predicciones
y_pred = model.predict(X)

# Métricas
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"Ecuación: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")
print(f"R²: {r2:.3f}, MSE: {mse:.3f}")

# Visualización
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred, color='red', label='Recta ajustada')
plt.title('Regresión Lineal Simple')
plt.xlabel('Horas de estudio')
plt.ylabel('Nota final')
plt.legend()
plt.show()
Interpretación:
El modelo ajusta una línea recta a los datos. Un R² alto indica una buena proporción de varianza explicada por las horas de estudio.

1.3 Supuestos y Limitaciones
Supuestos Fundamentales
SupuestoDescripciónLinealidadLa relación entre X e Y es linealIndependenciaLos residuos no están correlacionadosHomocedasticidadLa varianza de los errores es constanteNormalidadLos residuos siguen una distribución normalNo multicolinealidadVariables independientes no altamente correlacionadas
Limitaciones

No modela relaciones no lineales
Sensible a valores atípicos (outliers)
Depende del cumplimiento estricto de supuestos para inferencias válidas


2. Regresión Polinómica
2.1 Concepto General
La regresión polinómica extiende el modelo lineal al incluir potencias de las variables independientes:
Yᵢ = β₀ + β₁Xᵢ + β₂Xᵢ² + ... + βₐXᵢᵈ + εᵢ
Características clave:

Aunque el modelo es no lineal respecto a X, sigue siendo lineal en los parámetros β
Permite capturar relaciones curvilíneas entre variables
Se adapta mejor a tendencias no lineales

Matriz de diseño:
X = [ 1  X₁  X₁²  ...  X₁ᵈ ]
    [ 1  X₂  X₂²  ...  X₂ᵈ ]
    [ ⋮   ⋮   ⋮   ⋱    ⋮  ]
    [ 1  Xₙ  Xₙ²  ...  Xₙᵈ ]

2.2 Análisis Práctico
pythonfrom sklearn.preprocessing import PolynomialFeatures

# Generar características polinómicas de grado 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Ajustar modelo polinómico
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

y_poly_pred = poly_model.predict(X_poly)

# Métricas
r2_poly = r2_score(y, y_poly_pred)
mse_poly = mean_squared_error(y, y_poly_pred)

print(f"R² (Polinómica grado 3): {r2_poly:.3f}, MSE: {mse_poly:.3f}")

# Visualización
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_poly_pred, color='green', label='Modelo polinómico (grado 3)')
plt.title('Regresión Polinómica')
plt.xlabel('Horas de estudio')
plt.ylabel('Nota final')
plt.legend()
plt.show()
Interpretación:

A mayor grado del polinomio, mejor ajuste a los datos (menor error de entrenamiento)
Grado muy alto puede causar Sobreajuste (overfitting) y pérdida de capacidad de generalización


2.3 Supuestos y Limitaciones
Supuestos

Los errores siguen siendo independientes, homocedásticos y normales
El modelo sigue siendo lineal en los coeficientes β

Limitaciones
LimitaciónImpactoAlto riesgo de sobreajusteCon grados grandes, el modelo memoriza ruidoMala extrapolaciónPredicciones inestables fuera del rango de X observadoDificultad de interpretaciónCoeficientes complejos al aumentar el grado

Comparación
CaracterísticaRegresión LinealRegresión PolinómicaTipo de relaciónLinealNo lineal (curvilínea)Complejidad del modeloBajaMedia–AltaInterpretabilidadAltaMedia o bajaRiesgo de sobreajusteBajoAltoGeneralizaciónBuenaVariableAplicacionesTendencias lineales, pronósticos simplesModelos con curvaturas, fenómenos físicos/biológicos
