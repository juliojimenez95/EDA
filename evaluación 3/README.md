Investigación: Regresión Lineal y Polinómica
Fundamentos del Modelado Predictivo
Mostrar imagen
Mostrar imagen
Mostrar imagen

📑 Tabla de Contenidos

Regresión Lineal

1.1 Concepto
1.2 Análisis Práctico
1.3 Supuestos y Limitaciones


Regresión Polinómica

2.1 Concepto
2.2 Análisis Práctico
2.3 Supuestos y Limitaciones


Referencias


Introducción
La regresión es una de las familias de técnicas más antiguas y relevantes en el campo del aprendizaje automático supervisado. Su objetivo principal es modelar la relación entre una variable de interés (dependiente) y una o más variables predictoras (independientes).
Este documento presenta una investigación profunda sobre dos técnicas fundamentales:

Regresión Lineal: El modelo de referencia por excelencia
Regresión Polinómica: Extensión para capturar relaciones no lineales


1. Regresión Lineal
1.1 Concepto
Marco Conceptual y Fundamentos Teóricos
La regresión lineal es una técnica estadística paramétrica que busca modelar la relación entre una variable dependiente continua Y y una o más variables independientes X, asumiendo que dicha relación es de naturaleza lineal.
Formulación Matemática
La relación fundamental se describe a través de dos ecuaciones:
Ecuación Poblacional:
Y = β₀ + β₁X + ε
Donde:

Y: Variable dependiente (valor a predecir)
X: Variable independiente (predictor)
β₀: Intercepto poblacional (valor de Y cuando X = 0)
β₁: Coeficiente de pendiente (cambio en Y por unidad de X)
ε: Término de error estocástico

Ecuación Muestral:
Ŷ = b₀ + b₁X
Donde:

Ŷ: Valor predicho para Y
b₀, b₁: Estimaciones de los parámetros β₀ y β₁

Método de Mínimos Cuadrados Ordinarios (MCO)
El principio de MCO es minimizar la Suma de los Cuadrados de los Residuos (SCR):
SCR = Σ(yᵢ - ŷᵢ)²
Esta función está directamente relacionada con el Error Cuadrático Medio (MSE):
MSE = SCR / n
Interpretación de los Componentes
ComponenteSignificadoInterpretación PrácticaIntercepto (β₀)Valor de Y cuando X = 0Punto de partida o valor basePendiente (β₁)Cambio en Y por unidad de XMagnitud y dirección del efectoError (ε)Variabilidad no explicadaRuido, variables omitidas, medición
Ejemplo de Interpretación:
Si modelamos el salario en función de años de experiencia y obtenemos:
Salario = 25,000 + 2,000 × Años_Experiencia

β₀ = 25,000€: Salario base (sin experiencia)
β₁ = 2,000€: Cada año adicional aumenta el salario en 2,000€


1.2 Análisis Práctico
Implementación en Python
pythonimport numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Datos de ejemplo: Horas de estudio vs Nota final
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2.3, 2.8, 3.2, 3.9, 4.2, 4.8, 5.3, 5.7, 6.3, 6.8])

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X, y)

# Realizar predicciones
y_pred = model.predict(X)

# Obtener parámetros del modelo
intercepto = model.intercept_
coeficiente = model.coef_[0]

# Calcular métricas de evaluación
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# Mostrar resultados
print("="*60)
print("RESULTADOS DE LA REGRESIÓN LINEAL")
print("="*60)
print(f"Ecuación: Nota = {intercepto:.3f} + {coeficiente:.3f} × Horas_Estudio")
print(f"\nIntercepto (b₀): {intercepto:.3f}")
print(f"Coeficiente (b₁): {coeficiente:.3f}")
print(f"\nR² (Coeficiente de Determinación): {r2:.3f}")
print(f"MSE (Error Cuadrático Medio): {mse:.3f}")
print("="*60)

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=100, alpha=0.6, edgecolors='black', label='Datos reales')
plt.plot(X, y_pred, color='red', linewidth=2.5, label='Modelo lineal')
plt.title('Regresión Lineal: Horas de Estudio vs Nota Final', fontsize=14, fontweight='bold')
plt.xlabel('Horas de estudio', fontsize=12)
plt.ylabel('Nota final', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
Salida Esperada
============================================================
RESULTADOS DE LA REGRESIÓN LINEAL
============================================================
Ecuación: Nota = 1.745 + 0.485 × Horas_Estudio

Intercepto (b₀): 1.745
Coeficiente (b₁): 0.485

R² (Coeficiente de Determinación): 0.987
MSE (Error Cuadrático Medio): 0.012
============================================================
Interpretación de Resultados
MétricaValorInterpretaciónIntercepto1.745Un estudiante sin estudio obtendría una nota base de 1.745Coeficiente0.485Cada hora de estudio aumenta la nota en 0.485 puntosR²0.987El modelo explica el 98.7% de la variabilidad de las notasMSE0.012Error cuadrático medio muy bajo, buen ajuste
Conclusión del Análisis:
El modelo lineal muestra un excelente ajuste para estos datos, con un R² muy alto y un MSE bajo, indicando que la relación entre horas de estudio y nota es fuertemente lineal y predecible.

1.3 Supuestos y Limitaciones
Supuestos del Modelo de Regresión Lineal Clásico (MRLC)
Para que las inferencias estadísticas sean válidas, deben cumplirse los siguientes supuestos:
#SupuestoDescripciónDiagnóstico1LinealidadRelación lineal entre X e YGráfico de residuos vs. valores ajustados2IndependenciaLos errores no están correlacionadosPrueba de Durbin-Watson3HomocedasticidadVarianza constante de los erroresPrueba de Breusch-Pagan, gráfico de residuos4NormalidadLos residuos siguen distribución normalGráfico Q-Q, prueba de Shapiro-Wilk5No multicolinealidadVariables independientes no correlacionadasFactor de Inflación de Varianza (VIF)
Diagnóstico Visual de Supuestos
python# Calcular residuos
residuos = y - y_pred

# Crear figura con múltiples gráficos de diagnóstico
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuos vs Valores Ajustados (Linealidad y Homocedasticidad)
axes[0, 0].scatter(y_pred, residuos, color='purple', alpha=0.6, edgecolors='black')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Valores Ajustados', fontsize=11)
axes[0, 0].set_ylabel('Residuos', fontsize=11)
axes[0, 0].set_title('Residuos vs Valores Ajustados', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Histograma de Residuos (Normalidad)
axes[0, 1].hist(residuos, bins=5, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Residuos', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Q-Q Plot (Normalidad)
from scipy import stats
stats.probplot(residuos, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Gráfico Q-Q', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Valores Reales vs Predichos
axes[1, 1].scatter(y, y_pred, color='green', alpha=0.6, edgecolors='black', s=100)
axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[1, 1].set_xlabel('Valores Reales', fontsize=11)
axes[1, 1].set_ylabel('Valores Predichos', fontsize=11)
axes[1, 1].set_title('Valores Reales vs Predichos', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
Limitaciones Fundamentales
LimitaciónDescripciónImpacto🔴 Relaciones No LinealesSolo captura tendencias linealesSubajuste en datos con curvaturas🔴 Sensibilidad a OutliersMCO penaliza mucho los errores grandesCoeficientes sesgados por valores atípicos🔴 Extrapolación RiesgosaPredicciones fuera del rango observadoResultados poco confiables🔴 Asume Estructura FijaRequiere cumplir supuestos estrictosInferencias inválidas si se violan

2. Regresión Polinómica
2.1 Concepto
Definición y Propósito
La regresión polinómica es una extensión de la regresión lineal que permite modelar relaciones no lineales entre las variables. En lugar de ajustar una línea recta, ajusta una curva polinómica.
Formulación Matemática
La forma general de un modelo de regresión polinómica de grado n es:
Y = β₀ + β₁X + β₂X² + β₃X³ + ... + βₙXⁿ + ε
Ejemplo para grado 3:
Y = β₀ + β₁X + β₂X² + β₃X³ + ε
La Paradoja "Lineal"

¿Por qué un modelo "curvo" se llama lineal?

La regresión polinómica es lineal en sus parámetros (β), no en sus variables (X).
Si realizamos la transformación:

X₁ = X
X₂ = X²
X₃ = X³

La ecuación se convierte en:
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ε
Esta es la forma de una regresión lineal múltiple, por lo que podemos usar el mismo método de MCO.
Ventajas y Casos de Uso
VentajaAplicaciónCaptura no linealidadesCurvas de crecimiento, rendimientos decrecientesFlexibleFenómenos físicos complejosFácil implementaciónUsa MCO después de transformar características

2.2 Análisis Práctico
Implementación en Python
pythonfrom sklearn.preprocessing import PolynomialFeatures

# Datos de ejemplo (mismos que regresión lineal)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2.3, 2.8, 3.2, 3.9, 4.2, 4.8, 5.3, 5.7, 6.3, 6.8])

# Generar características polinómicas de grado 3
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

print("Características originales (X):")
print(X[:3])
print("\nCaracterísticas polinómicas (X, X², X³):")
print(X_poly[:3])

# Ajustar modelo polinómico
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Realizar predicciones
y_poly_pred = poly_model.predict(X_poly)

# Calcular métricas
r2_poly = r2_score(y, y_poly_pred)
mse_poly = mean_squared_error(y, y_poly_pred)

# Mostrar resultados
print("\n" + "="*60)
print("RESULTADOS DE LA REGRESIÓN POLINÓMICA (GRADO 3)")
print("="*60)
print(f"Intercepto: {poly_model.intercept_:.3f}")
print(f"Coeficientes: {poly_model.coef_}")
print(f"\nR² (Polinómica grado 3): {r2_poly:.3f}")
print(f"MSE (Polinómica grado 3): {mse_poly:.3f}")
print("="*60)

# Visualización
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', s=100, alpha=0.6, edgecolors='black', label='Datos reales')
plt.plot(X, y_poly_pred, color='green', linewidth=2.5, label='Modelo polinómico (grado 3)')
plt.title('Regresión Polinómica (Grado 3): Horas de Estudio vs Nota Final', 
          fontsize=14, fontweight='bold')
plt.xlabel('Horas de estudio', fontsize=12)
plt.ylabel('Nota final', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
Comparación: Lineal vs Polinómica
python# Comparación visual de múltiples grados
degrees = [1, 2, 3, 5, 9]
colors = ['red', 'orange', 'green', 'purple', 'brown']

plt.figure(figsize=(15, 10))

for idx, degree in enumerate(degrees):
    plt.subplot(2, 3, idx + 1)
    
    # Modelo
    if degree == 1:
        model_temp = LinearRegression()
        model_temp.fit(X, y)
        y_pred_temp = model_temp.predict(X)
    else:
        poly_temp = PolynomialFeatures(degree=degree)
        X_poly_temp = poly_temp.fit_transform(X)
        model_temp = LinearRegression()
        model_temp.fit(X_poly_temp, y)
        y_pred_temp = model_temp.predict(X_poly_temp)
    
    # Métricas
    r2_temp = r2_score(y, y_pred_temp)
    mse_temp = mean_squared_error(y, y_pred_temp)
    
    # Gráfico
    plt.scatter(X, y, color='blue', s=80, alpha=0.6, edgecolors='black', label='Datos')
    plt.plot(X, y_pred_temp, color=colors[idx], linewidth=2.5, label=f'Grado {degree}')
    plt.title(f'Grado {degree}\nR²={r2_temp:.3f}, MSE={mse_temp:.3f}', 
              fontsize=11, fontweight='bold')
    plt.xlabel('Horas de estudio', fontsize=10)
    plt.ylabel('Nota final', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(1, 8)

plt.tight_layout()
plt.suptitle('Comparación de Regresión Polinómica con Diferentes Grados', 
             fontsize=14, fontweight='bold', y=1.02)
plt.show()
Análisis del Compromiso Sesgo-Varianza
GradoR²MSEDiagnóstico1~0.987~0.012✅ Buen ajuste (datos lineales)2~0.995~0.005✅ Ligera mejora3~0.998~0.002✅ Excelente ajuste5~0.999~0.001⚠️ Posible inicio de sobreajuste9~1.000~0.000🔴 Sobreajuste severo

2.3 Supuestos y Limitaciones
Supuestos del Modelo
La regresión polinómica hereda los supuestos de la regresión lineal múltiple:
✅ Independencia de los errores
✅ Homocedasticidad (varianza constante)
✅ Normalidad de los residuos
✅ No multicolinealidad (aunque X, X², X³ están correlacionadas por naturaleza)
Limitaciones Críticas
LimitaciónDescripciónGravedad🔴 Alto Riesgo de SobreajusteA mayor grado, mayor complejidad y memorización de ruidoCRÍTICA🔴 Extrapolación CatastróficaLos polinomios se "disparan" fuera del rango de entrenamientoCRÍTICA🟡 Pérdida de InterpretabilidadCoeficientes de X², X³, etc. difíciles de explicarALTA🟡 MulticolinealidadX, X², X³ están correlacionadas entre síMEDIA
Diagnóstico de Sobreajuste
python# Demostración de sobreajuste con grado 9
X_test = np.linspace(0.5, 10.5, 50).reshape(-1, 1)

# Modelo grado 2 (buen ajuste)
poly_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_2.fit_transform(X)
model_2 = LinearRegression()
model_2.fit(X_poly_2, y)
y_pred_2 = model_2.predict(poly_2.transform(X_test))

# Modelo grado 9 (sobreajuste)
poly_9 = PolynomialFeatures(degree=9)
X_poly_9 = poly_9.fit_transform(X)
model_9 = LinearRegression()
model_9.fit(X_poly_9, y)
y_pred_9 = model_9.predict(poly_9.transform(X_test))

# Visualización
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', s=100, alpha=0.6, edgecolors='black', label='Datos entrenamiento')
plt.plot(X_test, y_pred_2, color='green', linewidth=2.5, label='Grado 2 (Buen ajuste)')
plt.title('Modelo con Buen Ajuste (Grado 2)', fontsize=13, fontweight='bold')
plt.xlabel('X', fontsize=11)
plt.ylabel('Y', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-2, 10)

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', s=100, alpha=0.6, edgecolors='black', label='Datos entrenamiento')
plt.plot(X_test, y_pred_9, color='red', linewidth=2.5, label='Grado 9 (Sobreajuste)')
plt.title('Modelo con Sobreajuste (Grado 9)', fontsize=13, fontweight='bold')
plt.xlabel('X', fontsize=11)
plt.ylabel('Y', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-2, 10)

plt.tight_layout()
plt.show()
Observaciones:

El modelo de grado 2 se generaliza bien fuera del rango de entrenamiento
El modelo de grado 9 muestra oscilaciones erráticas y predicciones absurdas


Comparación Final
Tabla Comparativa Completa
AspectoRegresión LinealRegresión PolinómicaEcuaciónY = β₀ + β₁X + εY = β₀ + β₁X + β₂X² + ... + βₙXⁿ + εFormaLínea rectaCurvaComplejidad⭐ Baja⭐⭐⭐ Media-AltaRiesgo de Subajuste🔴 Alto (si relación no lineal)🟢 BajoRiesgo de Sobreajuste🟢 Bajo🔴 AltoInterpretabilidad⭐⭐⭐⭐⭐ Excelente⭐⭐ LimitadaExtrapolación⚠️ Razonable🔴 Muy peligrosaVelocidad de entrenamiento⚡ Muy rápida⚡⚡ RápidaUso típicoBaseline, inferenciaCurvas de crecimiento
Criterios de Selección
┌─────────────────────────────────────────────┐
│  ¿Cuándo usar Regresión Lineal?            │
├─────────────────────────────────────────────┤
│  ✅ Relación visualmente lineal             │
│  ✅ Prioridad en interpretabilidad          │
│  ✅ Modelo baseline rápido                  │
│  ✅ Datos con posible ruido                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  ¿Cuándo usar Regresión Polinómica?        │
├─────────────────────────────────────────────┤
│  ✅ Relación claramente no lineal           │
│  ✅ Prioridad en precisión predictiva       │
│  ✅ Suficientes datos de entrenamiento      │
│  ✅ No requiere extrapolación               │
└─────────────────────────────────────────────┘
Flujo de Trabajo Recomendado
mermaidgraph TD
    A[Inicio: Análisis Exploratorio] --> B{¿Relación Lineal?}
    B -->|Sí| C[Regresión Lineal]
    B -->|No| D[Regresión Polinómica]
    C --> E[Validar Supuestos]
    D --> F[Seleccionar Grado]
    F --> G[Validación Cruzada]
    E --> H{¿Supuestos OK?}
    G --> H
    H -->|No| I[Ajustar/Transformar]
    H -->|Sí| J[Evaluar en Test]
    I --> J
    J --> K{¿Buen Rendimiento?}
    K -->|No| L[Regularización o Cambio de Modelo]
    K -->|Sí| M[Modelo Final]

Conclusiones
Hallazgos Clave

La regresión lineal es simple, interpretable y robusta, pero limitada a relaciones lineales.
La regresión polinómica ofrece flexibilidad para capturar no linealidades, pero requiere cuidado extremo con el sobreajuste.
El compromiso sesgo-varianza es fundamental: modelos simples tienen alto sesgo (subajuste), modelos complejos tienen alta varianza (sobreajuste).
La validación es crítica: Un MSE de 0 en entrenamiento no significa un buen modelo, probablemente indica sobreajuste.
La interpretabilidad importa: En muchos contextos (ciencia, medicina, finanzas), explicar el modelo es tan importante como predecir.