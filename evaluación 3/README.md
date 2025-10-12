Investigación: Regresión Lineal y Polinómica (Continuación)

3. Análisis Avanzado y Casos de Estudio
3.1 Detección y Manejo de Problemas Comunes
3.1.1 Diagnóstico Completo de Residuos
El análisis de residuos es fundamental para validar la calidad del modelo. A continuación se presenta un análisis exhaustivo:
pythonimport numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import seaborn as sns

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Datos de ejemplo con características más realistas
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2.3, 2.8, 3.2, 3.9, 4.2, 4.8, 5.3, 5.7, 6.3, 6.8])

# Ajustar modelos
model_lin = LinearRegression()
model_lin.fit(X, y)
y_pred_lin = model_lin.predict(X)

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)

# Calcular residuos
residuos_lin = y - y_pred_lin
residuos_poly = y - y_pred_poly

# Crear dashboard de diagnóstico
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# ==================== REGRESIÓN LINEAL ====================
fig.text(0.25, 0.96, 'REGRESIÓN LINEAL - DIAGNÓSTICO', 
         ha='center', fontsize=16, fontweight='bold', color='darkred')

# 1. Residuos vs Valores Ajustados (Linealidad y Homocedasticidad)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_pred_lin, residuos_lin, color='red', alpha=0.6, s=100, edgecolors='black')
ax1.axhline(y=0, color='blue', linestyle='--', linewidth=2)
ax1.set_xlabel('Valores Ajustados', fontsize=11, fontweight='bold')
ax1.set_ylabel('Residuos', fontsize=11, fontweight='bold')
ax1.set_title('Residuos vs Valores Ajustados', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Histograma de Residuos (Normalidad)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(residuos_lin, bins=6, color='coral', edgecolor='black', alpha=0.7, density=True)
# Superponer curva normal
mu, sigma = residuos_lin.mean(), residuos_lin.std()
x_norm = np.linspace(residuos_lin.min(), residuos_lin.max(), 100)
ax2.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'b-', linewidth=2, label='Normal teórica')
ax2.set_xlabel('Residuos', fontsize=11, fontweight='bold')
ax2.set_ylabel('Densidad', fontsize=11, fontweight='bold')
ax2.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Q-Q Plot (Normalidad)
ax3 = fig.add_subplot(gs[0, 2])
stats.probplot(residuos_lin, dist="norm", plot=ax3)
ax3.set_title('Gráfico Q-Q (Normalidad)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Residuos Estandarizados
ax4 = fig.add_subplot(gs[1, 0])
residuos_std_lin = (residuos_lin - residuos_lin.mean()) / residuos_lin.std()
ax4.scatter(range(len(residuos_std_lin)), residuos_std_lin, 
           color='red', alpha=0.6, s=100, edgecolors='black')
ax4.axhline(y=0, color='blue', linestyle='--', linewidth=2)
ax4.axhline(y=2, color='orange', linestyle=':', linewidth=1.5, label='±2σ')
ax4.axhline(y=-2, color='orange', linestyle=':', linewidth=1.5)
ax4.set_xlabel('Índice de Observación', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residuos Estandarizados', fontsize=11, fontweight='bold')
ax4.set_title('Residuos Estandarizados (Outliers)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Valores Reales vs Predichos
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y, y_pred_lin, color='red', alpha=0.6, s=100, edgecolors='black')
ax5.plot([y.min(), y.max()], [y.min(), y.max()], 'b--', linewidth=2, label='Predicción perfecta')
ax5.set_xlabel('Valores Reales', fontsize=11, fontweight='bold')
ax5.set_ylabel('Valores Predichos', fontsize=11, fontweight='bold')
ax5.set_title('Reales vs Predichos', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Métricas de Evaluación
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
metrics_text_lin = f"""
MÉTRICAS DEL MODELO LINEAL

R² Score: {r2_score(y, y_pred_lin):.4f}
MSE: {mean_squared_error(y, y_pred_lin):.4f}
RMSE: {np.sqrt(mean_squared_error(y, y_pred_lin)):.4f}
MAE: {mean_absolute_error(y, y_pred_lin):.4f}

Intercepto: {model_lin.intercept_:.4f}
Coeficiente: {model_lin.coef_[0]:.4f}

DIAGNÓSTICO DE RESIDUOS:
Media: {residuos_lin.mean():.6f}
Desv. Estándar: {residuos_lin.std():.4f}
Mínimo: {residuos_lin.min():.4f}
Máximo: {residuos_lin.max():.4f}
"""
ax6.text(0.1, 0.5, metrics_text_lin, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ==================== REGRESIÓN POLINÓMICA ====================
fig.text(0.75, 0.96, 'REGRESIÓN POLINÓMICA (Grado 3) - DIAGNÓSTICO', 
         ha='center', fontsize=16, fontweight='bold', color='darkgreen')

# 7. Residuos vs Valores Ajustados
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(y_pred_poly, residuos_poly, color='green', alpha=0.6, s=100, edgecolors='black')
ax7.axhline(y=0, color='blue', linestyle='--', linewidth=2)
ax7.set_xlabel('Valores Ajustados', fontsize=11, fontweight='bold')
ax7.set_ylabel('Residuos', fontsize=11, fontweight='bold')
ax7.set_title('Residuos vs Valores Ajustados', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Histograma de Residuos
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(residuos_poly, bins=6, color='lightgreen', edgecolor='black', alpha=0.7, density=True)
mu_poly, sigma_poly = residuos_poly.mean(), residuos_poly.std()
x_norm_poly = np.linspace(residuos_poly.min(), residuos_poly.max(), 100)
ax8.plot(x_norm_poly, stats.norm.pdf(x_norm_poly, mu_poly, sigma_poly), 
         'b-', linewidth=2, label='Normal teórica')
ax8.set_xlabel('Residuos', fontsize=11, fontweight='bold')
ax8.set_ylabel('Densidad', fontsize=11, fontweight='bold')
ax8.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 9. Q-Q Plot
ax9 = fig.add_subplot(gs[2, 2])
stats.probplot(residuos_poly, dist="norm", plot=ax9)
ax9.set_title('Gráfico Q-Q (Normalidad)', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)

# 10. Residuos Estandarizados
ax10 = fig.add_subplot(gs[3, 0])
residuos_std_poly = (residuos_poly - residuos_poly.mean()) / residuos_poly.std()
ax10.scatter(range(len(residuos_std_poly)), residuos_std_poly, 
            color='green', alpha=0.6, s=100, edgecolors='black')
ax10.axhline(y=0, color='blue', linestyle='--', linewidth=2)
ax10.axhline(y=2, color='orange', linestyle=':', linewidth=1.5, label='±2σ')
ax10.axhline(y=-2, color='orange', linestyle=':', linewidth=1.5)
ax10.set_xlabel('Índice de Observación', fontsize=11, fontweight='bold')
ax10.set_ylabel('Residuos Estandarizados', fontsize=11, fontweight='bold')
ax10.set_title('Residuos Estandarizados (Outliers)', fontsize=12, fontweight='bold')
ax10.legend()
ax10.grid(True, alpha=0.3)

# 11. Valores Reales vs Predichos
ax11 = fig.add_subplot(gs[3, 1])
ax11.scatter(y, y_pred_poly, color='green', alpha=0.6, s=100, edgecolors='black')
ax11.plot([y.min(), y.max()], [y.min(), y.max()], 'b--', linewidth=2, label='Predicción perfecta')
ax11.set_xlabel('Valores Reales', fontsize=11, fontweight='bold')
ax11.set_ylabel('Valores Predichos', fontsize=11, fontweight='bold')
ax11.set_title('Reales vs Predichos', fontsize=12, fontweight='bold')
ax11.legend()
ax11.grid(True, alpha=0.3)

# 12. Métricas de Evaluación
ax12 = fig.add_subplot(gs[3, 2])
ax12.axis('off')
metrics_text_poly = f"""
MÉTRICAS DEL MODELO POLINÓMICO

R² Score: {r2_score(y, y_pred_poly):.4f}
MSE: {mean_squared_error(y, y_pred_poly):.4f}
RMSE: {np.sqrt(mean_squared_error(y, y_pred_poly)):.4f}
MAE: {mean_absolute_error(y, y_pred_poly):.4f}

Intercepto: {model_poly.intercept_:.4f}
Coeficientes: [{', '.join([f'{c:.4f}' for c in model_poly.coef_])}]

DIAGNÓSTICO DE RESIDUOS:
Media: {residuos_poly.mean():.6f}
Desv. Estándar: {residuos_poly.std():.4f}
Mínimo: {residuos_poly.min():.4f}
Máximo: {residuos_poly.max():.4f}
"""
ax12.text(0.1, 0.5, metrics_text_poly, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.savefig('diagnostico_completo_modelos.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ANÁLISIS COMPARATIVO DE DIAGNÓSTICO")
print("="*80)
print(f"\n{'Criterio':<30} {'Lineal':<20} {'Polinómica (g=3)':<20}")
print("-"*80)
print(f"{'R² Score':<30} {r2_score(y, y_pred_lin):<20.4f} {r2_score(y, y_pred_poly):<20.4f}")
print(f"{'MSE':<30} {mean_squared_error(y, y_pred_lin):<20.4f} {mean_squared_error(y, y_pred_poly):<20.4f}")
print(f"{'Media de Residuos':<30} {residuos_lin.mean():<20.6f} {residuos_poly.mean():<20.6f}")
print(f"{'Desv. Std. Residuos':<30} {residuos_lin.std():<20.4f} {residuos_poly.std():<20.4f}")
print("="*80)
3.1.2 Pruebas Estadísticas de Supuestos
pythonfrom scipy.stats import shapiro, normaltest, anderson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

def pruebas_supuestos(X, y, y_pred, nombre_modelo="Modelo"):
    """
    Realiza pruebas estadísticas para validar los supuestos del modelo
    """
    residuos = y - y_pred
    
    print("\n" + "="*70)
    print(f"PRUEBAS ESTADÍSTICAS DE SUPUESTOS - {nombre_modelo}")
    print("="*70)
    
    # 1. NORMALIDAD DE RESIDUOS
    print("\n1. NORMALIDAD DE RESIDUOS")
    print("-"*70)
    
    # Prueba de Shapiro-Wilk
    stat_shapiro, p_shapiro = shapiro(residuos)
    print(f"   Prueba de Shapiro-Wilk:")
    print(f"   Estadístico: {stat_shapiro:.4f}")
    print(f"   P-valor: {p_shapiro:.4f}")
    if p_shapiro > 0.05:
        print(f"   ✅ No se rechaza H0: Los residuos parecen seguir distribución normal")
    else:
        print(f"   ❌ Se rechaza H0: Los residuos NO siguen distribución normal")
    
    # Prueba de D'Agostino-Pearson
    stat_dagostino, p_dagostino = normaltest(residuos)
    print(f"\n   Prueba de D'Agostino-Pearson:")
    print(f"   Estadístico: {stat_dagostino:.4f}")
    print(f"   P-valor: {p_dagostino:.4f}")
    if p_dagostino > 0.05:
        print(f"   ✅ No se rechaza H0: Los residuos parecen seguir distribución normal")
    else:
        print(f"   ❌ Se rechaza H0: Los residuos NO siguen distribución normal")
    
    # 2. INDEPENDENCIA (Autocorrelación)
    print("\n2. INDEPENDENCIA DE ERRORES")
    print("-"*70)
    
    dw_stat = durbin_watson(residuos)
    print(f"   Prueba de Durbin-Watson:")
    print(f"   Estadístico: {dw_stat:.4f}")
    print(f"   Interpretación:")
    if 1.5 < dw_stat < 2.5:
        print(f"   ✅ No hay evidencia de autocorrelación (ideal: ~2.0)")
    elif dw_stat <= 1.5:
        print(f"   ⚠️ Posible autocorrelación positiva")
    else:
        print(f"   ⚠️ Posible autocorrelación negativa")
    
    # 3. HOMOCEDASTICIDAD
    print("\n3. HOMOCEDASTICIDAD (Varianza Constante)")
    print("-"*70)
    
    try:
        X_with_const = sm.add_constant(X)
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuos, X_with_const)
        print(f"   Prueba de Breusch-Pagan:")
        print(f"   Estadístico LM: {lm:.4f}")
        print(f"   P-valor: {lm_pvalue:.4f}")
        if lm_pvalue > 0.05:
            print(f"   ✅ No se rechaza H0: Homocedasticidad presente")
        else:
            print(f"   ❌ Se rechaza H0: Heterocedasticidad detectada")
    except:
        print(f"   ⚠️ No se pudo realizar la prueba (requiere más datos)")
    
    # 4. RESUMEN DE RESIDUOS
    print("\n4. ESTADÍSTICAS DESCRIPTIVAS DE RESIDUOS")
    print("-"*70)
    print(f"   Media: {residuos.mean():.6f} (ideal: ~0)")
    print(f"   Mediana: {np.median(residuos):.6f}")
    print(f"   Desviación Estándar: {residuos.std():.4f}")
    print(f"   Mínimo: {residuos.min():.4f}")
    print(f"   Máximo: {residuos.max():.4f}")
    print(f"   Rango: {residuos.max() - residuos.min():.4f}")
    
    print("\n" + "="*70)

# Ejecutar pruebas para ambos modelos
pruebas_supuestos(X, y, y_pred_lin, "REGRESIÓN LINEAL")
pruebas_supuestos(X, y, y_pred_poly, "REGRESIÓN POLINÓMICA (Grado 3)")

3.2 Validación Cruzada y Selección del Grado Óptimo
La validación cruzada es esencial para evitar el sobreajuste y seleccionar el grado óptimo del polinomio.
pythonfrom sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

def seleccionar_grado_optimo(X, y, max_degree=10, cv=5):
    """
    Selecciona el grado óptimo del polinomio usando validación cruzada
    """
    degrees = range(1, max_degree + 1)
    train_scores = []
    cv_scores_mean = []
    cv_scores_std = []
    
    print("\n" + "="*80)
    print("SELECCIÓN DEL GRADO ÓPTIMO CON VALIDACIÓN CRUZADA")
    print("="*80)
    print(f"\n{'Grado':<10} {'R² Train':<15} {'R² CV (mean)':<15} {'R² CV (std)':<15} {'Estado':<20}")
    print("-"*80)
    
    for degree in degrees:
        # Crear pipeline
        if degree == 1:
            model = LinearRegression()
            model.fit(X, y)
            train_score = model.score(X, y)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        else:
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('model', LinearRegression())
            ])
            pipeline.fit(X, y)
            train_score = pipeline.score(X, y)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
        
        train_scores.append(train_score)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        
        # Diagnóstico
        diff = train_score - cv_scores.mean()
        if diff < 0.05:
            estado = "✅ Buen ajuste"
        elif diff < 0.15:
            estado = "⚠️ Posible sobreajuste"
        else:
            estado = "🔴 Sobreajuste severo"
        
        print(f"{degree:<10} {train_score:<15.4f} {cv_scores.mean():<15.4f} "
              f"{cv_scores.std():<15.4f} {estado:<20}")
    
    # Encontrar grado óptimo
    optimal_degree = degrees[np.argmax(cv_scores_mean)]
    print("-"*80)
    print(f"🏆 GRADO ÓPTIMO: {optimal_degree} (R² CV = {max(cv_scores_mean):.4f})")
    print("="*80)
    
    # Visualización
    plt.figure(figsize=(14, 6))
    
    # Gráfico 1: R² Train vs CV
    plt.subplot(1, 2, 1)
    plt.plot(degrees, train_scores, 'o-', linewidth=2, markersize=8, 
             label='R² Entrenamiento', color='blue')
    plt.plot(degrees, cv_scores_mean, 's-', linewidth=2, markersize=8, 
             label='R² Validación Cruzada', color='red')
    plt.fill_between(degrees, 
                     np.array(cv_scores_mean) - np.array(cv_scores_std),
                     np.array(cv_scores_mean) + np.array(cv_scores_std),
                     alpha=0.2, color='red')
    plt.axvline(optimal_degree, color='green', linestyle='--', linewidth=2, 
                label=f'Grado Óptimo = {optimal_degree}')
    plt.xlabel('Grado del Polinomio', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title('Curva de Validación: Selección de Complejidad', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(degrees)
    
    # Gráfico 2: Brecha Train-CV (Sobreajuste)
    plt.subplot(1, 2, 2)
    gap = np.array(train_scores) - np.array(cv_scores_mean)
    colors = ['green' if g < 0.05 else 'orange' if g < 0.15 else 'red' for g in gap]
    bars = plt.bar(degrees, gap, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(0.05, color='orange', linestyle='--', linewidth=1.5, 
                label='Umbral Alerta (0.05)')
    plt.axhline(0.15, color='red', linestyle='--', linewidth=1.5, 
                label='Umbral Crítico (0.15)')
    plt.xlabel('Grado del Polinomio', fontsize=12, fontweight='bold')
    plt.ylabel('Brecha (R² Train - R² CV)', fontsize=12, fontweight='bold')
    plt.title('Diagnóstico de Sobreajuste', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(degrees)
    
    plt.tight_layout()
    plt.savefig('validacion_cruzada_grado_optimo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_degree, cv_scores_mean, cv_scores_std

# Ejecutar selección de grado óptimo
grado_optimo, cv_means, cv_stds = seleccionar_grado_optimo(X, y, max_degree=10, cv=5)

3.3 Caso de Estudio Completo: Predicción de Precios de Viviendas
Aplicación práctica con un dataset más realista que simula precios de viviendas.
python# Generar dataset sintético más complejo
np.random.seed(123)
n_samples = 100

# Tamaño de la vivienda (metros cuadrados)
tamano = np.random.uniform(50, 300, n_samples)

# Precio con relación no lineal + ruido
# Ley de rendimientos decrecientes: precio por m² disminuye con el tamaño
precio_base = 1500  # precio base por m²
precio = (precio_base * tamano 
          - 0.8 * tamano**2  # efecto decreciente
          + 0.001 * tamano**3  # ligera corrección
          + np.random.normal(0, 15000, n_samples))  # ruido

# Preparar datos
X_vivienda = tamano.reshape(-1, 1)
y_vivienda = precio

# División en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vivienda, y_vivienda, test_size=0.2, random_state=42
)

print("\n" + "="*80)
print("CASO DE ESTUDIO: PREDICCIÓN DE PRECIOS DE VIVIENDAS")
print("="*80)
print(f"\nDatos de entrenamiento: {len(X_train)} viviendas")
print(f"Datos de prueba: {len(X_test)} viviendas")
print(f"Rango de tamaños: {tamano.min():.0f} - {tamano.max():.0f} m²")
print(f"Rango de precios: €{precio.min():,.0f} - €{precio.max():,.0f}")

# Probar diferentes modelos
models_comparison = {}

# Modelo 1: Lineal
model_lin_case = LinearRegression()
model_lin_case.fit(X_train, y_train)
y_pred_lin_train = model_lin_case.predict(X_train)
y_pred_lin_test = model_lin_case.predict(X_test)

models_comparison['Lineal'] = {
    'model': model_lin_case,
    'r2_train': r2_score(y_train, y_pred_lin_train),
    'r2_test': r2_score(y_test, y_pred_lin_test),
    'mse_train': mean_squared_error(y_train, y_pred_lin_train),
    'mse_test': mean_squared_error(y_test, y_pred_lin_test),
    'predictions_test': y_pred_lin_test
}

# Modelos Polinómicos (grados 2, 3, 5)
for degree in [2, 3, 5]:
    poly_temp = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_temp.fit_transform(X_train)
    X_test_poly = poly_temp.transform(X_test)
    
    model_poly_temp = LinearRegression()
    model_poly_temp.fit(X_train_poly, y_train)
    
    y_pred_poly_train = model_poly_temp.predict(X_train_poly)
    y_pred_poly_test = model_poly_temp.predict(X_test_poly)
    
    models_comparison[f'Polinómico (g={degree})'] = {
        'model': model_poly_temp,
        'poly': poly_temp,
        'r2_train': r2_score(y_train, y_pred_poly_train),
        'r2_test': r2_score(y_test, y_pred_poly_test),
        'mse_train': mean_squared_error(y_train, y_pred_poly_train),
        'mse_test': mean_squared_error(y_test, y_pred_poly_test),
        'predictions_test': y_pred_poly_test
    }

# Tabla comparativa
print("\n" + "="*100)
print(f"{'Modelo':<25} {'R² Train':<15} {'R² Test':<15} {'MSE Train':<20} {'MSE Test':<20}")
print("="*100)

for name, metrics in models_comparison.items():
    print(f"{name:<25} {metrics['r2_train']:<15.4f} {metrics['r2_test']:<15.4f} "
          f"{metrics['mse_train']:<20,.0f} {metrics['mse_test']:<20,.0f}")

print("="*100)

# Visualización comparativa
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Caso de Estudio: Predicción de Precios de Viviendas', 
             fontsize=16, fontweight='bold', y=0.995)

# Generar puntos para curvas suaves
X_plot = np.linspace(X_vivienda.min(), X_vivi
