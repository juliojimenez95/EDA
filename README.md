# 📊 Taller de Análisis de Datos: Exploración, EDA y Preprocesamiento

Este proyecto es el resultado del segundo evento evaluativo del curso de Análisis de Datos. El objetivo es aplicar los conocimientos adquiridos en las primeras semanas del curso en un proyecto práctico que abarca desde la exploración de diferentes fuentes de datos hasta el preprocesamiento y la reducción de dimensionalidad de un dataset seleccionado.

## 🎯 Objetivos

- **Explorar** múltiples datasets para seleccionar el más adecuado
- **Analizar** profundamente el dataset seleccionado 
- **Preprocesar** y preparar los datos para modelado futuro
- **Aplicar** técnicas de reducción de dimensionalidad

## 📁 Estructura del Proyecto

```
.
├── EDA/
│   ├── Exploracion_base_1.ipynb
│   ├── Exploracion_base_2.ipynb
│   ├── fase_2_EDA_medicamentos.ipynb
│   ├── fase_3_preprocesamiento_y_reduccion.ipynb
│   ├── medicamentos.json
│   └── df_light.parquet
└── README.md
```

### Descripción de Archivos

| Archivo | Descripción |
|---------|-------------|
| `Exploracion_base_1.ipynb` | Exploración del dataset de estratos socioeconómicos de Medellín |
| `Exploracion_base_2.ipynb` | Exploración del dataset de clasificación de imágenes de Intel |
| `fase_2_EDA_medicamentos.ipynb` | Análisis Exploratorio de Datos del dataset de medicamentos |
| `fase_3_preprocesamiento_y_reduccion.ipynb` | Preprocesamiento y reducción de dimensionalidad |
| `medicamentos.json` | Dataset principal utilizado en las fases 2 y 3 |
| `df_light.parquet` | Versión procesada y optimizada del dataset de medicamentos |

## 🔍 Metodología

### Fase 1: Exploración de Bases de Datos

Se exploraron tres datasets diferentes para seleccionar el más apropiado:

#### Datasets Evaluados

| Dataset | Fuente | Tipo | Descripción |
|---------|--------|------|-------------|
| **Estrato Socioeconómico de Medellín** | Alcaldía de Medellín | Datos Tabulares y Geoespaciales | Estratificación socioeconómica por manzanas |
| **Intel Image Classification** | Kaggle | Imágenes | Clasificación de escenas naturales (bosques, glaciares, montañas) |
| **Medicamentos** ✅ | - | JSON anidado | Información detallada sobre medicamentos, principios activos, fabricantes |

#### ✅ Dataset Seleccionado: Medicamentos

**Razones de selección:**
- Estructura compleja (JSON anidado)
- Riqueza de variables
- Reto interesante para análisis y preprocesamiento

### Fase 2: Análisis Exploratorio de Datos (EDA)

Análisis profundo del dataset de medicamentos documentado en `fase_2_EDA_medicamentos.ipynb`.

#### Pasos Realizados

1. **📥 Carga de Datos**: Importación del archivo `medicamentos.json`
2. **🔄 Normalización**: Conversión de JSON anidado a formato tabular usando `pandas.json_normalize`
3. **📈 Expansión**: Procesamiento de columnas con listas de diccionarios (`apresentacoes`, `principiosAtivos`)
4. **✅ Validación**: 
   - Revisión de tipos de datos
   - Análisis exhaustivo de valores nulos
   - Visualización de proporciones de datos faltantes

### Fase 3: Preprocesamiento y Reducción de Dimensionalidad

Preparación de datos para modelado futuro, detallada en `fase_3_preprocesamiento_y_reduccion.ipynb`.

#### Estrategia Implementada

##### 🔍 Análisis de Complejidad
- Evaluación de cardinalidad de variables categóricas
- Estrategia de "reducción primero" debido a la alta dimensionalidad

##### 🧹 Limpieza y Codificación
- **Manejo de Nulos**: Reemplazo por "Desconocido"
- **Reducción de Cardinalidad**: Agrupación de categorías poco frecuentes en "Otros"
- **One-Hot Encoding**: Implementación con matriz dispersa para optimización de memoria

##### 📉 Reducción de Dimensionalidad (PCA)
- **Método**: PCA incremental para manejo eficiente de memoria
- **Resultado**: 100 componentes principales
- **Varianza explicada**: 95%
- **Reducción**: De ~180,000 características a 100

#### 📊 Visualizaciones Generadas
- Gráfico de varianza explicada por componente
- Gráfico de varianza acumulada
- Dispersión de los primeros dos componentes principales
- Histogramas de distribución de componentes

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos Previos

```bash
pip install -r requirements.txt
```

### Orden de Ejecución

1. **[Opcional]** Exploración inicial:
   - `Exploracion_base_1.ipynb`
   - `Exploracion_base_2.ipynb`

2. **EDA Principal**:
   - `fase_2_EDA_medicamentos.ipynb`

3. **Preprocesamiento**:
   - `fase_3_preprocesamiento_y_reduccion.ipynb`

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Pandas** - Manipulación de datos
- **NumPy** - Computación numérica
- **Scikit-learn** - PCA y preprocesamiento
- **Matplotlib/Seaborn** - Visualización
- **Jupyter Notebook** - Desarrollo interactivo

## 📈 Resultados Clave

- ✅ Normalización exitosa de estructura JSON compleja
- ✅ Reducción dimensional significativa (99.94% de reducción)
- ✅ Conservación del 95% de varianza con solo 100 componentes
- ✅ Optimización de memoria mediante matrices dispersas

## 👥 Contribuciones

Este proyecto fue desarrollado como parte del curso de Análisis de Datos. Las contribuciones y sugerencias son bienvenidas a través de issues y pull requests.

## 📄 Licencia

Este proyecto es de uso académico y educativo.

---

*Desarrollado con ❤️ para el aprendizaje de Análisis de Datos*