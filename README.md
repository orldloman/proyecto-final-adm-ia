# Proyecto Final — Detección de Fraude Transaccional

**Aprendizaje de Máquina e Inteligencia Artificial 2026-2**  
**Unidad Profesional Interdisciplinaria de Ingeniería Campus Tlaxcala — IPN**

---

## Descripción

Este proyecto implementa un pipeline de inteligencia artificial para analizar transacciones con tarjeta de crédito y apoyar la toma de decisiones ante posibles fraudes.

El sistema integra tres etapas:

1. **Modelo no supervisado:** detección de anomalías y agrupamiento mediante DBSCAN y GMM.
2. **Modelo supervisado:** estimación de la probabilidad de fraude mediante Regresión Logística.
3. **Modelo probabilístico y de decisión:** selección de la acción con Máxima Utilidad Esperada (MEU): aprobar, revisar o bloquear.

El objetivo no es únicamente clasificar una transacción, sino convertir la evidencia obtenida por los modelos en una recomendación operativa explicable.

---

## Equipo y responsabilidades

| Integrante | Responsabilidad principal | Evidencia de participación |
|---|---|---|
| Orlando Lomán Córdova | Creación y administración inicial del repositorio; modelo no supervisado con DBSCAN y GMM | `02_no_supervisado.ipynb`, métricas, gráficas y commits |
| Andrés | Modelo supervisado con Regresión Logística y ensamblado del reporte técnico final | `03_supervisado.ipynb`, métricas, matriz de confusión, ROC y sección del reporte |
| Jonathan Chavarría Peralta | Modelo probabilístico y de decisión mediante MEU | `04_modelo_decision.ipynb`, curvas de utilidad, análisis de sensibilidad y resultados |
| Nicole Morales Mendoza | Integración funcional del pipeline, aplicación interactiva, pruebas finales y apoyo a la presentación | `app/app.py`, validación de entradas/salidas, pruebas de ejecución y material de integración |

---

## Dataset

**Credit Card Fraud Detection — Kaggle**

- 284,807 transacciones anonimizadas.
- 492 fraudes, equivalentes aproximadamente al 0.17 %.
- Variables `V1` a `V28` transformadas mediante PCA.
- Variables adicionales: `Time`, `Amount` y `Class`.

Fuente: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> El archivo `creditcard.csv` debe colocarse en `data/raw/`. No se versiona en GitHub debido a su tamaño.

---

## Estructura actual del repositorio

```text
proyecto-final-adm-ia/
├── README.md
├── .gitignore
├── 02_no_supervisado.ipynb
├── 03_supervisado.ipynb
├── 04_modelo_decision.ipynb
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       └── features_no_supervisado.csv
├── results/
│   ├── anomalias.png
│   ├── clusters.png
│   ├── gmm_bic.png
│   ├── pca_vista_inicial.png
│   ├── metricas_no_supervisado.csv
│   ├── matriz_confusion.png
│   ├── roc_curve.png
│   ├── metricas_supervisado.csv
│   ├── predicciones_supervisado.csv
│   ├── curvas_utilidad_meu.png
│   ├── distribucion_acciones_meu.png
│   ├── resultados_modelo_decision.csv
│   ├── sensibilidad_meu.csv
│   └── tabla_utilidades_parametros.csv
└── app/
    └── app.py
```

> La carpeta `app/` se incorporará durante la etapa de integración funcional.

---

## Flujo del pipeline

```text
Transacción
    ↓
Preprocesamiento y escalamiento
    ↓
DBSCAN / GMM
Detección de anomalía y perfil transaccional
    ↓
Regresión Logística
Probabilidad estimada de fraude
    ↓
Máxima Utilidad Esperada
Aprobar / Revisar / Bloquear
    ↓
Aplicación interactiva
```

---

## Modelos implementados

### 1. Modelo no supervisado

Se utilizaron:

- **DBSCAN**, para detectar observaciones aisladas o densidades atípicas.
- **GMM**, para generar agrupamientos probabilísticos y analizar perfiles transaccionales.

Resultados registrados:

| Modelo | Anomalías detectadas | Proporción | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| DBSCAN | 124 | 0.0155 | 0.0242 | 0.1875 | 0.0429 |
| GMM | 80 | 0.0100 | 0.0375 | 0.1875 | 0.0625 |

Los modelos no supervisados no utilizan la clase durante el entrenamiento; la etiqueta se conserva únicamente para evaluar posteriormente la relación entre anomalías y fraudes.

### 2. Modelo supervisado

Se utilizó **Regresión Logística** con validación cruzada estratificada y predicciones *out-of-fold*.

Resultados:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.8588 |
| Precision | 0.0097 |
| Recall | 0.6875 |
| F1-score | 0.0191 |
| ROC-AUC | 0.8509 |

El ROC-AUC muestra una capacidad razonable de discriminación, mientras que la baja precisión evidencia el efecto del fuerte desbalance de clases.

### 3. Modelo de decisión MEU

El módulo MEU recibe la probabilidad estimada de fraude y calcula la utilidad esperada de tres acciones:

- Aprobar.
- Revisar manualmente.
- Bloquear.

Parámetros del prototipo:

| Parámetro | Valor |
|---|---:|
| Margen de transacción legítima | 0.02 |
| Costo de revisión | 8.0 |
| Fracción de pérdida durante revisión | 0.10 |
| Costo de bloquear una operación legítima | 0.12 |
| Beneficio de bloquear un fraude | 0.90 |

Resultados del análisis de sensibilidad:

| Costo de bloqueo legítimo | Aprobar | Revisar | Bloquear |
|---:|---:|---:|---:|
| 0.05 | 2507 | 0 | 5493 |
| 0.12 | 4233 | 0 | 3767 |
| 0.25 | 5200 | 334 | 2466 |

Estos resultados muestran que la política de decisión cambia conforme se modifica el costo de bloquear erróneamente una transacción legítima.

---

## Ejecución

### Requisitos

Instalar las dependencias necesarias:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter streamlit
```

### Ejecución de notebooks

Desde la raíz del repositorio:

```bash
jupyter notebook 02_no_supervisado.ipynb
jupyter notebook 03_supervisado.ipynb
jupyter notebook 04_modelo_decision.ipynb
```

El orden recomendado es:

1. Modelo no supervisado.
2. Modelo supervisado.
3. Modelo de decisión MEU.

### Aplicación interactiva

Una vez incorporado `app/app.py`:

```bash
streamlit run app/app.py
```

La aplicación deberá mostrar:

- Indicador o señal de anomalía.
- Probabilidad estimada de fraude.
- Utilidades esperadas de cada acción.
- Acción recomendada: aprobar, revisar o bloquear.

---

## Archivos principales de salida

| Archivo | Descripción |
|---|---|
| `results/metricas_no_supervisado.csv` | Evaluación de DBSCAN y GMM |
| `results/metricas_supervisado.csv` | Métricas de Regresión Logística |
| `results/predicciones_supervisado.csv` | Probabilidad de fraude por transacción |
| `results/resultados_modelo_decision.csv` | Utilidades y acción MEU por transacción |
| `results/sensibilidad_meu.csv` | Cambios de política bajo diferentes costos |
| `results/curvas_utilidad_meu.png` | Comparación de utilidades esperadas |
| `results/distribucion_acciones_meu.png` | Frecuencia de aprobar, revisar y bloquear |

---

## Integración y validación final

La integración funcional deberá comprobar:

- Que los notebooks se ejecuten de principio a fin.
- Que las rutas sean relativas.
- Que los archivos generados tengan las columnas esperadas.
- Que la probabilidad de fraude esté entre 0 y 1.
- Que la acción recomendada pertenezca a `aprobar`, `revisar` o `bloquear`.
- Que la aplicación maneje errores de entrada.
- Que el reporte, las diapositivas y el video utilicen los mismos resultados.

---

## Restricciones

- No se utiliza Random Forest ni ensambles avanzados.
- Las utilidades son supuestos académicos del prototipo.
- El sistema no debe interpretarse como una herramienta financiera lista para producción.
- El fuerte desbalance de clases limita especialmente la precisión del modelo supervisado.

---

## Licencia

Proyecto académico para fines educativos.
