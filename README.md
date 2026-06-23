# Proyecto Final — Detección de Fraude Transaccional
**Aprendizaje de Máquina e Inteligencia Artificial 2026-2**

Unidad Profesional Interdisciplinaria de Ingeniería Campus Tlaxcala — IPN

---

## Descripción

Pipeline de inteligencia artificial para la detección de fraude en transacciones con tarjeta de crédito. El sistema combina tres etapas complementarias:

1. **Modelo no supervisado** — detección de anomalías con DBSCAN y GMM para identificar transacciones atípicas sin etiquetas.
2. **Modelo supervisado** — clasificador binario para predecir si una transacción es fraude.
3. **Modelo probabilístico / de decisión** — red de decisión con utilidad esperada máxima (MEU) para recomendar una acción: aprobar, revisar o bloquear.

---

## Equipo

| Integrante | Responsabilidad | Rama |
|---|---|---|
| Jonathan | Repositorio, integración, app | `main` |
| Orlando Lomán Córdova | Modelo no supervisado | `feature/orlando-no-supervisado` |
| Andrés | Modelo supervisado | `feature/andres-supervisado` |
| Nicole Morales Mendoza | Modelo probabilístico / decisión | `feature/nicole-decision` |

---

## Dataset

**Credit Card Fraud Detection** — Kaggle  
284 807 transacciones reales anonimizadas, 492 fraudes (0.17%)  
Variables: V1–V28 (PCA), Amount, Time, Class  
Fuente: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> Descargar `creditcard.csv` y colocarlo en `data/raw/` antes de ejecutar los notebooks.

---

## Estructura del Repositorio

```
proyecto-final-adm-ia/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # creditcard.csv (no versionado)
│   └── processed/            # features_no_supervisado.csv, etc.
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_no_supervisado.ipynb
│   ├── 03_supervisado.ipynb
│   └── 04_modelo_decision.ipynb
├── src/
│   ├── preprocessing.py
│   ├── unsupervised.py
│   ├── supervised.py
│   ├── decision_model.py
│   └── utils.py
├── app/
│   └── app.py
├── reports/
│   ├── main.tex
│   ├── referencias.bib
│   └── figuras/
├── slides/
│   └── presentacion.pdf
└── results/
    ├── metricas_no_supervisado.csv
    ├── matriz_confusion.png
    ├── clusters.png
    └── decision_meu.png
```

---

## Instalación

```bash
git clone https://github.com/<usuario>/proyecto-final-adm-ia.git
cd proyecto-final-adm-ia
pip install -r requirements.txt
```

---

## Ejecución de Notebooks

Correr en orden desde la raíz del repositorio:

```bash
jupyter notebook notebooks/01_exploracion.ipynb
jupyter notebook notebooks/02_no_supervisado.ipynb
jupyter notebook notebooks/03_supervisado.ipynb
jupyter notebook notebooks/04_modelo_decision.ipynb
```

> Cada notebook usa rutas relativas (`../data/`, `../results/`). No modificar la estructura de carpetas.

---

## App Interactiva

```bash
streamlit run app/app.py
```

La app permite ingresar los datos de una transacción nueva y muestra:
- Cluster asignado (modelo no supervisado)
- Probabilidad de fraude (modelo supervisado)
- Acción recomendada (modelo de decisión)

---

## Resultados Principales

| Modelo | Métrica | Valor |
|---|---|---|
| DBSCAN | Silhouette | ver `results/metricas_no_supervisado.csv` |
| GMM | Silhouette | ver `results/metricas_no_supervisado.csv` |
| Supervisado | F1-score | ver `results/matriz_confusion.png` |
| MEU | Acción óptima | ver `results/decision_meu.png` |

---

## Restricciones del Proyecto

- No se usa Random Forest ni ensambles avanzados.
- Modelos permitidos: SVM, Regresión Logística, KNN, Árbol CART, Red Neuronal simple, GMM, DBSCAN, K-Means, Red Bayesiana, MEU.

---

## Reporte

El reporte en PDF se genera compilando `reports/main.tex`:

```bash
pdflatex reports/main.tex
```

---

## Licencia

Proyecto académico — uso educativo únicamente.
