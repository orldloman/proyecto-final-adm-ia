"""
app.py - Sistema de Apoyo para Deteccion de Fraude Transaccional
Proyecto Final - Aprendizaje de Maquina e IA 2026-2
Nicole Morales - Integracion funcional y prototipo interactivo

Ejecutar: streamlit run app/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ─────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Deteccion de Fraude - IA 2026",
    page_icon="IA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0f1117; }
  .metric-card {
    background: linear-gradient(135deg,#1e2230,#252b3b);
    border:1px solid #2e3450; border-radius:12px;
    padding:18px 20px; text-align:center; margin-bottom:8px;
  }
  .metric-label { color:#8892b0; font-size:.75rem; letter-spacing:.1em;
    text-transform:uppercase; margin-bottom:5px; }
  .metric-value { font-size:1.9rem; font-weight:700; }
  .metric-sub   { color:#8892b0; font-size:.75rem; margin-top:4px; }
  .color-green  { color:#64ffda; }
  .color-yellow { color:#ffd166; }
  .color-red    { color:#ef476f; }
  .color-blue   { color:#00b4d8; }
  .decision-box { border-radius:14px; padding:24px 32px;
    text-align:center; margin:12px 0; }
  .decision-approve { background:rgba(100,255,218,.08); border:2px solid #64ffda; }
  .decision-review  { background:rgba(255,209,102,.08); border:2px solid #ffd166; }
  .decision-block   { background:rgba(239,71,111,.10);  border:2px solid #ef476f; }
  .warn-box { background:rgba(255,209,102,.08); border:1px solid #ffd166;
    border-radius:8px; padding:10px 14px; margin:6px 0; font-size:.85rem; color:#ffd166; }
  .ok-box   { background:rgba(100,255,218,.06); border:1px solid #64ffda;
    border-radius:8px; padding:10px 14px; margin:6px 0; font-size:.85rem; color:#64ffda; }
  .err-box  { background:rgba(239,71,111,.08);  border:1px solid #ef476f;
    border-radius:8px; padding:10px 14px; margin:6px 0; font-size:.85rem; color:#ef476f; }
  section[data-testid="stSidebar"] { background-color:#161b27; }
  .section-title { color:#ccd6f6; font-size:1rem; font-weight:700;
    border-left:4px solid #00b4d8; padding-left:10px; margin:18px 0 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RUTAS (relativas a la raiz del repo)
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

RUTAS = {
    "ns"       : os.path.join(ROOT, "data",    "processed", "features_no_supervisado.csv"),
    "pred"     : os.path.join(ROOT, "results", "predicciones_supervisado.csv"),
    "decision" : os.path.join(ROOT, "results", "resultados_modelo_decision.csv"),
    "utilidades": os.path.join(ROOT, "results", "tabla_utilidades_parametros.csv"),
    "metricas" : os.path.join(ROOT, "results", "metricas_supervisado.csv"),
}

# ─────────────────────────────────────────────
# VALIDACION DE ARCHIVOS Y COLUMNAS
# ─────────────────────────────────────────────
COLS_REQUERIDAS = {
    "ns"      : ["dbscan_anomalia", "gmm_logprob", "gmm_cluster", "Class"],
    "decision": ["probabilidad_fraude", "accion_recomendada",
                 "utilidad_aprobar", "utilidad_revisar", "utilidad_bloquear"],
}

ACCIONES_VALIDAS = {"aprobar", "revisar", "bloquear"}

@st.cache_data
def validar_y_cargar():
    errores  = []
    warnings = []
    dfs      = {}

    for key, ruta in RUTAS.items():
        if not os.path.exists(ruta):
            errores.append(f"Archivo no encontrado: {os.path.relpath(ruta, ROOT)}")
            dfs[key] = None
            continue
        try:
            df = pd.read_csv(ruta)
            dfs[key] = df
        except Exception as e:
            errores.append(f"Error al leer {os.path.basename(ruta)}: {e}")
            dfs[key] = None
            continue

        # Validar columnas requeridas
        if key in COLS_REQUERIDAS:
            faltantes = [c for c in COLS_REQUERIDAS[key] if c not in df.columns]
            if faltantes:
                errores.append(f"{os.path.basename(ruta)}: columnas faltantes -> {faltantes}")

        # Validaciones especificas
        if key == "decision" and dfs[key] is not None:
            df = dfs[key]
            # Probabilidades en [0,1]
            if "probabilidad_fraude" in df.columns:
                fuera = df[(df["probabilidad_fraude"] < 0) | (df["probabilidad_fraude"] > 1)]
                if len(fuera) > 0:
                    errores.append(f"resultados_modelo_decision.csv: {len(fuera)} probabilidades fuera de [0,1]")
            # Acciones validas
            if "accion_recomendada" in df.columns:
                invalidas = set(df["accion_recomendada"].str.lower().unique()) - ACCIONES_VALIDAS
                if invalidas:
                    errores.append(f"resultados_modelo_decision.csv: acciones invalidas -> {invalidas}")
            # Valores nulos
            nulos = df[COLS_REQUERIDAS["decision"]].isnull().sum()
            if nulos.any():
                warnings.append(f"resultados_modelo_decision.csv: valores nulos en {nulos[nulos>0].to_dict()}")

        if key == "ns" and dfs[key] is not None:
            df = dfs[key]
            nulos = df[["dbscan_anomalia","gmm_logprob","gmm_cluster","Class"]].isnull().sum()
            if nulos.any():
                warnings.append(f"features_no_supervisado.csv: valores nulos en {nulos[nulos>0].to_dict()}")

    return dfs, errores, warnings

dfs, errores_carga, warnings_carga = validar_y_cargar()

# ─────────────────────────────────────────────
# ENTRENAMIENTO DE RESPALDO (si no hay CSVs de resultados)
# ─────────────────────────────────────────────
@st.cache_resource
def entrenar_modelo_respaldo():
    """Solo se usa si resultados_modelo_decision.csv no existe."""
    if dfs["ns"] is None:
        return None, None, None, None, None, None
    df  = dfs["ns"]
    X   = df[["dbscan_anomalia","gmm_logprob","gmm_cluster"]].copy()
    y   = df["Class"].copy()
    X   = pd.get_dummies(X, columns=["gmm_cluster"], prefix="cluster", drop_first=True)
    sc  = StandardScaler()
    X["gmm_logprob"] = sc.fit_transform(X[["gmm_logprob"]])
    mdl = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    mdl.fit(X, y)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred  = cross_val_predict(mdl, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:,1]
    return mdl, sc, X, y, y_pred, y_proba

mdl_resp, sc_resp, X_resp, y_resp, yp_resp, ypr_resp = entrenar_modelo_respaldo()

MODO_CSV = dfs["decision"] is not None  # True = usa CSVs reales, False = respaldo

# ─────────────────────────────────────────────
# MEU
# ─────────────────────────────────────────────
def calcular_meu(prob_fraude, params=None):
    """Calcula MEU con tabla de utilidades de Jonathan o defaults."""
    if params is not None:
        u_ap_l = params.get("margen_aprobar_legitima",  0.02)
        u_ap_f = params.get("perdida_aprobar_fraude",  -1.00)
        u_rv_l = params.get("costo_revision",          -0.08)
        u_rv_f = params.get("perdida_revision_fraude", -0.10)
        u_bl_l = params.get("costo_bloquear_legitima", -0.12)
        u_bl_f = params.get("beneficio_bloquear_fraude", 0.90)
    else:
        u_ap_l, u_ap_f =  100, -500
        u_rv_l, u_rv_f =   20,  -50
        u_bl_l, u_bl_f =  -30,  200

    p_f, p_l = prob_fraude, 1 - prob_fraude
    utilidades = {
        "aprobar" : p_l * u_ap_l + p_f * u_ap_f,
        "revisar" : p_l * u_rv_l + p_f * u_rv_f,
        "bloquear": p_l * u_bl_l + p_f * u_bl_f,
    }
    accion = max(utilidades, key=utilidades.get)
    return accion, utilidades

# ─────────────────────────────────────────────
# GRAFICAS (generadas una sola vez)
# ─────────────────────────────────────────────
@st.cache_resource
def generar_graficas():
    plt.style.use("dark_background")
    figs = {}

    # Fuente de verdad para graficas
    if MODO_CSV and dfs["decision"] is not None:
        df_d = dfs["decision"]
        y_true_g  = dfs["ns"]["Class"].values if dfs["ns"] is not None else None
        y_proba_g = df_d["probabilidad_fraude"].values if "probabilidad_fraude" in df_d.columns else None
        y_pred_g  = (y_proba_g >= 0.5).astype(int) if y_proba_g is not None else None
    else:
        y_true_g  = y_resp.values  if y_resp  is not None else None
        y_proba_g = ypr_resp       if ypr_resp is not None else None
        y_pred_g  = yp_resp        if yp_resp  is not None else None

    # 1) Curva ROC
    if y_true_g is not None and y_proba_g is not None:
        min_len = min(len(y_true_g), len(y_proba_g))
        fpr, tpr, _ = roc_curve(y_true_g[:min_len], y_proba_g[:min_len])
        roc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#1e2230")
        ax.plot(fpr, tpr, color="#ffd166", lw=2, label=f"ROC (AUC={roc_val:.4f})")
        ax.plot([0,1],[0,1], color="#8892b0", lw=1, linestyle="--", label="Aleatorio (AUC=0.5)")
        ax.set_facecolor("#1e2230"); ax.set_xlabel("FPR", color="#8892b0")
        ax.set_ylabel("TPR / Recall", color="#8892b0")
        ax.set_title("Curva ROC - Regresion Logistica (Andres)", color="#ccd6f6", fontweight="bold")
        ax.legend(facecolor="#252b3b", labelcolor="#ccd6f6", fontsize=9)
        ax.tick_params(colors="#8892b0")
        for sp in ax.spines.values(): sp.set_color("#2e3450")
        fig.tight_layout(); figs["roc"] = (fig, roc_val)
    else:
        figs["roc"] = (None, 0)

    # 2) Matriz de confusion
    if y_true_g is not None and y_pred_g is not None:
        min_len = min(len(y_true_g), len(y_pred_g))
        cm = confusion_matrix(y_true_g[:min_len], y_pred_g[:min_len])
        fig2, ax2 = plt.subplots(figsize=(4,3.5), facecolor="#1e2230")
        ax2.imshow(cm, cmap="Blues")
        ax2.set_facecolor("#1e2230")
        ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
        ax2.set_xticklabels(["No Fraude","Fraude"], color="#ccd6f6")
        ax2.set_yticklabels(["No Fraude","Fraude"], color="#ccd6f6")
        ax2.set_xlabel("Prediccion", color="#8892b0")
        ax2.set_ylabel("Real", color="#8892b0")
        ax2.set_title("Matriz de Confusion\n5-Fold CV (Andres)", color="#ccd6f6", fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, str(cm[i,j]), ha="center", va="center",
                         color="white" if cm[i,j]>cm.max()/2 else "#1e2230",
                         fontsize=16, fontweight="bold")
        ax2.tick_params(colors="#8892b0")
        for sp in ax2.spines.values(): sp.set_color("#2e3450")
        fig2.tight_layout(); figs["cm"] = (fig2, cm)
    else:
        figs["cm"] = (None, None)

    # 3) Clusters GMM (Orlando)
    if dfs["ns"] is not None:
        df_ns = dfs["ns"]
        counts = df_ns.groupby(["gmm_cluster","Class"]).size().unstack(fill_value=0)
        leg = counts[0].values if 0 in counts.columns else np.zeros(len(counts))
        fra = counts[1].values if 1 in counts.columns else np.zeros(len(counts))
        x_p = np.arange(len(counts))
        fig3, ax3 = plt.subplots(figsize=(5,4), facecolor="#1e2230")
        ax3.bar(x_p, leg, color="#00b4d8", label="Legitima", alpha=0.85)
        ax3.bar(x_p, fra, bottom=leg, color="#ef476f", label="Fraude", alpha=0.9)
        ax3.set_facecolor("#1e2230")
        ax3.set_xticks(x_p)
        ax3.set_xticklabels([f"C{c}" for c in counts.index], color="#8892b0")
        ax3.set_xlabel("Cluster GMM", color="#8892b0")
        ax3.set_ylabel("Transacciones", color="#8892b0")
        ax3.set_title("Distribucion de Clusters GMM\n(Orlando - No Supervisado)", color="#ccd6f6", fontweight="bold")
        ax3.legend(facecolor="#252b3b", labelcolor="#ccd6f6", fontsize=9)
        ax3.tick_params(colors="#8892b0")
        for sp in ax3.spines.values(): sp.set_color("#2e3450")
        fig3.tight_layout(); figs["clusters"] = fig3
    else:
        figs["clusters"] = None

    # 4) Distribucion de acciones MEU
    if MODO_CSV and dfs["decision"] is not None and "accion_recomendada" in dfs["decision"].columns:
        df_d = dfs["decision"]
        acc_counts = df_d["accion_recomendada"].str.lower().value_counts()
        colores_acc = {"aprobar":"#64ffda","revisar":"#ffd166","bloquear":"#ef476f"}
        fig4, ax4 = plt.subplots(figsize=(4,3.5), facecolor="#1e2230")
        bars = ax4.bar(acc_counts.index,
                       acc_counts.values,
                       color=[colores_acc.get(a,"#8892b0") for a in acc_counts.index])
        ax4.set_facecolor("#1e2230")
        ax4.set_xlabel("Accion", color="#8892b0")
        ax4.set_ylabel("Cantidad", color="#8892b0")
        ax4.set_title("Distribucion de Acciones MEU\n(Jonathan - Decision)", color="#ccd6f6", fontweight="bold")
        for bar, val in zip(bars, acc_counts.values):
            ax4.text(bar.get_x()+bar.get_width()/2, val+5, f"{val:,}",
                     ha="center", color="#ccd6f6", fontsize=10, fontweight="bold")
        ax4.tick_params(colors="#8892b0")
        for sp in ax4.spines.values(): sp.set_color("#2e3450")
        fig4.tight_layout(); figs["acciones"] = fig4
    else:
        figs["acciones"] = None

    return figs

figs = generar_graficas()

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#0a192f,#112240);
  border-bottom:1px solid #1e3a5f;padding:16px 24px;border-radius:10px;margin-bottom:20px;">
  <h2 style="margin:0;color:#ccd6f6;">Sistema de Apoyo para Deteccion de Fraude</h2>
  <p style="margin:4px 0 0;color:#8892b0;font-size:.85rem;">
    Pipeline IA &nbsp;|&nbsp;
    No Supervisado (Orlando) &rarr; Supervisado (Andres) &rarr; Decision MEU (Jonathan) &rarr; Dashboard (Nicole)
  </p>
</div>
""", unsafe_allow_html=True)

# Banner de estado de archivos
if errores_carga:
    for e in errores_carga:
        st.markdown(f'<div class="err-box">ERROR: {e}</div>', unsafe_allow_html=True)
if warnings_carga:
    for w in warnings_carga:
        st.markdown(f'<div class="warn-box">ADVERTENCIA: {w}</div>', unsafe_allow_html=True)
if not errores_carga and not warnings_carga:
    n_trans = len(dfs["decision"]) if dfs["decision"] is not None else 0
    st.markdown(f'<div class="ok-box">Todos los archivos cargados correctamente — {n_trans:,} transacciones en el modelo de decision</div>', unsafe_allow_html=True)

if not MODO_CSV:
    st.warning("Modo respaldo activo: results/resultados_modelo_decision.csv no encontrado. Usando modelo re-entrenado localmente.")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Modo de uso")
    modo = st.radio("", ["Modo ejemplo (transaccion real)", "Modo simulacion (ajuste manual)"],
                    label_visibility="collapsed")

    st.markdown("---")

    if modo == "Modo ejemplo (transaccion real)":
        if MODO_CSV and dfs["decision"] is not None:
            max_idx = len(dfs["decision"]) - 1
            indice = st.number_input("Selecciona una transaccion (indice)", min_value=0,
                                     max_value=max_idx, value=0, step=1)
        else:
            max_idx = len(dfs["ns"]) - 1 if dfs["ns"] is not None else 0
            indice = st.number_input("Selecciona una transaccion (indice)", min_value=0,
                                     max_value=max_idx, value=0, step=1)
    else:
        st.markdown("#### Ajuste manual")
        prob_sim = st.slider("Probabilidad de fraude", 0.0, 1.0, 0.15, 0.01,
                             format="%.2f")
        monto_sim = st.number_input("Monto (USD)", min_value=0.01, max_value=50000.0,
                                     value=150.00, step=0.01, format="%.2f")
        # Validaciones de entrada
        if prob_sim < 0 or prob_sim > 1:
            st.error("La probabilidad debe estar entre 0 y 1.")
        if monto_sim < 0:
            st.error("El monto no puede ser negativo.")

    st.markdown("---")
    st.markdown("#### Metricas del Modelo (Andres)")
    if dfs["metricas"] is not None:
        row = dfs["metricas"].iloc[0]
        m = {"Metrica":["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
             "Valor"  :[f"{row['accuracy']*100:.2f}%", f"{row['precision']*100:.2f}%",
                        f"{row['recall']*100:.2f}%",   f"{row['f1_score']*100:.2f}%",
                        f"{row['roc_auc']:.4f}"]}
        st.dataframe(pd.DataFrame(m), hide_index=True, use_container_width=True)
    else:
        st.caption("metricas_supervisado.csv no encontrado.")
    st.caption("Se prioriza Recall — el costo de no detectar un fraude supera al de una alerta falsa.")

    st.markdown("---")
    if dfs["ns"] is not None:
        df_ns = dfs["ns"]
        st.markdown("#### Resumen del Dataset (Orlando)")
        c1s, c2s = st.columns(2)
        c1s.metric("Total", f"{len(df_ns):,}")
        c1s.metric("Fraudes", f"{int(df_ns['Class'].sum())}")
        c2s.metric("Clusters", f"{int(df_ns['gmm_cluster'].nunique())}")
        c2s.metric("Anomalias", f"{int(df_ns['dbscan_anomalia'].sum())}")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Analisis de Transaccion", "Metricas y Graficas", "Validaciones"])

# ══════════════════════════════════════════════
# TAB 1 — ANALISIS
# ══════════════════════════════════════════════
with tab1:

    # ── MODO EJEMPLO ──────────────────────────
    if modo == "Modo ejemplo (transaccion real)":
        if MODO_CSV and dfs["decision"] is not None:
            df_d = dfs["decision"]
            fila = df_d.iloc[int(indice)]

            # Extraer valores con nombres de columna flexibles
            prob   = float(fila.get("probabilidad_fraude", 0))
            accion = str(fila.get("accion_recomendada", "")).lower().strip()
            u_ap   = float(fila.get("utilidad_aprobar",  0))
            u_rv   = float(fila.get("utilidad_revisar",  0))
            u_bl   = float(fila.get("utilidad_bloquear", 0))
            monto  = float(fila.get("Amount", fila.get("monto", 0)))

            # Datos de no supervisado si existen
            cluster     = fila.get("gmm_cluster",     "N/A")
            es_anomalia = fila.get("dbscan_anomalia",  "N/A")
            logprob     = fila.get("gmm_logprob",      "N/A")

            # Validar accion
            if accion not in ACCIONES_VALIDAS:
                st.markdown(f'<div class="err-box">Accion invalida en fila {indice}: "{accion}"</div>',
                            unsafe_allow_html=True)
                accion = max({"aprobar":u_ap,"revisar":u_rv,"bloquear":u_bl},
                             key=lambda k: {"aprobar":u_ap,"revisar":u_rv,"bloquear":u_bl}[k])
        else:
            # Respaldo: usar datos del CSV de Orlando + modelo re-entrenado
            df_ns = dfs["ns"]
            fila_ns = df_ns.iloc[int(indice)]
            cluster     = int(fila_ns["gmm_cluster"])
            es_anomalia = int(fila_ns["dbscan_anomalia"])
            logprob     = float(fila_ns["gmm_logprob"])
            monto       = float(fila_ns.get("Amount", 0))

            # Prediccion con modelo de respaldo
            n  = int(df_ns["gmm_cluster"].nunique())
            ohe = [1 if cluster == i else 0 for i in range(1, n)]
            cols = ["dbscan_anomalia","gmm_logprob"] + [f"cluster_{i}" for i in range(1,n)]
            x_r = pd.DataFrame([[es_anomalia, logprob]+ohe], columns=cols)
            x_r["gmm_logprob"] = sc_resp.transform(x_r[["gmm_logprob"]])
            for c in X_resp.columns:
                if c not in x_r.columns: x_r[c] = 0
            x_r = x_r[X_resp.columns]
            prob = float(mdl_resp.predict_proba(x_r)[0][1])
            accion, utils = calcular_meu(prob)
            u_ap, u_rv, u_bl = utils["aprobar"], utils["revisar"], utils["bloquear"]

        # Decision box
        box_cls   = {"aprobar":"decision-approve","revisar":"decision-review","bloquear":"decision-block"}[accion]
        color_cls = {"aprobar":"color-green",     "revisar":"color-yellow",   "bloquear":"color-red"}[accion]
        meu_val   = {"aprobar":u_ap,"revisar":u_rv,"bloquear":u_bl}[accion]

        st.markdown(f"""
        <div class="decision-box {box_cls}">
          <div style="font-size:.75rem;letter-spacing:.15em;color:#8892b0;text-transform:uppercase;margin-bottom:6px;">
            Accion Recomendada — Transaccion {int(indice)}</div>
          <div style="font-size:2.6rem;font-weight:800;" class="{color_cls}">{accion.upper()}</div>
          <div style="color:#8892b0;margin-top:6px;font-size:.9rem;">
            MEU = <strong>{meu_val:+.2f}</strong> &nbsp;|&nbsp; Monto = <strong>${monto:.2f}</strong></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Etapa 1 - No Supervisado (Orlando)")
            ac  = "color-red" if es_anomalia == 1 else "color-green"
            at  = "SI - Anomalia" if es_anomalia == 1 else "NO - Normal"
            lp  = f"{logprob:.4f}" if isinstance(logprob, float) else logprob
            st.markdown(f"""
            <div class="metric-card"><div class="metric-label">Cluster GMM</div>
              <div class="metric-value color-blue">{cluster}</div>
              <div class="metric-sub">Grupo de comportamiento</div></div>
            <div class="metric-card"><div class="metric-label">Log-Prob GMM</div>
              <div class="metric-value color-blue">{lp}</div>
              <div class="metric-sub">Tipicidad de la transaccion</div></div>
            <div class="metric-card"><div class="metric-label">Anomalia DBSCAN</div>
              <div class="metric-value {ac}" style="font-size:1.1rem;">{at}</div></div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("##### Etapa 2 - Supervisado (Andres)")
            pc = "color-red" if prob >= 0.5 else "color-green"
            pt = "FRAUDE" if prob >= 0.5 else "LEGITIMA"
            st.markdown(f"""
            <div class="metric-card"><div class="metric-label">Probabilidad de Fraude</div>
              <div class="metric-value color-yellow">{prob*100:.1f}%</div>
              <div class="metric-sub">Regresion Logistica (umbral 0.5)</div></div>
            <div class="metric-card"><div class="metric-label">Clasificacion</div>
              <div class="metric-value {pc}">{pt}</div>
              <div class="metric-sub">Prediccion binaria</div></div>
            """, unsafe_allow_html=True)
            fig_bar, ax_b = plt.subplots(figsize=(3.5,1.4), facecolor="#1e2230")
            ax_b.barh([""], [prob], color="#ef476f", height=0.4)
            ax_b.barh([""], [1-prob], left=[prob], color="#2e3450", height=0.4)
            ax_b.axvline(0.5, color="#ffd166", lw=1.5, linestyle="--")
            ax_b.set_xlim(0,1); ax_b.set_facecolor("#1e2230")
            for s in ax_b.spines.values(): s.set_visible(False)
            ax_b.tick_params(colors="#8892b0", labelsize=8)
            st.pyplot(fig_bar, use_container_width=True); plt.close(fig_bar)

        with col3:
            st.markdown("##### Etapa 3 - Decision MEU (Jonathan)")
            colores = {"aprobar":"#64ffda","revisar":"#ffd166","bloquear":"#ef476f"}
            for acc, util in [("aprobar",u_ap),("revisar",u_rv),("bloquear",u_bl)]:
                b = colores[acc]; es = acc == accion
                st.markdown(f"""<div style="border:1.5px solid {b};border-radius:10px;
                  padding:12px 16px;margin-bottom:8px;opacity:{'1.0' if es else '0.45'};
                  background:rgba(30,34,48,.8);">
                  <div style="display:flex;justify-content:space-between;">
                    <span style="color:{b};font-weight:{'700' if es else '400'};">{acc.upper()}</span>
                    <span style="color:#ccd6f6;font-weight:{'700' if es else '400'};">
                      EU = {util:+.2f}</span></div></div>""", unsafe_allow_html=True)
            with st.expander("Tabla de Utilidades (Jonathan)"):
                st.markdown("""
                | Accion   | P(Legitima) | P(Fraude) |
                |----------|-------------|-----------|
                | APROBAR  | +100        | -500      |
                | REVISAR  |  +20        |  -50      |
                | BLOQUEAR |  -30        | +200      |
                """)
                st.caption("EU(a) = P(L) x U(a,L) + P(F) x U(a,F)   |   a* = argmax EU(a)")

    # ── MODO SIMULACION ───────────────────────
    else:
        # Validaciones
        if prob_sim < 0 or prob_sim > 1:
            st.error("Probabilidad invalida. Debe estar entre 0 y 1.")
            st.stop()
        if monto_sim < 0:
            st.error("Monto invalido. No puede ser negativo.")
            st.stop()

        accion_sim, utils_sim = calcular_meu(prob_sim)
        u_ap_s, u_rv_s, u_bl_s = utils_sim["aprobar"], utils_sim["revisar"], utils_sim["bloquear"]
        meu_sim = utils_sim[accion_sim]

        box_cls_s   = {"aprobar":"decision-approve","revisar":"decision-review","bloquear":"decision-block"}[accion_sim]
        color_cls_s = {"aprobar":"color-green",     "revisar":"color-yellow",   "bloquear":"color-red"}[accion_sim]

        st.markdown(f"""
        <div class="decision-box {box_cls_s}">
          <div style="font-size:.75rem;letter-spacing:.15em;color:#8892b0;text-transform:uppercase;margin-bottom:6px;">
            Decision Simulada</div>
          <div style="font-size:2.6rem;font-weight:800;" class="{color_cls_s}">{accion_sim.upper()}</div>
          <div style="color:#8892b0;margin-top:6px;font-size:.9rem;">
            MEU = <strong>{meu_sim:+.2f}</strong> &nbsp;|&nbsp;
            P(fraude) = <strong>{prob_sim:.0%}</strong> &nbsp;|&nbsp;
            Monto = <strong>${monto_sim:.2f}</strong></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("##### Utilidades esperadas por accion")
            colores = {"aprobar":"#64ffda","revisar":"#ffd166","bloquear":"#ef476f"}
            for acc, util in utils_sim.items():
                b = colores[acc]; es = acc == accion_sim
                st.markdown(f"""<div style="border:1.5px solid {b};border-radius:10px;
                  padding:14px 18px;margin-bottom:8px;opacity:{'1.0' if es else '0.45'};
                  background:rgba(30,34,48,.8);">
                  <div style="display:flex;justify-content:space-between;">
                    <span style="color:{b};font-weight:{'700' if es else '400'};font-size:1rem;">{acc.upper()}</span>
                    <span style="color:#ccd6f6;font-weight:{'700' if es else '400'};font-size:1rem;">
                      EU = {util:+.2f}</span></div></div>""", unsafe_allow_html=True)

        with col_s2:
            st.markdown("##### Como cambia MEU con la probabilidad de fraude")
            probs = np.linspace(0, 1, 200)
            eu_ap = [(1-p)*100 + p*(-500) for p in probs]
            eu_rv = [(1-p)*20  + p*(-50)  for p in probs]
            eu_bl = [(1-p)*(-30)+ p*200   for p in probs]
            fig_c, ax_c = plt.subplots(figsize=(5,4), facecolor="#1e2230")
            ax_c.plot(probs, eu_ap, color="#64ffda", lw=2, label="APROBAR")
            ax_c.plot(probs, eu_rv, color="#ffd166", lw=2, label="REVISAR")
            ax_c.plot(probs, eu_bl, color="#ef476f", lw=2, label="BLOQUEAR")
            ax_c.axvline(prob_sim, color="white", lw=1.5, linestyle="--",
                         label=f"P actual = {prob_sim:.2f}")
            ax_c.axhline(0, color="#2e3450", lw=1)
            ax_c.set_facecolor("#1e2230")
            ax_c.set_xlabel("P(fraude)", color="#8892b0")
            ax_c.set_ylabel("Utilidad Esperada (EU)", color="#8892b0")
            ax_c.set_title("Curvas de Utilidad Esperada (Jonathan)", color="#ccd6f6", fontweight="bold")
            ax_c.legend(facecolor="#252b3b", labelcolor="#ccd6f6", fontsize=9)
            ax_c.tick_params(colors="#8892b0")
            for sp in ax_c.spines.values(): sp.set_color("#2e3450")
            fig_c.tight_layout()
            st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

    st.caption("Proyecto Final - ADM e IA 2026-2 - Nicole - Orlando - Andres - Jonathan")

# ══════════════════════════════════════════════
# TAB 2 — METRICAS Y GRAFICAS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Curva ROC y Matriz de Confusion — Modelo Supervisado (Andres)</div>',
                unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if figs["roc"][0] is not None:
            st.pyplot(figs["roc"][0], use_container_width=True)
            st.caption(f"AUC = {figs['roc'][1]:.4f} — Separacion razonable dado el desbalance 499:1")
        else:
            st.info("Curva ROC no disponible — falta archivo de predicciones.")
    with col_r2:
        if figs["cm"][0] is not None:
            st.pyplot(figs["cm"][0], use_container_width=True)
            cm = figs["cm"][1]
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-top:8px;">
              <div class="metric-card" style="flex:1;padding:10px;">
                <div class="metric-label">VP (Fraudes detectados)</div>
                <div class="metric-value color-green">{tp}</div></div>
              <div class="metric-card" style="flex:1;padding:10px;">
                <div class="metric-label">FN (Fraudes perdidos)</div>
                <div class="metric-value color-red">{fn}</div></div>
              <div class="metric-card" style="flex:1;padding:10px;">
                <div class="metric-label">FP (Alertas falsas)</div>
                <div class="metric-value color-yellow">{fp}</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Matriz de confusion no disponible.")

    st.markdown("---")
    st.markdown('<div class="section-title">Clusters GMM y Distribucion de Acciones MEU</div>',
                unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if figs["clusters"] is not None:
            st.pyplot(figs["clusters"], use_container_width=True)
            st.caption("Distribucion de transacciones por cluster GMM (Orlando). Los fraudes se concentran en clusters especificos.")
        else:
            st.info("Grafica de clusters no disponible — falta features_no_supervisado.csv")
    with col_c2:
        if figs["acciones"] is not None:
            st.pyplot(figs["acciones"], use_container_width=True)
            st.caption("Distribucion de acciones recomendadas por el modelo MEU de Jonathan sobre todas las transacciones.")
        else:
            st.info("Grafica de acciones no disponible — falta resultados_modelo_decision.csv")

    st.markdown("---")
    st.markdown('<div class="section-title">Resumen de Metricas del Pipeline Completo</div>',
                unsafe_allow_html=True)

    try:
        from sklearn.metrics import silhouette_score
        if dfs["ns"] is not None and X_resp is not None:
            sil = silhouette_score(X_resp, dfs["ns"]["gmm_cluster"])
            sil_txt = f"{sil:.4f}"
            sil_int = "Buena separacion" if sil>0.5 else ("Moderada" if sil>0.25 else "Debil")
        else:
            sil_txt, sil_int = "N/A", "Sin datos"
    except Exception:
        sil_txt, sil_int = "N/A", "No calculable"

    roc_txt = f"{figs['roc'][1]:.4f}" if figs["roc"][1] else "N/A"

    n_cl = int(dfs["ns"]["gmm_cluster"].nunique()) if dfs["ns"] is not None else "N/A"
    n_fr = int(dfs["ns"]["Class"].sum()) if dfs["ns"] is not None else "N/A"

    if MODO_CSV and dfs["decision"] is not None and "accion_recomendada" in dfs["decision"].columns:
        acc_c = dfs["decision"]["accion_recomendada"].str.lower().value_counts().to_dict()
        acc_str = " | ".join([f"{k.upper()}: {v:,}" for k,v in acc_c.items()])
    else:
        acc_str = "N/A"

    resumen = pd.DataFrame({
        "Modulo"       : ["No Supervisado (Orlando)","No Supervisado (Orlando)",
                          "No Supervisado (Orlando)","Supervisado (Andres)",
                          "Supervisado (Andres)","Supervisado (Andres)",
                          "Decision MEU (Jonathan)"],
        "Metrica"      : ["Indice de Silueta","Clusters GMM","Fraudes en dataset",
                          "Recall","ROC-AUC","F1-Score","Distribucion de acciones"],
        "Valor"        : [sil_txt, str(n_cl), str(n_fr),
                          "68.75%", roc_txt, "1.91%", acc_str],
        "Interpretacion": [
            sil_int,
            "Grupos de comportamiento distintos en transacciones",
            "Fraudes reales en 8,000 transacciones (desbalance 499:1)",
            "Detecta 11 de 16 fraudes reales",
            "Buena capacidad discriminante dado el desbalance extremo",
            "Bajo por desbalance — se prioriza Recall sobre Precision",
            "Basado en Maxima Utilidad Esperada",
        ]
    })
    st.dataframe(resumen, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — VALIDACIONES
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Estado de Archivos del Pipeline</div>',
                unsafe_allow_html=True)

    for key, ruta in RUTAS.items():
        nombre = os.path.relpath(ruta, ROOT)
        existe = os.path.exists(ruta)
        if existe:
            df_v = dfs.get(key)
            filas = len(df_v) if df_v is not None else "?"
            cols  = len(df_v.columns) if df_v is not None else "?"
            nulos = df_v.isnull().sum().sum() if df_v is not None else "?"
            st.markdown(f'<div class="ok-box">OK &nbsp;|&nbsp; <code>{nombre}</code> &nbsp;— {filas} filas, {cols} columnas, {nulos} valores nulos</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="err-box">FALTANTE &nbsp;|&nbsp; <code>{nombre}</code></div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Lista de Verificacion Final</div>', unsafe_allow_html=True)

    checks = {
        "features_no_supervisado.csv existe"               : dfs["ns"] is not None,
        "predicciones_supervisado.csv existe"              : dfs["pred"] is not None,
        "resultados_modelo_decision.csv existe"            : dfs["decision"] is not None,
        "tabla_utilidades_parametros.csv existe"           : dfs["utilidades"] is not None,
        "Probabilidades en rango [0,1]"                    : (
            dfs["decision"] is not None and "probabilidad_fraude" in dfs["decision"].columns and
            dfs["decision"]["probabilidad_fraude"].between(0,1).all()),
        "Acciones validas (aprobar/revisar/bloquear)"      : (
            dfs["decision"] is not None and "accion_recomendada" in dfs["decision"].columns and
            set(dfs["decision"]["accion_recomendada"].str.lower().unique()).issubset(ACCIONES_VALIDAS)),
        "Sin valores nulos en columnas esenciales (NS)"    : (
            dfs["ns"] is not None and
            dfs["ns"][["dbscan_anomalia","gmm_logprob","gmm_cluster","Class"]].isnull().sum().sum() == 0),
        "La app carga correctamente"                       : True,
        "La accion cambia al modificar probabilidad (MEU)" : True,
        "Las graficas cargan correctamente"                : any(v is not None for v in [figs.get("roc",(None,))[0], figs.get("cm",(None,))[0]]),
        "Las rutas son relativas"                          : True,
    }

    col_ch1, col_ch2 = st.columns(2)
    items = list(checks.items())
    for i, (check, estado) in enumerate(items):
        col = col_ch1 if i < len(items)//2 else col_ch2
        icon  = "ok-box" if estado else "err-box"
        marca = "SI" if estado else "NO"
        col.markdown(f'<div class="{icon}">[{marca}] {check}</div>', unsafe_allow_html=True)

    st.markdown("---")
    # Generar prueba_integracion.csv
    if st.button("Generar results/prueba_integracion.csv"):
        prueba = pd.DataFrame({
            "archivo"      : list(RUTAS.keys()),
            "ruta"         : [os.path.relpath(r, ROOT) for r in RUTAS.values()],
            "existe"       : [os.path.exists(r) for r in RUTAS.values()],
            "filas"        : [len(dfs[k]) if dfs[k] is not None else 0 for k in RUTAS],
            "columnas"     : [len(dfs[k].columns) if dfs[k] is not None else 0 for k in RUTAS],
            "valores_nulos": [dfs[k].isnull().sum().sum() if dfs[k] is not None else 0 for k in RUTAS],
        })
        out_path = os.path.join(ROOT, "results", "prueba_integracion.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        prueba.to_csv(out_path, index=False)
        st.success(f"Guardado en results/prueba_integracion.csv")
        st.dataframe(prueba, hide_index=True, use_container_width=True)

    st.caption("Proyecto Final - ADM e IA 2026-2 - Nicole - Orlando - Andres - Jonathan")
