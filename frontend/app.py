import streamlit as st
import requests
import os
import json
import textwrap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="MedTranscript Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CARGAR ESTILOS CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Asumiendo que se corre desde la raíz del proyecto
load_css("Diseños Tablero/styles.css")

# --- VARIABLES GLOBALES ---
API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v3.0-sigmoid-10classes")

# --- INYECTAR NAVBAR ORIGINAL ---
navbar_html = f"""
    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon">M</div>
            <div class="logo-text">
                <h1>MedTranscript Classifier</h1>
                <p>NLP • Specialty Classification • ML API</p>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="background:#1a1a2e;color:#a78bfa;padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:600;">{MODEL_VERSION}</span>
            <div class="api-status">API: Online</div>
        </div>
    </nav>
"""
st.markdown("".join([line.strip() for line in navbar_html.split("\n")]), unsafe_allow_html=True)

# --- SISTEMA DE NAVEGACIÓN (TABS DE STREAMLIT) ---
tab_home, tab_clasificar, tab_analytics = st.tabs(["🏠 Home", "📊 Clasificar", "📈 Analytics"])

# ==========================================
# PESTAÑA 1: HOME
# ==========================================
with tab_home:
    # Datos verdaderos del dataset basados en el último output de entrenamiento
    # (1962 registros, 10 clases, 94.0% Accuracy en el mejor modelo)
    home_html = """
    <section class="hero-section" style="border-radius: 12px; margin-bottom: 2rem;">
        <div class="hero-content">
            <h1>Sistema de Clasificación de Transcripciones Médicas</h1>
            <p>Clasifica transcripciones médicas automáticamente por especialidad utilizando Machine Learning (NLP).</p>
            <div class="hero-stats">
                <div class="stat-item"><span class="stat-value">1,962</span><span class="stat-label">Transcripciones Entrenamiento</span></div>
                <div class="stat-item"><span class="stat-value">94.0%</span><span class="stat-label">Precisión del Modelo</span></div>
                <div class="stat-item"><span class="stat-value">10</span><span class="stat-label">Especialidades</span></div>
            </div>
        </div>
    </section>

    <div class="container" style="padding: 0;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1rem;">
            <div class="card">
                <div class="card-header">
                    <h2>🩺 Instrucciones de Uso</h2>
                </div>
                <div style="padding-top: 1rem; color: var(--text-color); font-size: 0.95rem; line-height: 1.6;">
                    <p>Bienvenido al prototipo de interfaz para el endpoint de clasificación. Para evaluar el funcionamiento real:</p>
                    <ol style="margin-top: 0.5rem; margin-left: 1.5rem; margin-bottom: 0;">
                        <li style="margin-bottom: 0.5rem;">Dirígete a la pestaña <strong>Clasificar</strong>.</li>
                        <li style="margin-bottom: 0.5rem;">Pega el texto de una transcripción médica, nota clínica o síntomas en inglés.</li>
                        <li style="margin-bottom: 0.5rem;">Haz clic en <em>Clasificar</em> y la API procesará tu texto en tiempo real.</li>
                    </ol>
                </div>
            </div>

            <div class="card tips-card" style="background-color: #fff3cd; border: 1px solid #ffeeba; border-left: 5px solid #ffc107;">
                <h3 style="color: #856404; margin-bottom: 0.75rem; font-size: 1.1rem;">⚠️ Descargo de Responsabilidad Médico</h3>
                <p style="color: #856404; font-size: 0.9rem; line-height: 1.5; text-align: justify;">
                    <strong>Aviso importante:</strong> Esta herramienta es un prototipo con fines investigativos diseñado como apoyo analítico mediante Inteligencia Artificial.
                    Los resultados generados son estimaciones probabilísticas y <strong>bajo ninguna circunstancia constituyen un diagnóstico definitivo ni deben sustituir el criterio, evaluación o toma de decisiones de un profesional de la salud.</strong>
                </p>
            </div>
        </div>
    </div>
    """
    st.markdown("".join([line.strip() for line in home_html.split("\n")]), unsafe_allow_html=True)

# ==========================================
# PESTAÑA 2: CLASIFICAR
# ==========================================
with tab_clasificar:
    st.markdown('<div class="container" style="padding: 0;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        input_html = """
        <div class="card input-section" style="margin-bottom: 1rem;">
            <div class="card-header">
                <h2>📝 Entrada</h2>
            </div>
        </div>
        """
        st.markdown("".join([line.strip() for line in input_html.split("\n")]), unsafe_allow_html=True)
        
        input_text = st.text_area(
            "Texto médico",
            height=250,
            placeholder="Pegar o escribir la transcripción médica aquí...\nEjemplo:\nPaciente masculino de 58 años con dolor torácico y disnea...",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            submit_btn = st.button("🔍 Clasificar", type="primary", use_container_width=True)
        with col_btn2:
            limpiar_btn = st.button("🗑️ Limpiar", use_container_width=True)
            
        if limpiar_btn:
            pass

    with col2:
        results_header_html = """
        <div class="card" style="margin-bottom: 1rem;">
            <div class="card-header">
                <h2>📊 Resultados</h2>
            </div>
        </div>
        """
        st.markdown("".join([line.strip() for line in results_header_html.split("\n")]), unsafe_allow_html=True)
        
        if submit_btn and input_text:
            with st.spinner("Clasificando..."):
                data = None
                try:
                    # Enviar request a la API
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"transcription": input_text},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                    else:
                        st.toast(f"⚠️ Error en API ({response.status_code}). Usando mock data...")
                        data = {"specialty": "Cardiology", "confidence": 0.89, "top_3": [{"specialty": "Neurology", "probability": 0.08}, {"specialty": "Orthopedic", "probability": 0.03}]}
                except Exception as e:
                    st.toast("⚠️ Error de conexión a la API. Usando mock data...")
                    data = {"specialty": "Cardiology", "confidence": 0.89, "top_3": [{"specialty": "Neurology", "probability": 0.08}, {"specialty": "Orthopedic", "probability": 0.03}]}
                
                if data:
                    main_specialty = data["specialty"]
                    # Convertir confianza a porcentaje entero
                    confidence_pct = int(data["confidence"] * 100)
                    
                    top_3 = data.get("top_3", [])
                    
                    # Construir visualización de resultados "aplastado"
                    html_results = f"""
                        <div class="card" style="margin-top:-1rem;">
                            <div style="margin-bottom: 1.5rem;">
                                <p style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.5rem;">Especialidad predicha</p>
                                <span class="result-badge" style="display:inline-block; padding:0.75rem 1.5rem; background:var(--primary-color); color:white; border-radius:12px; font-weight:600; font-size:1.1rem;">{main_specialty}</span>
                            </div>
                            
                            <div class="confidence-section" style="margin: 1.5rem 0;">
                                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                    <p style="font-size: 0.875rem; color: var(--text-muted);">Confianza del modelo</p>
                                    <span class="confidence-value" style="font-size: 2rem; font-weight: 700; color: var(--primary-color);">{confidence_pct}<span style="font-size: 1rem;">%</span></span>
                                </div>
                                <div class="confidence-bar" style="height: 16px; background: var(--bg-input); border-radius: 16px; overflow: hidden; margin-top: 0.5rem;">
                                    <div class="confidence-fill" style="height: 100%; background: linear-gradient(90deg, var(--primary-color), var(--accent-teal)); border-radius: 16px; width: {confidence_pct}%;"></div>
                                </div>
                            </div>
                            
                            <div class="top-specialties" style="margin: 1.5rem 0;">
                                <h3 style="font-size: 0.875rem; margin-bottom: 1rem;">Top 3 especialidades</h3>
                    """
                    
                    color_vars = ["var(--primary-color)", "var(--accent-purple)", "var(--accent-orange)"]
                    
                    for i, pred in enumerate(top_3):
                        spec = pred["specialty"]
                        conf = int(pred["probability"] * 100)
                        color_var = color_vars[i] if i < len(color_vars) else "var(--primary-color)"
                        
                        html_results += f"""
                            <div class="specialty-bar" style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                <span class="specialty-name" style="width: 120px; font-weight: 500; font-size: 0.875rem;">{spec}</span>
                                <div class="specialty-track" style="flex: 1; height: 16px; background: var(--bg-input); border-radius: 8px; overflow: hidden;">
                                    <div class="specialty-fill" style="height: 100%; border-radius: 8px; background: {color_var}; width: {conf}%;"></div>
                                </div>
                                <span class="specialty-value" style="width: 40px; text-align: right; font-weight: 600; font-size: 0.875rem;">{conf}%</span>
                            </div>
                        """
                        
                    html_results += """
                            </div>
                        </div>
                    """
                    
                    html_limpio = "".join([line.strip() for line in html_results.split("\n")])
                    st.markdown(html_limpio, unsafe_allow_html=True)
        elif submit_btn and not input_text:
            st.warning("⚠️ Por favor ingresa el texto de una transcripción médica.")
        else:
            st.info("Ingresa un texto en el panel izquierdo y haz clic en 'Clasificar' para inferir resultados dinámicos de la API.")

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PESTAÑA 3: ANALYTICS (datos reales de /dashboard-data)
# ==========================================
DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN", "Team12Dashboard")

with tab_analytics:
    st.markdown('<div class="container" style="padding: 0;">', unsafe_allow_html=True)

    analytics_header = """
    <div class="card" style="margin-bottom: 1rem;">
        <div class="card-header">
            <h2>📈 Monitoreo de Producción</h2>
        </div>
        <p style="color: var(--text-muted); font-size: 0.9rem;">Datos históricos reales de predicciones del API en producción.</p>
    </div>
    """
    st.markdown("".join([line.strip() for line in analytics_header.split("\n")]), unsafe_allow_html=True)

    # Fetch dashboard data
    @st.cache_data(ttl=30)
    def fetch_dashboard_data():
        try:
            resp = requests.get(
                f"{API_URL}/dashboard-data",
                headers={"Authorization": f"Bearer {DASHBOARD_TOKEN}"},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
            return []
        except Exception:
            return []

    raw_data = fetch_dashboard_data()

    if not raw_data:
        st.warning("⚠️ No hay datos disponibles. Verifica que el API esté activa y el token sea correcto.")
    else:
        # Parse response bodies to extract predictions
        records = []
        for entry in raw_data:
            try:
                resp_body = json.loads(entry.get("response_body", "{}"))
                req_body = json.loads(entry.get("request_body", "{}"))
                records.append({
                    "timestamp": entry.get("timestamp", ""),
                    "specialty": resp_body.get("specialty", "Unknown"),
                    "confidence": resp_body.get("confidence", 0),
                    "response_time_ms": entry.get("response_time_ms", 0),
                    "status_code": entry.get("status_code", 0),
                    "transcription": req_body.get("transcription", "")[:100] + "...",
                })
            except (json.JSONDecodeError, TypeError):
                continue

        if not records:
            st.warning("⚠️ No se pudieron parsear los datos de predicciones.")
        else:
            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

            # --- KPIs ---
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                st.metric("Total Predicciones", f"{len(df):,}")
            with kpi_cols[1]:
                st.metric("Confianza Promedio", f"{df['confidence'].mean():.0%}")
            with kpi_cols[2]:
                st.metric("Tiempo Resp. Promedio", f"{df['response_time_ms'].mean():.1f} ms")
            with kpi_cols[3]:
                st.metric("Especialidades Únicas", f"{df['specialty'].nunique()}")

            st.markdown("---")

            # --- Charts: 2 columns ---
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Pie chart - Distribución de especialidades
                specialty_counts = df["specialty"].value_counts().reset_index()
                specialty_counts.columns = ["Especialidad", "Cantidad"]
                fig_pie = px.pie(
                    specialty_counts, names="Especialidad", values="Cantidad",
                    title="Distribución de Especialidades Predichas",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4
                )
                fig_pie.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                # Histogram - Distribución de confianza
                fig_conf = px.histogram(
                    df, x="confidence", nbins=20,
                    title="Distribución de Confianza del Modelo",
                    labels={"confidence": "Confianza", "count": "Frecuencia"},
                    color_discrete_sequence=["#7c3aed"]
                )
                fig_conf.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    margin=dict(t=40, b=20, l=20, r=20),
                    xaxis=dict(tickformat=".0%")
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            # --- Second row of charts ---
            chart_col3, chart_col4 = st.columns(2)

            with chart_col3:
                # Bar chart - Confianza promedio por especialidad
                avg_conf = df.groupby("specialty")["confidence"].mean().sort_values(ascending=True).reset_index()
                avg_conf.columns = ["Especialidad", "Confianza Promedio"]
                fig_bar = px.bar(
                    avg_conf, x="Confianza Promedio", y="Especialidad",
                    orientation="h",
                    title="Confianza Promedio por Especialidad",
                    color="Confianza Promedio",
                    color_continuous_scale="Viridis"
                )
                fig_bar.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    margin=dict(t=40, b=20, l=20, r=20),
                    xaxis=dict(tickformat=".0%"),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with chart_col4:
                # Scatter - Tiempo de respuesta
                fig_time = px.scatter(
                    df, x="timestamp", y="response_time_ms",
                    color="specialty",
                    title="Tiempo de Respuesta por Predicción",
                    labels={"response_time_ms": "Tiempo (ms)", "timestamp": "Fecha/Hora"},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_time.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_time, use_container_width=True)

            # --- Data table ---
            st.markdown("---")
            st.markdown("### 📋 Registro de Predicciones")
            display_df = df[["timestamp", "specialty", "confidence", "response_time_ms", "transcription"]].copy()
            display_df.columns = ["Fecha/Hora", "Especialidad", "Confianza", "Tiempo (ms)", "Transcripción"]
            display_df["Confianza"] = display_df["Confianza"].apply(lambda x: f"{x:.0%}")
            display_df["Tiempo (ms)"] = display_df["Tiempo (ms)"].apply(lambda x: f"{x:.1f}")
            st.dataframe(display_df, use_container_width=True, height=300)

            # Refresh button
            if st.button("🔄 Actualizar Datos", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
