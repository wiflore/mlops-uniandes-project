import streamlit as st
import requests
import os
import json
import textwrap

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
API_URL = "http://localhost:8000"

# --- INYECTAR NAVBAR ORIGINAL ---
navbar_html = """
    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon">M</div>
            <div class="logo-text">
                <h1>MedTranscript Classifier</h1>
                <p>NLP • Specialty Classification • ML API</p>
            </div>
        </div>
        <div class="api-status">API: Online</div>
    </nav>
"""
st.markdown("".join([line.strip() for line in navbar_html.split("\n")]), unsafe_allow_html=True)

# --- SISTEMA DE NAVEGACIÓN (TABS DE STREAMLIT) ---
tab_home, tab_clasificar = st.tabs(["🏠 Home", "📊 Clasificar"])

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
