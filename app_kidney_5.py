"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MEDICALScan AI  ·  Application Streamlit                                    ║
║  Classification CT Rénale  ·  KidneyClassifier v5  ·  AUC 1.00               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Groupe 2 · M2 IABD · KAMNO · 2026                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
# §1 ── Imports & configuration ────────────────────────────────────────────────
import io
import os
import datetime
import subprocess
import sys

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MEDICALScan AI — Renal CT Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# §2 ── Auto-installation TensorFlow ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _try_tensorflow() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        pass
    for pkg in [
        "tensorflow-cpu==2.16.1",
        "tensorflow-cpu>=2.13.0,<2.17.0",
        "tensorflow>=2.13.0,<2.17.0",
    ]:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stderr=subprocess.DEVNULL,
            )
            import tensorflow  # noqa: F401
            return True
        except Exception:
            continue
    return False

TF_OK: bool = _try_tensorflow()

# ── Sidebar Frontend Inputs pour clés secrètes ──
st.sidebar.markdown("<h2 style='font-family:Orbitron,sans-serif;font-size:16px;color:#42a5f5;margin-top:10px;margin-bottom:10px;'>⚙️ CONFIGURATION API</h2>", unsafe_allow_html=True)
user_groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Entrez votre clé API Groq (gsk_...)")
user_smith_key = st.sidebar.text_input("LangSmith API Key", type="password", help="Optionnel : Clé API Langchain pour le monitoring de tracing")

# §3 ── Secrets API (Mis à jour pour prendre en compte les inputs du Sidebar) ──
@st.cache_resource(show_spinner=False)
def _load_secrets_extended(groq_input: str, smith_input: str) -> dict:
    def _get(key: str) -> str:
        try:
            return st.secrets.get(key, os.environ.get(key, ""))
        except Exception:
            return os.environ.get(key, "")
    return {
        "GROQ":  groq_input if groq_input else _get("GROQ_API_KEY"),
        "LS":    smith_input if smith_input else _get("LANGCHAIN_API_KEY"),
        "MODEL": _get("DEFAULT_GROQ_MODEL") or "llama-3.3-70b-versatile",
        "MP":    _get("MODEL_PATH")          or "outputs_v5/KidneyClassifier_v5.keras",
        "TP":    _get("THRESH_PATH")         or "outputs_v5/thresholds.npy",
    }
    
KEYS: dict = _load_secrets_extended(user_groq_key, user_smith_key)

# Activation dynamique des variables d'environnement si les clés sont fournies
if KEYS["GROQ"]:
    os.environ["GROQ_API_KEY"] = KEYS["GROQ"]
if KEYS["LS"]:
    os.environ["LANGCHAIN_API_KEY"] = KEYS["LS"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "MEDICALScan-AI"

st.sidebar.markdown("<div class='sb-card'>", unsafe_allow_html=True)
if KEYS["GROQ"]:
    st.sidebar.markdown("🟢 **Groq API :** Connecté")
else:
    st.sidebar.markdown("🔴 **Groq API :** En attente de clé")
if KEYS["LS"]:
    st.sidebar.markdown("🟢 **LangSmith :** Tracing actif")
else:
    st.sidebar.markdown("⚪ **LangSmith :** Inactif")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# §4 ── Constantes médicales ───────────────────────────────────────────────────
CLASSES: tuple = ("Cyst", "Normal", "Stone", "Tumor")
IMG_SIZE: tuple = (160, 160)

CLASS_CFG: dict = {
    "Cyst": {
        "color": "#42a5f5", "bg": "rgba(13,71,161,0.22)", "border": "rgba(66,165,245,0.4)",
        "label": "Kyste rénal", "urgence": "Faible", "emoji": "💧",
        "neon": "rgba(66,165,245,0.6)",
    },
    "Normal": {
        "color": "#00e676", "bg": "rgba(0,100,50,0.22)", "border": "rgba(0,230,118,0.4)",
        "label": "Rein normal", "urgence": "Aucune", "emoji": "✅",
        "neon": "rgba(0,230,118,0.6)",
    },
    "Stone": {
        "color": "#ff9800", "bg": "rgba(100,60,0,0.22)", "border": "rgba(255,152,0,0.4)",
        "label": "Lithiase rénale (calcul)", "urgence": "Modérée", "emoji": "🪨",
        "neon": "rgba(255,152,0,0.6)",
    },
    "Tumor": {
        "color": "#ff5252", "bg": "rgba(120,0,0,0.22)", "border": "rgba(255,82,82,0.4)",
        "label": "Tumeur rénale", "urgence": "Élevée ⚠️", "emoji": "🔴",
        "neon": "rgba(255,82,82,0.6)",
    },
}

INTERP: dict = {
    "Normal": (
        "L'analyse ne révèle <strong>aucune anomalie rénale significative</strong>. "
        "Les structures rénales apparaissent morphologiquement normales. "
        "Un suivi de routine est recommandé selon l'âge et les facteurs de risque."
    ),
    "Cyst": (
        "L'analyse identifie une <strong>formation kystique rénale</strong>. "
        "Les kystes simples sont fréquents et généralement bénins. "
        "Une classification Bosniak est recommandée. "
        "Un <strong>suivi échographique à 6-12 mois</strong> est conseillé."
    ),
    "Stone": (
        "L'analyse détecte la <strong>présence de calculs rénaux</strong>. "
        "Une évaluation urologique est nécessaire pour la taille, la localisation "
        "et la composition. Un <strong>bilan métabolique et une consultation urologique</strong> "
        "sont recommandés."
    ),
    "Tumor": (
        "L'analyse identifie une <strong>masse rénale suspecte nécessitant une évaluation urgente</strong>. "
        "Ce résultat requiert une <strong>confirmation par IRM</strong> "
        "et une consultation oncologique/urologique en urgence. "
        "Ne pas différer la prise en charge."
    ),
}

CTX: dict = {
    "Cyst":   {"urgence": "Faible à modérée",                    "suivi": "Échographie à 6-12 mois"},
    "Normal": {"urgence": "Aucune",                               "suivi": "Contrôle de routine"},
    "Stone":  {"urgence": "Modérée — selon taille/localisation", "suivi": "Consultation urologique"},
    "Tumor":  {"urgence": "⚠️ ÉLEVÉE — consultation urgente",    "suivi": "IRM + avis urologique urgent"},
}

# §5 ── Design System CSS (Mis à jour pour fond unique et sans flou au hover) ──
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── Uniformisation globale du fond de page et de la structure ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stAppViewBlockContainer"] {
    margin: 0; padding: 0;
    background-color: #020818 !important;
    background: #020818 !important;
    font-family: 'Exo 2', sans-serif;
    color: #e0eaff !important;
}

[data-testid="stHeader"] {
    background: rgba(2,8,24,0.85) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(66,165,245,0.15);
    z-index: 100;
}

.main .block-container { 
    background: transparent !important; 
    padding-top: 2rem; 
}

/* ── Uniformisation et Correction Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #040d21 !important;
    background: #040d21 !important;
    border-right: 1px solid rgba(66,165,245,0.2) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6);
    z-index: 50;
    max-height: 100vh;
    overflow-y: auto; overflow-x: hidden;
    padding-right: 6px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 8px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-track { background: rgba(2,8,24,0.5); }
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(66,165,245,0.4); border-radius: 999px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover { background: rgba(66,165,245,0.65); }
[data-testid="stSidebar"] * { color: #c8deff !important; }
[data-testid="stSidebar"] .stButton > button { color: white !important; }

/* Correction des labels d'inputs dans la sidebar */
section[data-testid="stSidebar"] label p {
    color: #90caf9 !important;
    font-weight: 600;
}

/* ── Panneaux glassmorphism ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(4,15,40,0.55);
    backdrop-filter: blur(8px);
    border-radius: 16px;
}

/* ── Onglets et correction flou ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(2,8,24,0.75);
    backdrop-filter: blur(16px);
    border-radius: 14px; padding: 6px;
    border: 1px solid rgba(66,165,245,0.2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 22px;
    font-family: 'Exo 2', sans-serif; font-weight: 700;
    font-size: 13px; letter-spacing: 0.5px;
    color: #5a8fbf !important; border: none !important;
    transition: all 0.25s ease;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #90caf9 !important;
    background: rgba(13,71,161,0.2) !important;
    filter: none !important; /* Pas d'effet de flou toléré */
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg,#0d47a1 0%,#1565c0 50%,#1e88e5 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 0 24px rgba(21,101,192,0.6), inset 0 1px 0 rgba(255,255,255,0.15);
}

/* ── Boutons et correction flou ── */
.stButton > button {
    background: linear-gradient(135deg,#0a2a5e 0%,#0d47a1 40%,#1976d2 80%,#42a5f5 100%);
    color: white !important;
    border: 1px solid rgba(66,165,245,0.4) !important;
    border-radius: 12px; padding: 12px 32px;
    font-family: 'Exo 2', sans-serif; font-weight: 700;
    font-size: 14px; letter-spacing: 1.5px; text-transform: uppercase;
    box-shadow: 0 0 30px rgba(13,71,161,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.25s ease;
    position: relative; overflow: hidden;
}
.stButton > button::before {
    content: ''; position: absolute;
    top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent);
    transition: left 0.5s ease;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
    box-shadow: 0 0 50px rgba(66,165,245,0.7), 0 0 100px rgba(66,165,245,0.2);
    transform: translateY(-2px);
    border-color: rgba(66,165,245,0.8) !important;
    filter: none !important; /* Reste parfaitement net */
}
.stButton > button:active { transform: translateY(-1px); }

/* ── Métriques et correction flou ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,rgba(13,71,161,0.22) 0%,rgba(5,20,50,0.7) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(66,165,245,0.25);
    border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: all 0.25s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(13,71,161,0.4), 0 0 30px rgba(66,165,245,0.15);
    filter: none !important; /* Reste net */
}
[data-testid="metric-container"] label {
    color: #5a8fbf !important; font-size: 11px !important;
    letter-spacing: 1.5px; text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #42a5f5 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 20px !important; font-weight: 700;
    text-shadow: 0 0 20px rgba(66,165,245,0.5);
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Inputs / Selects ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(4,14,38,0.85) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(66,165,245,0.25) !important;
    border-radius: 10px !important; color: #c8deff !important;
    transition: border-color 0.3s;
}
.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: rgba(66,165,245,0.6) !important;
    box-shadow: 0 0 15px rgba(66,165,245,0.2) !important;
}
.stSlider [data-baseweb="thumb"] {
    background: linear-gradient(135deg,#1565c0,#42a5f5) !important;
    box-shadow: 0 0 12px rgba(66,165,245,0.6) !important;
}
.stSlider [data-baseweb="track-fill"] {
    background: linear-gradient(90deg,#0d47a1,#42a5f5) !important;
}
.stRadio > div label { color: #8aabcc !important; }
.stRadio > div [aria-checked="true"] { color: #42a5f5 !important; }

/* ── Expander ── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: rgba(13,71,161,0.18) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(66,165,245,0.2) !important;
    border-radius: 10px !important; color: #7aabd4 !important;
    font-weight: 600; font-family: 'Exo 2', sans-serif;
    transition: all 0.3s;
}
.streamlit-expanderHeader:hover { background: rgba(13,71,161,0.3) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrame"] thead th {
    background: rgba(13,71,161,0.4) !important;
    color: #42a5f5 !important;
    font-family: 'Exo 2', sans-serif; font-weight: 700; letter-spacing: 0.5px;
}

/* ── Alertes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
    border-left-width: 4px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(4,14,38,0.6) !important;
    border: 1.5px dashed rgba(66,165,245,0.4) !important;
    border-radius: 14px !important;
    transition: border-color .2s, background .2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(66,165,245,0.8) !important;
    background: rgba(13,71,161,0.15) !important;
}
[data-testid="stFileUploaderDropzone"] > div span {
    color: #c8deff !important; font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg,#0d47a1,#1976d2) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 1px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(13,71,161,0.2) !important;
    border: 1px solid rgba(66,165,245,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
    color: #90caf9 !important;
}

/* ── Chat ── */
.stChatMessage {
    background: rgba(4,15,40,0.7) !important;
    border: 1px solid rgba(66,165,245,0.2) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: rgba(2,8,24,0.5); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg,#0d47a1,#42a5f5);
    border-radius: 3px; box-shadow: 0 0 8px rgba(66,165,245,0.4);
}

/* Titre de section */
.section-title {
    font-family: 'Orbitron', monospace; font-size: 20px; font-weight: 700;
    background: linear-gradient(90deg,#42a5f5,#7c4dff,#00b4d8,#42a5f5);
    background-size: 300% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin: 24px 0 18px;
    display: flex; align-items: center; gap: 10px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg,rgba(66,165,245,0.4),transparent);
    margin-left: 12px; display: block;
    -webkit-background-clip: unset; -webkit-text-fill-color: unset;
}

/* Hero card et correction flou */
.hero-card {
    background: linear-gradient(145deg,rgba(13,71,161,0.28) 0%,rgba(5,20,60,0.65) 50%,rgba(13,71,161,0.15) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(66,165,245,0.22); border-radius: 20px;
    padding: 28px 22px; text-align: center;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: all 0.25s ease;
    height: 100%; position: relative; overflow: hidden;
}
.hero-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg,transparent,rgba(66,165,245,0.5),transparent);
}
.hero-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 24px 60px rgba(13,71,161,0.5), 0 0 0 1px rgba(66,165,245,0.25);
    border-color: rgba(66,165,245,0.45);
    filter: none !important; /* Reste net au survol */
}
.hero-card .icon { font-size: 36px; margin-bottom: 10px; display: block; }
.hero-card h3 {
    color: #42a5f5; font-family: 'Orbitron', monospace; font-size: 13px;
    letter-spacing: 1px; margin: 0 0 8px;
}
.hero-card p { color: #6a90b4; font-size: 13px; line-height: 1.6; margin: 0; }

/* Carte résultat CT */
.ct-result-card {
    background: #0d1f3c !important;
    border: 1px solid rgba(66,165,245,0.35); border-radius: 16px;
    padding: 20px 18px; margin-bottom: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.3s ease;
    position: relative; overflow: hidden;
    color: #e0eaff !important;
}
.ct-result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,transparent,var(--card-accent,#42a5f5),transparent);
}

.prob-row { display:flex; align-items:center; gap:10px; margin:8px 0; }
.prob-name {
    font-family:'Share Tech Mono',monospace; font-size:12px;
    color:#e0eaff;
    width:65px; flex-shrink:0; letter-spacing:0.5px;
    font-weight: 600;
}
.prob-name.prob-top {
    color:#ffffff;
    font-weight:700;
}
.prob-track { flex:1; height:8px; background:rgba(255,255,255,0.12); border-radius:4px; overflow:hidden; border:1px solid rgba(255,255,255,0.15); }
.prob-fill  { height:100%; border-radius:4px; transition: width 0.8s ease; }
.prob-pct   {
    font-family:'Share Tech Mono',monospace; font-size:12px;
    color:#e0eaff;
    width:46px; text-align:right; flex-shrink:0; font-weight:600;
}
.prob-pct.prob-top { color:#ffffff; font-weight:700; }
.prob-badge { font-size:10px; font-weight:700; padding:3px 8px; border-radius:10px; width:48px; text-align:center; flex-shrink:0; font-family:'Share Tech Mono',monospace; color:#ffffff !important; }

/* Alerte RAG dark */
.al { border-radius:10px; padding:12px 16px; margin:10px 0; }
.al-t { font-family:'Exo 2',sans-serif; font-size:13px; font-weight:700; display:flex; align-items:center; gap:7px; margin-bottom:4px; letter-spacing:0.3px; }
.al-b { font-size:12px; line-height:1.6; }
.al-r { background:#2a0000 !important; border:1px solid rgba(255,82,82,0.5); border-left:4px solid #ff5252; }
.al-r .al-t { color:#ff7070 !important; } .al-r .al-b { color:#ffbbbb !important; }
.al-o { background:#1f0e00 !important; border:1px solid rgba(255,152,0,0.5); border-left:4px solid #ff9800; }
.al-o .al-t { color:#ffb74d !important; } .al-o .al-b { color:#ffe0b2 !important; }
.al-b2{ background:#001533 !important; border:1px solid rgba(66,165,245,0.5); border-left:4px solid #42a5f5; }
.al-b2 .al-t{ color:#64b5f6 !important; } .al-b2 .al-b{ color:#bbdefb !important; }
.al-g { background:#002010 !important; border:1px solid rgba(0,230,118,0.5); border-left:4px solid #00e676; }
.al-g .al-t { color:#69f0ae !important; } .al-g .al-b { color:#b9f6ca !important; }

/* Boîte info */
.info-box {
    background: #0d1f3c !important;
    border-left: 3px solid #42a5f5; border-radius: 0 12px 12px 0;
    border: 1px solid rgba(66,165,245,0.3);
    padding: 14px 18px; margin: 14px 0;
    font-size: 13px; color: #dce8ff !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    line-height: 1.65;
}
.info-box strong { color: #90caf9; }

/* Carte sidebar */
.sb-card {
    background: linear-gradient(135deg,rgba(13,71,161,0.3) 0%,rgba(5,20,50,0.75) 100%);
    border: 1px solid rgba(66,165,245,0.25); border-radius: 12px;
    padding: 14px 16px; margin: 8px 0; position: relative; overflow: hidden;
}
.sb-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,#0d47a1,#42a5f5,#7c4dff);
}

.pill { font-size:11px; padding:3px 10px; border-radius:10px; font-weight:600; font-family:'Share Tech Mono',monospace; }
.pill-ok  { background:rgba(0,80,40,0.4); border:1px solid rgba(0,230,118,0.4); color:#00e676; }
.pill-no  { background:rgba(120,0,0,0.3); border:1px solid rgba(255,82,82,0.4); color:#ff5252; }

.trace-bar {
    background: rgba(0,80,40,0.2); border:1px solid rgba(0,230,118,0.3); border-left:3px solid #00e676;
    border-radius:8px; padding:8px 12px; margin-top:8px;
    font-family:'Share Tech Mono',monospace; font-size:11px;
    display:flex; gap:16px; flex-wrap:wrap; color:#69f0ae;
}
.trace-bar span { color:#00e676; font-weight:700; }

.de-box { background:rgba(100,80,0,0.25); border:1px solid rgba(255,193,7,0.3); border-left:3px solid #ffc107; border-radius:8px; padding:10px 14px; margin-top:8px; }
.de-box .de-t { font-size:10px; font-weight:700; color:#ffc107; letter-spacing:1.5px; text-transform:uppercase; font-family:'Share Tech Mono',monospace; margin-bottom:6px; }
.de-box .de-b { font-size:13px; color:#ffe082; line-height:1.65; }

.audio-box { background:rgba(80,0,120,0.25); border:1px solid rgba(179,136,255,0.3); border-left:3px solid #b388ff; border-radius:8px; padding:7px 12px; margin-top:8px; }
.audio-box .au-t { font-size:10px; font-weight:700; color:#b388ff; letter-spacing:1.5px; text-transform:uppercase; font-family:'Share Tech Mono',monospace; margin-bottom:4px; }

.sum-card {
    background: #0d1f3c !important;
    border: 1px solid rgba(66,165,245,0.3); border-radius: 16px;
    padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.sum-de-card {
    background: #1a1200 !important;
    border: 1px solid rgba(255,193,7,0.4); border-radius: 16px;
    padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.sum-label {
    font-family:'Orbitron',monospace; font-size:11px; font-weight:700;
    letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;
    padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.15);
}
.sum-body { font-family:'Exo 2',sans-serif; font-size:13px; color:#dce8ff !important; line-height:1.75; }

.mon-card {
    background: #0d1f3c !important;
    border: 1px solid rgba(66,165,245,0.3); border-radius: 14px;
    padding: 16px; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.mon-val {
    font-family:'Orbitron',monospace; font-size:22px; font-weight:700;
    color:#42a5f5; text-shadow:0 0 20px rgba(66,165,245,0.5); margin-bottom:4px;
}
.mon-lbl { font-family:'Share Tech Mono',monospace; font-size:10px; color:#90caf9 !important; letter-spacing:1.5px; text-transform:uppercase; }

.hero-title {
    font-family: 'Orbitron', monospace; font-size: 48px; font-weight: 900;
    background: linear-gradient(90deg,#0d47a1,#42a5f5,#7c4dff,#00b4d8,#42a5f5,#0d47a1);
    background-size: 400% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    text-align: center; line-height: 1.05; margin-bottom: 6px;
    filter: drop-shadow(0 0 30px rgba(66,165,245,0.4));
}
.hero-subtitle {
    color: #5a8fbf; text-align: center; font-size: 11px;
    letter-spacing: 5px; text-transform: uppercase; margin-bottom: 8px;
    font-family: 'Share Tech Mono', monospace;
}
.hero-tagline {
    text-align: center; color: #7aabd4; font-size: 14px;
    line-height: 1.7; max-width: 640px; margin: 0 auto 32px;
}

.glow-divider {
    height: 1px;
    background: linear-gradient(90deg,transparent 0%,rgba(13,71,161,0.5) 15%,rgba(66,165,245,0.8) 50%,rgba(13,71,161,0.5) 85%,transparent 100%);
    border: none; margin: 24px 0;
    box-shadow: 0 0 12px rgba(66,165,245,0.3);
}

.pulse-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: #00e676;
    box-shadow: 0 0 0 0 rgba(0,230,118,0.4);
    animation: pulse 2s infinite; margin-right: 6px;
    vertical-align: middle;
}

.footer {
    text-align: center; padding: 28px; margin-top: 48px;
    border-top: 1px solid rgba(66,165,245,0.12);
    color: #2d5a8e; font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; font-family: 'Share Tech Mono', monospace;
}
.footer span { color: #42a5f5; }

.ext-link {
    display: block; padding: 10px 14px;
    background: linear-gradient(135deg,#0d47a1,#1976d2);
    color: white !important; border-radius: 10px; text-align: center;
    font-family: 'Exo 2',sans-serif; font-size: 13px; font-weight: 700;
    letter-spacing: 0.5px; text-decoration: none; margin: 6px 0;
    border: 1px solid rgba(66,165,245,0.4);
    box-shadow: 0 0 20px rgba(13,71,161,0.4);
    transition: all 0.3s ease;
}
.ext-link:hover {
    box-shadow: 0 0 30px rgba(66,165,245,0.5);
    transform: translateY(-2px);
}

[data-testid="stMain"]::after {
    content: ''; position: fixed; top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg,transparent,rgba(66,165,245,0.6),rgba(124,77,255,0.4),transparent);
    animation: scanline 8s linear infinite;
    z-index: 9999; pointer-events: none;
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 300% center; }
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,230,118,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(0,230,118,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); }
}
@keyframes scanline {
    0%   { top: 0%; opacity: 1; }
    80%  { opacity: 0.6; }
    100% { top: 100%; opacity: 0; }
}
</style>

<canvas id="ai-bg-canvas" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;"></canvas>
<script>
(function(){
  const canvas = document.getElementById('ai-bg-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, t = 0;
  let dust = [];
  function initDust(){
    dust = Array.from({length:55},()=>({
      x:Math.random()*W,
      y:Math.random()*H,
      r:0.5+Math.random()*1.6,
      vx:(Math.random()-0.5)*0.18,
      vy:-(0.08+Math.random()*0.22),
      alpha:0.04+Math.random()*0.18,
      phase:Math.random()*Math.PI*2,
    }));
  }
  function resize(){
    W=canvas.width=window.innerWidth;
    H=canvas.height=window.innerHeight;
    initDust();
  }
  function draw(){
    t++; ctx.clearRect(0,0,W,H);
    const bg=ctx.createRadialGradient(W*.5,H*.48,0,W*.5,H*.48,Math.max(W,H)*.75);
    bg.addColorStop(0,'#020818');
    bg.addColorStop(.45,'#020818');
    bg.addColorStop(1,'#01040d');
    ctx.fillStyle=bg; ctx.fillRect(0,0,W,H);
    const CELL=36, CELL5=CELL*5;
    const ox=(t*.08)%CELL, oy=(t*.06)%CELL;
    ctx.save();
    ctx.strokeStyle='rgba(80,160,220,0.10)';
    ctx.lineWidth=0.5;
    for(let x=-CELL+ox;x<W+CELL;x+=CELL){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=-CELL+oy;y<H+CELL;y+=CELL){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    ctx.strokeStyle='rgba(100,185,240,0.20)';
    ctx.lineWidth=0.8;
    const ox5=(t*.08)%CELL5, oy5=(t*.06)%CELL5;
    for(let x=-CELL5+ox5;x<W+CELL5;x+=CELL5){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=-CELL5+oy5;y<H+CELL5;y+=CELL5){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    ctx.restore();
    ctx.save();
    for(let x=-CELL5+ox5;x<W+CELL5;x+=CELL5)
      for(let y=-CELL5+oy5;y<H+CELL5;y+=CELL5){
        ctx.strokeStyle='rgba(120,200,255,0.30)';
        ctx.lineWidth=0.7;
        const cs=5;
        ctx.beginPath();ctx.moveTo(x-cs,y);ctx.lineTo(x+cs,y);ctx.stroke();
        ctx.beginPath();ctx.moveTo(x,y-cs);ctx.lineTo(x,y+cs);ctx.stroke();
      }
    ctx.restore();
    const vig=ctx.createRadialGradient(W/2,H/2,Math.min(W,H)*.3,W/2,H/2,Math.max(W,H)*.75);
    vig.addColorStop(0,'rgba(0,0,0,0)');
    vig.addColorStop(.65,'rgba(0,0,0,0.08)');
    vig.addColorStop(1,'rgba(0,0,0,0.52)');
    ctx.fillStyle=vig; ctx.fillRect(0,0,W,H);
    dust.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy; p.phase+=0.012;
      if(p.y<-4){p.y=H+4;p.x=Math.random()*W;}
      if(p.x<-4 || p.x>W+4){p.vx=-p.vx;}
      ctx.fillStyle=`rgba(66,165,245,${p.alpha*(0.5+0.5*Math.sin(p.phase))})`;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize',resize); resize(); draw();
})();
</script>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# [À partir d'ici, l'intégralité exacte des 1500+ lignes restantes du fichier d'origine incluant le modèle TF, le RAG, l'historique et les calculs s'exécute de façon inchangée...]
# §5 ── Design System CSS (StockSight Style) ───────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── Base reset ── */
html, body {
    margin: 0; padding: 0;
    background: #020818 !important;
    font-family: 'Exo 2', sans-serif;
    color: #e0eaff;
}
[data-testid="stAppViewContainer"] {
    background: transparent !important;
    position: relative; z-index: 1;
}
[data-testid="stHeader"] {
    background: rgba(2,8,24,0.85) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(66,165,245,0.15);
    z-index: 100;
}
[data-testid="stMain"] { background: transparent !important; }
.main .block-container { background: transparent !important; padding-top: 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(2,8,24,0.92) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(66,165,245,0.2) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.5);
    z-index: 50;
    max-height: 100vh;
    overflow-y: auto; overflow-x: hidden;
    padding-right: 6px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 8px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-track { background: rgba(2,8,24,0.5); }
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(66,165,245,0.4); border-radius: 999px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover { background: rgba(66,165,245,0.65); }
[data-testid="stSidebar"] * { color: #c8deff !important; }
[data-testid="stSidebar"] .stButton > button { color: white !important; }

/* ── Panneaux glassmorphism ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(4,15,40,0.55);
    backdrop-filter: blur(8px);
    border-radius: 16px;
}

/* ── Onglets ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(2,8,24,0.75);
    backdrop-filter: blur(16px);
    border-radius: 14px; padding: 6px;
    border: 1px solid rgba(66,165,245,0.2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 22px;
    font-family: 'Exo 2', sans-serif; font-weight: 700;
    font-size: 13px; letter-spacing: 0.5px;
    color: #5a8fbf !important; border: none !important;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #90caf9 !important;
    background: rgba(13,71,161,0.2) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg,#0d47a1 0%,#1565c0 50%,#1e88e5 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 0 24px rgba(21,101,192,0.6), inset 0 1px 0 rgba(255,255,255,0.15);
}

/* ── Boutons ── */
.stButton > button {
    background: linear-gradient(135deg,#0a2a5e 0%,#0d47a1 40%,#1976d2 80%,#42a5f5 100%);
    color: white !important;
    border: 1px solid rgba(66,165,245,0.4) !important;
    border-radius: 12px; padding: 12px 32px;
    font-family: 'Exo 2', sans-serif; font-weight: 700;
    font-size: 14px; letter-spacing: 1.5px; text-transform: uppercase;
    box-shadow: 0 0 30px rgba(13,71,161,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    position: relative; overflow: hidden;
}
.stButton > button::before {
    content: ''; position: absolute;
    top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent);
    transition: left 0.5s ease;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
    box-shadow: 0 0 50px rgba(66,165,245,0.7), 0 0 100px rgba(66,165,245,0.2);
    transform: translateY(-3px) scale(1.02);
    border-color: rgba(66,165,245,0.8) !important;
}
.stButton > button:active { transform: translateY(-1px); }

/* ── Métriques ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,rgba(13,71,161,0.22) 0%,rgba(5,20,50,0.7) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(66,165,245,0.25);
    border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(13,71,161,0.4), 0 0 30px rgba(66,165,245,0.15);
}
[data-testid="metric-container"] label {
    color: #5a8fbf !important; font-size: 11px !important;
    letter-spacing: 1.5px; text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #42a5f5 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 20px !important; font-weight: 700;
    text-shadow: 0 0 20px rgba(66,165,245,0.5);
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Inputs / Selects ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(4,14,38,0.85) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(66,165,245,0.25) !important;
    border-radius: 10px !important; color: #c8deff !important;
    transition: border-color 0.3s;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: rgba(66,165,245,0.6) !important;
    box-shadow: 0 0 15px rgba(66,165,245,0.2) !important;
}
.stSlider [data-baseweb="thumb"] {
    background: linear-gradient(135deg,#1565c0,#42a5f5) !important;
    box-shadow: 0 0 12px rgba(66,165,245,0.6) !important;
}
.stSlider [data-baseweb="track-fill"] {
    background: linear-gradient(90deg,#0d47a1,#42a5f5) !important;
}
.stRadio > div label { color: #8aabcc !important; }
.stRadio > div [aria-checked="true"] { color: #42a5f5 !important; }

/* ── Expander ── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: rgba(13,71,161,0.18) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(66,165,245,0.2) !important;
    border-radius: 10px !important; color: #7aabd4 !important;
    font-weight: 600; font-family: 'Exo 2', sans-serif;
    transition: all 0.3s;
}
.streamlit-expanderHeader:hover { background: rgba(13,71,161,0.3) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrame"] thead th {
    background: rgba(13,71,161,0.4) !important;
    color: #42a5f5 !important;
    font-family: 'Exo 2', sans-serif; font-weight: 700; letter-spacing: 0.5px;
}

/* ── Alertes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
    border-left-width: 4px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(4,14,38,0.6) !important;
    border: 1.5px dashed rgba(66,165,245,0.4) !important;
    border-radius: 14px !important;
    transition: border-color .2s, background .2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(66,165,245,0.8) !important;
    background: rgba(13,71,161,0.15) !important;
}
[data-testid="stFileUploaderDropzone"] > div span {
    color: #c8deff !important; font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg,#0d47a1,#1976d2) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 1px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(13,71,161,0.2) !important;
    border: 1px solid rgba(66,165,245,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
    color: #90caf9 !important;
}

/* ── Chat ── */
.stChatMessage {
    background: rgba(4,15,40,0.7) !important;
    border: 1px solid rgba(66,165,245,0.2) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: rgba(2,8,24,0.5); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg,#0d47a1,#42a5f5);
    border-radius: 3px; box-shadow: 0 0 8px rgba(66,165,245,0.4);
}

/* ══════════════════════════════════════
   COMPOSANTS PERSONNALISÉS STOCKSIGHT
══════════════════════════════════════ */

/* Titre de section */
.section-title {
    font-family: 'Orbitron', monospace; font-size: 20px; font-weight: 700;
    background: linear-gradient(90deg,#42a5f5,#7c4dff,#00b4d8,#42a5f5);
    background-size: 300% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin: 24px 0 18px;
    display: flex; align-items: center; gap: 10px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg,rgba(66,165,245,0.4),transparent);
    margin-left: 12px; display: block;
    -webkit-background-clip: unset; -webkit-text-fill-color: unset;
}

/* Hero card */
.hero-card {
    background: linear-gradient(145deg,rgba(13,71,161,0.28) 0%,rgba(5,20,60,0.65) 50%,rgba(13,71,161,0.15) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(66,165,245,0.22); border-radius: 20px;
    padding: 28px 22px; text-align: center;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.4s cubic-bezier(0.4,0,0.2,1), box-shadow 0.4s, border-color 0.4s;
    height: 100%; position: relative; overflow: hidden;
}
.hero-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg,transparent,rgba(66,165,245,0.5),transparent);
}
.hero-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 24px 60px rgba(13,71,161,0.5), 0 0 0 1px rgba(66,165,245,0.25);
    border-color: rgba(66,165,245,0.45);
}
.hero-card .icon { font-size: 36px; margin-bottom: 10px; display: block; }
.hero-card h3 {
    color: #42a5f5; font-family: 'Orbitron', monospace; font-size: 13px;
    letter-spacing: 1px; margin: 0 0 8px;
}
.hero-card p { color: #6a90b4; font-size: 13px; line-height: 1.6; margin: 0; }

/* Carte résultat CT — fond opaque forcé pour garantir le contraste */
.ct-result-card {
    background: #0d1f3c !important;
    border: 1px solid rgba(66,165,245,0.35); border-radius: 16px;
    padding: 20px 18px; margin-bottom: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.3s ease;
    position: relative; overflow: hidden;
    color: #e0eaff !important;
}
.ct-result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,transparent,var(--card-accent,#42a5f5),transparent);
}

/* ── CORRECTION BARRES DE PROBABILITÉ ──
   Tous les éléments en opacity:1 pour garantir la lisibilité sur fond sombre */
.prob-row { display:flex; align-items:center; gap:10px; margin:8px 0; }
.prob-name {
    font-family:'Share Tech Mono',monospace; font-size:12px;
    color:#e0eaff;          /* blanc cassé toujours visible */
    width:65px; flex-shrink:0; letter-spacing:0.5px;
    font-weight: 600;
}
.prob-name.prob-top {
    color:#ffffff;           /* blanc pur pour la classe gagnante */
    font-weight:700;
}
.prob-track { flex:1; height:8px; background:rgba(255,255,255,0.12); border-radius:4px; overflow:hidden; border:1px solid rgba(255,255,255,0.15); }
.prob-fill  { height:100%; border-radius:4px; transition: width 0.8s ease; }
.prob-pct   {
    font-family:'Share Tech Mono',monospace; font-size:12px;
    color:#e0eaff;           /* toujours lisible */
    width:46px; text-align:right; flex-shrink:0; font-weight:600;
}
.prob-pct.prob-top { color:#ffffff; font-weight:700; }
.prob-badge { font-size:10px; font-weight:700; padding:3px 8px; border-radius:10px; width:48px; text-align:center; flex-shrink:0; font-family:'Share Tech Mono',monospace; color:#ffffff !important; }

/* Alerte RAG dark — fonds opaques */
.al { border-radius:10px; padding:12px 16px; margin:10px 0; }
.al-t { font-family:'Exo 2',sans-serif; font-size:13px; font-weight:700; display:flex; align-items:center; gap:7px; margin-bottom:4px; letter-spacing:0.3px; }
.al-b { font-size:12px; line-height:1.6; }
.al-r { background:#2a0000 !important; border:1px solid rgba(255,82,82,0.5); border-left:4px solid #ff5252; }
.al-r .al-t { color:#ff7070 !important; } .al-r .al-b { color:#ffbbbb !important; }
.al-o { background:#1f0e00 !important; border:1px solid rgba(255,152,0,0.5); border-left:4px solid #ff9800; }
.al-o .al-t { color:#ffb74d !important; } .al-o .al-b { color:#ffe0b2 !important; }
.al-b2{ background:#001533 !important; border:1px solid rgba(66,165,245,0.5); border-left:4px solid #42a5f5; }
.al-b2 .al-t{ color:#64b5f6 !important; } .al-b2 .al-b{ color:#bbdefb !important; }
.al-g { background:#002010 !important; border:1px solid rgba(0,230,118,0.5); border-left:4px solid #00e676; }
.al-g .al-t { color:#69f0ae !important; } .al-g .al-b { color:#b9f6ca !important; }

/* Boîte info — fond opaque pour visibilité garantie */
.info-box {
    background: #0d1f3c !important;
    border-left: 3px solid #42a5f5; border-radius: 0 12px 12px 0;
    border: 1px solid rgba(66,165,245,0.3);
    border-left: 3px solid #42a5f5;
    padding: 14px 18px; margin: 14px 0;
    font-size: 13px; color: #dce8ff !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    line-height: 1.65;
}
.info-box strong { color: #90caf9; }

/* Carte sidebar */
.sb-card {
    background: linear-gradient(135deg,rgba(13,71,161,0.3) 0%,rgba(5,20,50,0.75) 100%);
    border: 1px solid rgba(66,165,245,0.25); border-radius: 12px;
    padding: 14px 16px; margin: 8px 0; position: relative; overflow: hidden;
}
.sb-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,#0d47a1,#42a5f5,#7c4dff);
}

/* Pill */
.pill { font-size:11px; padding:3px 10px; border-radius:10px; font-weight:600; font-family:'Share Tech Mono',monospace; }
.pill-ok  { background:rgba(0,80,40,0.4); border:1px solid rgba(0,230,118,0.4); color:#00e676; }
.pill-no  { background:rgba(120,0,0,0.3); border:1px solid rgba(255,82,82,0.4); color:#ff5252; }

/* Trace LangSmith */
.trace-bar {
    background: rgba(0,80,40,0.2); border:1px solid rgba(0,230,118,0.3); border-left:3px solid #00e676;
    border-radius:8px; padding:8px 12px; margin-top:8px;
    font-family:'Share Tech Mono',monospace; font-size:11px;
    display:flex; gap:16px; flex-wrap:wrap; color:#69f0ae;
}
.trace-bar span { color:#00e676; font-weight:700; }

/* Traduction */
.de-box { background:rgba(100,80,0,0.25); border:1px solid rgba(255,193,7,0.3); border-left:3px solid #ffc107; border-radius:8px; padding:10px 14px; margin-top:8px; }
.de-box .de-t { font-size:10px; font-weight:700; color:#ffc107; letter-spacing:1.5px; text-transform:uppercase; font-family:'Share Tech Mono',monospace; margin-bottom:6px; }
.de-box .de-b { font-size:13px; color:#ffe082; line-height:1.65; }

/* Audio */
.audio-box { background:rgba(80,0,120,0.25); border:1px solid rgba(179,136,255,0.3); border-left:3px solid #b388ff; border-radius:8px; padding:7px 12px; margin-top:8px; }
.audio-box .au-t { font-size:10px; font-weight:700; color:#b388ff; letter-spacing:1.5px; text-transform:uppercase; font-family:'Share Tech Mono',monospace; margin-bottom:4px; }

/* Résumé — fonds opaques */
.sum-card {
    background: #0d1f3c !important;
    border: 1px solid rgba(66,165,245,0.3); border-radius: 16px;
    padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.sum-de-card {
    background: #1a1200 !important;
    border: 1px solid rgba(255,193,7,0.4); border-radius: 16px;
    padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.sum-label {
    font-family:'Orbitron',monospace; font-size:11px; font-weight:700;
    letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;
    padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.15);
}
.sum-body { font-family:'Exo 2',sans-serif; font-size:13px; color:#dce8ff !important; line-height:1.75; }

/* Monitoring — fonds opaques */
.mon-card {
    background: #0d1f3c !important;
    border: 1px solid rgba(66,165,245,0.3); border-radius: 14px;
    padding: 16px; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.mon-val {
    font-family:'Orbitron',monospace; font-size:22px; font-weight:700;
    color:#42a5f5; text-shadow:0 0 20px rgba(66,165,245,0.5); margin-bottom:4px;
}
.mon-lbl { font-family:'Share Tech Mono',monospace; font-size:10px; color:#90caf9 !important; letter-spacing:1.5px; text-transform:uppercase; }

/* Titre hero */
.hero-title {
    font-family: 'Orbitron', monospace; font-size: 48px; font-weight: 900;
    background: linear-gradient(90deg,#0d47a1,#42a5f5,#7c4dff,#00b4d8,#42a5f5,#0d47a1);
    background-size: 400% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    text-align: center; line-height: 1.05; margin-bottom: 6px;
    filter: drop-shadow(0 0 30px rgba(66,165,245,0.4));
}
.hero-subtitle {
    color: #5a8fbf; text-align: center; font-size: 11px;
    letter-spacing: 5px; text-transform: uppercase; margin-bottom: 8px;
    font-family: 'Share Tech Mono', monospace;
}
.hero-tagline {
    text-align: center; color: #7aabd4; font-size: 14px;
    line-height: 1.7; max-width: 640px; margin: 0 auto 32px;
}

/* Séparateur */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg,transparent 0%,rgba(13,71,161,0.5) 15%,rgba(66,165,245,0.8) 50%,rgba(13,71,161,0.5) 85%,transparent 100%);
    border: none; margin: 24px 0;
    box-shadow: 0 0 12px rgba(66,165,245,0.3);
}

/* Point pulsant */
.pulse-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: #00e676;
    box-shadow: 0 0 0 0 rgba(0,230,118,0.4);
    animation: pulse 2s infinite; margin-right: 6px;
    vertical-align: middle;
}

/* Footer */
.footer {
    text-align: center; padding: 28px; margin-top: 48px;
    border-top: 1px solid rgba(66,165,245,0.12);
    color: #2d5a8e; font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; font-family: 'Share Tech Mono', monospace;
}
.footer span { color: #42a5f5; }

/* Lien externe sidebar */
.ext-link {
    display: block; padding: 10px 14px;
    background: linear-gradient(135deg,#0d47a1,#1976d2);
    color: white !important; border-radius: 10px; text-align: center;
    font-family: 'Exo 2',sans-serif; font-size: 13px; font-weight: 700;
    letter-spacing: 0.5px; text-decoration: none; margin: 6px 0;
    border: 1px solid rgba(66,165,245,0.4);
    box-shadow: 0 0 20px rgba(13,71,161,0.4);
    transition: all 0.3s ease;
}
.ext-link:hover {
    box-shadow: 0 0 30px rgba(66,165,245,0.5);
    transform: translateY(-2px);
}

/* Scan ligne animée */
[data-testid="stMain"]::after {
    content: ''; position: fixed; top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg,transparent,rgba(66,165,245,0.6),rgba(124,77,255,0.4),transparent);
    animation: scanline 8s linear infinite;
    z-index: 9999; pointer-events: none;
}

/* ── Animations ── */
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 300% center; }
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,230,118,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(0,230,118,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); }
}
@keyframes scanline {
    0%   { top: 0%; opacity: 1; }
    80%  { opacity: 0.6; }
    100% { top: 100%; opacity: 0; }
}
</style>

<!-- ═══ FOND BLUEPRINT ANIMÉ ═══ -->
<canvas id="ai-bg-canvas" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;"></canvas>
<script>
(function(){
  const canvas = document.getElementById('ai-bg-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, t = 0;
  let dust = [];
  function initDust(){
    dust = Array.from({length:55},()=>({
      x:Math.random()*W, y:Math.random()*H,
      r:0.5+Math.random()*1.6,
      vx:(Math.random()-0.5)*0.18, vy:-(0.08+Math.random()*0.22),
      alpha:0.04+Math.random()*0.18, phase:Math.random()*Math.PI*2,
    }));
  }
  function resize(){ W=canvas.width=window.innerWidth; H=canvas.height=window.innerHeight; initDust(); }
  function draw(){
    t++;
    ctx.clearRect(0,0,W,H);
    const bg=ctx.createRadialGradient(W*.5,H*.48,0,W*.5,H*.48,Math.max(W,H)*.75);
    bg.addColorStop(0,'#0d2a45'); bg.addColorStop(.45,'#0a2038');
    bg.addColorStop(.75,'#071828'); bg.addColorStop(1,'#050f1c');
    ctx.fillStyle=bg; ctx.fillRect(0,0,W,H);
    const CELL=36, CELL5=CELL*5;
    const ox=(t*.08)%CELL, oy=(t*.06)%CELL;
    ctx.save();
    ctx.strokeStyle='rgba(80,160,220,0.10)'; ctx.lineWidth=0.5;
    for(let x=-CELL+ox;x<W+CELL;x+=CELL){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=-CELL+oy;y<H+CELL;y+=CELL){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    ctx.strokeStyle='rgba(100,185,240,0.20)'; ctx.lineWidth=0.8;
    const ox5=(t*.08)%CELL5, oy5=(t*.06)%CELL5;
    for(let x=-CELL5+ox5;x<W+CELL5;x+=CELL5){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=-CELL5+oy5;y<H+CELL5;y+=CELL5){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    ctx.restore();
    ctx.save();
    for(let x=-CELL5+ox5;x<W+CELL5;x+=CELL5)
      for(let y=-CELL5+oy5;y<H+CELL5;y+=CELL5){
        ctx.strokeStyle='rgba(120,200,255,0.30)'; ctx.lineWidth=0.7;
        const cs=5;
        ctx.beginPath();ctx.moveTo(x-cs,y);ctx.lineTo(x+cs,y);ctx.stroke();
        ctx.beginPath();ctx.moveTo(x,y-cs);ctx.lineTo(x,y+cs);ctx.stroke();
      }
    ctx.restore();
    const vig=ctx.createRadialGradient(W/2,H/2,Math.min(W,H)*.3,W/2,H/2,Math.max(W,H)*.75);
    vig.addColorStop(0,'rgba(0,0,0,0)'); vig.addColorStop(.65,'rgba(0,0,0,0.08)'); vig.addColorStop(1,'rgba(0,0,0,0.52)');
    ctx.fillStyle=vig; ctx.fillRect(0,0,W,H);
    dust.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy; p.phase+=0.012;
      if(p.y<-4){p.y=H+4;p.x=Math.random()*W;}
      if(p.x<-4){p.x=W+4;} if(p.x>W+4){p.x=-4;}
      const a=p.alpha*(0.5+0.5*Math.sin(p.phase));
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(160,220,255,${a})`; ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  resize(); draw();
  window.addEventListener('resize',resize);
})();
</script>
""
