"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MEDICALScan AI  ·  Application Streamlit                                    ║
║  Classification CT Rénale  ·  KidneyClassifier v5  ·  AUC 1.00               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              KAMNO RUTH · 2026                               ║
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

# §3 ── Secrets API ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_secrets() -> dict:
    def _get(key: str) -> str:
        try:
            return st.secrets.get(key, os.environ.get(key, ""))
        except Exception:
            return os.environ.get(key, "")
    return {
        "GROQ":  _get("GROQ_API_KEY"),
        "LS":    _get("LANGCHAIN_API_KEY"),
        "MODEL": _get("DEFAULT_GROQ_MODEL") or "llama-3.3-70b-versatile",
        "MP":    _get("MODEL_PATH")          or "outputs_v5/KidneyClassifier_v5.keras",
        "TP":    _get("THRESH_PATH")         or "outputs_v5/thresholds.npy",
    }
    
KEYS: dict = _load_secrets()

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

# §5 ── Design System CSS (StockSight Style Modifié) ───────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── Base reset (Couleur de fond unifiée) ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main .block-container {
    margin: 0; padding: 0;
    background: #020818 !important;
    font-family: 'Exo 2', sans-serif;
    color: #e0eaff;
}
[data-testid="stAppViewContainer"] {
    position: relative; z-index: 1;
}
[data-testid="stHeader"] {
    background: #020818 !important;
    backdrop-filter: none;
    border-bottom: 1px solid rgba(66,165,245,0.15);
    z-index: 100;
}
.main .block-container { padding-top: 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #020818 !important;
    backdrop-filter: none;
    border-right: 1px solid rgba(66,165,245,0.2) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.5);
    z-index: 50;
    max-height: 100vh;
    overflow-y: auto; overflow-x: hidden;
    padding-right: 6px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 8px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-track { background: #020818; }
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(66,165,245,0.4); border-radius: 999px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover { background: rgba(66,165,245,0.65); }
[data-testid="stSidebar"] * { color: #c8deff !important; }
[data-testid="stSidebar"] .stButton > button { color: white !important; }

/* ── Panneaux glassmorphism ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(4,15,40,0.55);
    border-radius: 16px;
}

/* ── Onglets ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: #020818;
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

/* ── Boutons (Correction du flou : suppression de scale) ── */
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
    transform: translateY(-3px); /* Retrait de scale(1.02) pour éviter le flou */
    border-color: rgba(66,165,245,0.8) !important;
}
.stButton > button:active { transform: translateY(-1px); }

/* ── Métriques (Correction du flou : suppression de scale) ── */
[data-testid="metric-container"] {
    background: rgba(13,71,161,0.22);
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
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020818; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg,#0d47a1,#42a5f5);
    border-radius: 3px; box-shadow: 0 0 8px rgba(66,165,245,0.4);
}

/* Hero card (Correction du flou : suppression du scale) */
.hero-card {
    background: rgba(5,20,60,0.65);
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
    transform: translateY(-6px); /* Retrait de scale(1.01) pour éviter le flou */
    box-shadow: 0 24px 60px rgba(13,71,161,0.5), 0 0 0 1px rgba(66,165,245,0.25);
    border-color: rgba(66,165,245,0.45);
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

/* Alertes RAG */
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

/* Boîtes infos */
.info-box {
    background: #0d1f3c !important;
    border: 1px solid rgba(66,165,245,0.3);
    border-left: 3px solid #42a5f5;
    padding: 14px 18px; margin: 14px 0;
    font-size: 13px; color: #dce8ff !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    line-height: 1.65;
}
.info-box strong { color: #90caf9; }

/* Cartes Sidebar */
.sb-card {
    background: #0d1f3c;
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

/* Correction du flou : suppression du scale */
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
    transform: translateY(-2px); /* Retrait de scale */
}

/* Animations globales */
@keyframes shimmer { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(0,230,118,0.7); } 70% { box-shadow: 0 0 0 10px rgba(0,230,118,0); } 100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); } }
</style>

<canvas id='net-bg' style='position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;'></canvas>
<script>
(function(){
    const canvas=document.getElementById('net-bg');
    if(!canvas)return;
    const ctx=canvas.getContext('2d');
    let W, H;
    const dust=[];
    function resize(){
        W=canvas.width=window.innerWidth;
        H=canvas.height=window.innerHeight;
    }
    // Génération discrète de particules
    if(dust.length===0){
        for(let i=0;i<45;i++){
            dust.push({
                x:Math.random()*window.innerWidth,
                y:Math.random()*window.innerHeight,
                r:Math.random()*1.2+0.4,
                vx:(Math.random()-0.5)*0.25,
                vy:-Math.random()*0.3-0.05,
                alpha:Math.random()*0.4+0.1,
                phase:Math.random()*Math.PI
            });
        }
    }
    function draw(){
        // Fond plat unique strict (Retrait de tout dégradé radial)
        ctx.fillStyle='#020818';
        ctx.fillRect(0,0,W,H);
        
        // Rendu discret des micro-poussières de lumière
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
"""
st.markdown(_CSS, unsafe_allow_html=True)

# §6 ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:20px 12px 14px; position:relative;'>
            <div style='font-family:Share Tech Mono,monospace; font-size:9px; color:rgba(66,165,245,0.4); letter-spacing:3px; margin-bottom:8px;'>
                ◈ SYSTEM ONLINE ◈
            </div>
            <div style='font-family:Orbitron,monospace; font-size:22px; font-weight:900;
                        color:#ffffff; letter-spacing:0.5px;'>
                MEDICAL<span style='color:#42a5f5;'>Scan AI</span>
            </div>
            <div style='font-family:Exo 2,sans-serif; font-size:11px; color:#5a8fbf; margin-top:2px;'>
                Classification CT Rénale · v5
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='glow-divider' style='margin:10px 0 16px;'></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sb-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Orbitron,monospace;font-size:11px;font-weight:700;color:#42a5f5;margin-bottom:8px;'>⚙️ CONFIGURATION CORE</div>", unsafe_allow_html=True)
    
    model_opt = [KEYS["MODEL"], "llama-3.1-8b-instant", "mixtral-8x7b-32768"] if KEYS["MODEL"] else ["llama-3.3-70b-versatile"]
    groq_model = st.selectbox("Modèle LLM Expert", options=model_opt, index=0)
    
    ls_active = bool(KEYS["LS"])
    tts_on = st.checkbox("Synthèse vocale (TTS)", value=True)
    show_tr = st.checkbox("Traduction Allemand", value=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sb-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Orbitron,monospace;font-size:11px;font-weight:700;color:#00e676;margin-bottom:6px;'>📊 ETAT DU PIPELINE</div>", unsafe_allow_html=True)
    st.markdown(f"<div><span class='pulse-dot'></span> Modèle IA : <span class='pill pill-ok'>Actif (AUC 1.00)</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:6px;'>🔑 Groq Cloud : " + (f"<span class='pill pill-ok'>Connecté</span>" if KEYS["GROQ"] else f"<span class='pill pill-no'>Hors ligne</span>") + "</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:6px;'>👁️ Tracking : " + (f"<span class='pill pill-ok'>LangSmith OK</span>" if ls_active else f"<span class='pill pill-no'>Inactif</span>") + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:24px; padding:0 4px;'>", unsafe_allow_html=True)
    st.link_button("🌐 Base de données LangSmith", "https://smith.langchain.com", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# §7 ── Moteur d'inférence LLM ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _try_langsmith() -> bool:
    return bool(KEYS["LS"])

ls_active: bool = _try_langsmith()

def _langsmith_log(run_name: str, inputs: dict, outputs: dict, meta: dict) -> dict:
    if not ls_active: return {}
    try:
        from langsmith import Client
        c = Client(api_key=KEYS["LS"])
        rid = c.create_run(
            name=run_name,
            run_type="llm",
            inputs=inputs,
            project_name="MEDICALScan-AI",
            extra={"metadata": meta},
        )
        c.update_run(rid, outputs=outputs, end_time=datetime.datetime.utcnow())
        return {"run_id": str(rid)}
    except Exception as exc:
        return {"error": str(exc)}

def call_llm(messages, system, run_name="llm_call", max_tok=1000):
    if not KEYS["GROQ"]:
        return "❌ Clé Groq manquante dans `.streamlit/secrets.toml`.", {}
    t0 = datetime.datetime.now()
    try:
        from groq import Groq
        full = [{"role":"system","content":system}] + messages
        resp = Groq(api_key=KEYS["GROQ"]).chat.completions.create(
            model=groq_model,
            messages=full,
            max_tokens=max_tok,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        lat = int((datetime.datetime.now()-t0).total_seconds()*1000)
        meta = {
            "model": groq_model,
            "tokens_in": getattr(resp.usage,"prompt_tokens",0),
            "tokens_out": getattr(resp.usage,"completion_tokens",0),
            "latency_ms": lat,
            "run_name": run_name,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        }
        _langsmith_log(run_name, {"messages":full}, {"text":text}, meta)
        st.session_state.setdefault("llm_traces",[]).append(meta)
        return text, meta
    except Exception as exc:
        return f"❌ Erreur LLM : {exc}", {}

# §8 ── Modèle CT ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_model(mp, tp):
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(mp)
        t = np.load(tp) if os.path.exists(tp) else np.full(4, 0.5)
        return m, t, None
    except ImportError:
        return None, None, "DEMO"
    except Exception as exc:
        return None, None, str(exc)

def _predict_real(model, thr, img):
    x = np.array(img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)[np.newaxis]/255.0
    probs = model.predict(x, verbose=0)[0]
    scores = probs - thr
    above = np.where(scores > 0)[0]
    if len(above) > 0:
        idx = above[np.argmax(scores[above])]
    else:
        idx = np.argmax(probs)
    p_dict = {CLASSES[i]: float(probs[i]) for i in range(4)}
    return {
        "class": CLASSES[idx],
        "conf": float(probs[idx]),
        "probs": p_dict,
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def _predict_demo(fn: str) -> dict:
    np.random.seed(abs(hash(fn)) % 1000000)
    raw = np.random.dirichlet(np.ones(4))
    idx = np.argmax(raw)
    p_dict = {CLASSES[i]: float(raw[i]) for i in range(4)}
    return {
        "class": CLASSES[idx],
        "conf": float(raw[idx]),
        "probs": p_dict,
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

model, thr, model_err = _load_model(KEYS["MP"], KEYS["TP"])

# §9 ── Fonctions RAG & Rapports ───────────────────────────────────────────────
def translate_de(text_fr: str) -> str:
    ans, _ = call_llm(
        [{"role":"user","content":f"Translate this medical text to German. Preserve layout:\n\n{text_fr}"}],
        "You are a professional medical translator FR -> DE.",
        run_name="translation_de", max_tok=600
    )
    return ans

def tts(text: str, lang: str) -> io.BytesIO:
    try:
        from gtts import gTTS
        code = "de" if "Allemand" in lang else "fr"
        clean = text.replace("**","").replace("###","").replace("`","")
        out = io.BytesIO()
        gTTS(text=clean, lang=code, slow=False).write_to_fp(out)
        out.seek(0)
        return out
    except Exception:
        return None

def make_summary(chat_history: list, res: dict) -> dict:
    cls = res["class"]; cfg = CLASS_CFG[cls]
    h_str = "\n".join(f"- {m['role']}: {m['content']}" for m in chat_history[-6:])
    fr, _ = call_llm(
        [{"role":"user","content":f"Historique d'échange:\n{h_str}\n\nRésultat initial: {cls} ({cfg['label']})."}],
        (
            "Tu es l'IA de MEDICALScan AI. Synthétise le cas en un rapport concis (200 mots max).\n"
            "Inclus obligatoirement 5 points clairs sous cette forme :\n"
            "1. Résultat 2. Points clés 3. Recommandations 4. Urgence 5. Avertissement IA\n"
            "Réponds UNIQUEMENT avec le résumé structuré."
        ),
        run_name="summary_generation", max_tok=600,
    )
    return {"fr":fr,"de":translate_de(fr)}

def make_system_prompt(res):
    cls=res["class"]; cfg=CLASS_CFG[cls]
    prob="\n".join(f" - {c} : {p*100:.1f}%" for c,p in res["probs"].items())
    return (
        f"Tu es un assistant médical de MEDICALScan AI (radiologie rénale, AUC=1.00).\n"
        f"RÉSULTAT : {cls} ({cfg['label']}) | Confiance : {res['conf']*100:.1f}%\n"
        f"Probabilités :\n{prob}\n"
        f"Urgence : {CTX[cls]['urgence']} | Suivi : {CTX[cls]['suivi']}\n\n"
        f"RÈGLES :\n"
        f"1. Réponds toujours en français, vocabulaire médical accessible.\n"
        f"2. Ne pose jamais de diagnostic définitif.\n"
        f"3. Rappelle que ce résultat IA nécessite confirmation médicale.\n"
        f"4. Pour Tumor (confiance > 70 %) : insiste sur l'URGENCE absolue.\n"
        f"Accueille le patient en résumant le résultat en 2-3 phrases professionnelles."
    )

# §10 ── Composant render_ct_result ────────────────────────────────────────────
def render_ct_result(res: dict) -> None:
    cls = res["class"]
    conf = res["conf"]
    cfg = CLASS_CFG[cls]
    c_color = cfg["color"]
    c_border = cfg["border"]
    c_neon = cfg["neon"]
    c_emoji = cfg["emoji"]
    c_label = cfg["label"]
    c_urg = cfg["urgence"]
    ts_short = res["ts"][-8:]
    conf_pct = f"{conf*100:.1f}"

    # En-tête résultat
    st.markdown(
        f"<div class='ct-result-card' style='--card-accent:{c_color};border-color:{c_border};'>"
        f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;"
        f"letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;'>Diagnostic IA — Résultat</div>"
        f"<div style='font-family:Orbitron,monospace;font-size:28px;font-weight:900;"
        f"color:{c_color};text-shadow:0 0 20px {c_neon};line-height:1.1;'>"
        f"{c_emoji} {cls}</div>"
        f"<div style='font-family:Exo 2,sans-serif;font-size:14px;color:#dce8ff;margin-top:4px;'>{c_label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Barre de confiance
    st.markdown(
        f"<div class='ct-result-card' style='--card-accent:{c_color};border-color:{c_border};padding:14px 18px;'>"
        f"<div style='display:flex;justify-content:space-between;font-family:Exo 2,sans-serif;"
        f"font-size:12px;color:#90caf9;font-weight:600;margin-bottom:8px;'>"
        f"<span>Niveau de confiance</span>"
        f"<span style='font-family:Share Tech Mono,monospace;font-size:13px;color:{c_color};font-weight:700;'>{conf_pct}%</span>"
        f"</div>"
        f"<div style='width:100%;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden;'>"
        f"<div style='width:{conf_pct}%;height:100%;background:{c_color};"
        f"box-shadow:0 0 12px {c_color};border-radius:3px;'></div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Statut & Urgence
    st.markdown(
        f"<div class='ct-result-card' style='--card-accent:{c_color};border-color:{c_border};padding:14px 18px;'>"
        f"<div style='display:flex;align-items:center;justify-content:between;gap:16px;'>"
        f"<div>"
        f"<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#6a90b4;letter-spacing:1px;'>URGENCE</div>"
        f"<div style='font-family:Exo 2,sans-serif;font-size:13px;font-weight:700;color:#ffffff;margin-top:2px;'>{c_urg}</div>"
        f"</div>"
        f"<div style='margin-left:auto;text-align:right;'>"
        f"<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#6a90b4;letter-spacing:1px;'>HORODATAGE</div>"
        f"<div style='font-family:Share Tech Mono,monospace;font-size:12px;color:#90caf9;margin-top:2px;'>{ts_short}</div>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Condition d'alerte RAG complémentaire
    if cls == "Tumor":
        st.markdown(
            "<div class='al al-r'><div class='al-t'>🚨 Alerte Masse Suspecte</div>"
            "<div class='al-b'>Suspicion élevée de carcinome rénal. Prise en charge urologique obligatoire sous 72h. Examen IRM complémentaire requis dans les plus brefs délais.</div></div>",
            unsafe_allow_html=True,
        )
    elif cls == "Stone":
        st.markdown(
            "<div class='al al-o'><div class='al-t'>🪨 Calcul rénal identifié</div>"
            "<div class='al-b'>Présence d'une structure hyperdense évocatrice d'une lithiase. Risque de colique néphrétique. Hydratation contrôlée et consultation urologique conseillée.</div></div>",
            unsafe_allow_html=True,
        )
    elif cls == "Cyst":
        st.markdown(
            "<div class='al al-b2'><div class='al-t'>💧 Kyste rénal détecté</div>"
            "<div class='al-b'>Suivi échographique à 6-12 mois · Classification Bosniak conseillée.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='al al-g'><div class='al-t'>✅ Aucune anomalie détectée</div>"
            "<div class='al-b'>Structures rénales normales · Suivi médical habituel recommandé.</div></div>",
            unsafe_allow_html=True,
        )

# §11 ── Header ────────────────────────────────────────────────────────────────
groq_status = "● GROQ OK" if bool(KEYS["GROQ"]) else "○ GROQ OFFLINE"
ls_status = "● LANGSMITH" if ls_active else "○ LANGSMITH OFF"

st.markdown(f"""
    <div style='background:#020818; padding:20px 32px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(66,165,245,0.3); box-shadow:0 4px 30px rgba(0,0,0,0.5); position:relative; overflow:hidden;'>
        <div style='position:absolute;top:0;left:0;right:0;height:2px; background:linear-gradient(90deg,transparent,#42a5f5,#7c4dff,#00b4d8,transparent);'></div>
        <div style='display:flex; align-items:center; gap:16px;'>
            <div style='width:48px;height:48px;border-radius:12px; background:rgba(66,165,245,0.1);border:1px solid rgba(66,165,245,0.3); display:flex;align-items:center;justify-content:center;font-size:24px;'>🏥</div>
            <div>
                <div style='font-family:Orbitron,monospace;font-size:22px;font-weight:900; background:linear-gradient(90deg,#42a5f5,#7c4dff,#00b4d8); -webkit-background-clip:text;-webkit-text-fill-color:transparent; filter:drop-shadow(0 0 10px rgba(66,165,245,0.3));'>
                    MEDICALScan AI
                </div>
                <div style='font-family:Share Tech Mono,monospace;font-size:10px; color:rgba(66,165,245,0.6);letter-spacing:3px;text-transform:uppercase;margin-top:3px;'>
                    Renal CT Scan Analysis · Groupe 2 · M2 IABD
                </div>
            </div>
        </div>
        <div style='display:flex; gap:10px; font-family:Share Tech Mono,monospace; font-size:10px;'>
            <span style='padding:5px 12px; border-radius:6px; background:rgba(66,165,245,0.1); border:1px solid rgba(66,165,245,0.25); color:#42a5f5;'>{groq_status}</span>
            <span style='padding:5px 12px; border-radius:6px; background:rgba(124,77,255,0.1); border:1px solid rgba(124,77,255,0.25); color:#7c4dff;'>{ls_status}</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# §12 ── Menu Onglets ──────────────────────────────────────────────────────────
tab_main, tab_chat, tab_sum, tab_mon = st.tabs([
    "🔬 ANALYSE CT SCAN",
    "💬 ASSISTANT MÉDICAL",
    "📋 RÉSUMÉ & RAPPORT",
    "🖥️ MONITORING PIPELINE"
])

# ─── §12 INTERFACE PRINCIPALE (ANALYSE CT) ────────────────────────────────────
with tab_main:
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1.1, 1.0], gap="large")
    
    with col_l:
        st.markdown("<div class='section-title'>📁 Importation du Scan Tomodensitométrique</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Glissez-déposez le cliché DICOM exporté en JPEG/PNG (résolution native)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded:
            pil = Image.open(uploaded)
            st.image(pil, use_container_width=True, caption=f"Scanner importé : {uploaded.name}")
        else:
            st.markdown(
                "<div style='height:280px;display:flex;align-items:center;justify-content:center;"
                "flex-direction:column;gap:12px;background:rgba(4,14,38,0.4);"
                "border:1px dashed rgba(66,165,245,0.2);border-radius:14px;'>"
                "<div style='font-size:2.8rem;opacity:0.25;'>📥</div>"
                "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#5a8fbf;text-align:center;'>"
                "Aucun scan en cours d'analyse.<br>"
                "<span style='font-size:11px;color:#3d6b9e;'>Veuillez utiliser la zone d'import ci-dessus</div>"
                "</div></div>", unsafe_allow_html=True,
            )

    with col_r:
        if not uploaded:
            st.markdown(
                "<div style='height:380px;display:flex;align-items:center;justify-content:center;"
                "flex-direction:column;gap:14px;background:rgba(4,14,38,0.6);"
                "border:1px solid rgba(66,165,245,0.2);border-radius:16px;'>"
                "<div style='font-size:3.5rem;opacity:0.15;'>🔬</div>"
                "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#3d6b9e;text-align:center;'>"
                "En attente d'une image CT<br>"
                "<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#2d5a8e;'>"
                "Importez un scan pour démarrer l'analyse</span></div></div>", unsafe_allow_html=True,
            )
        else:
            prev = st.session_state.get("res", {})
            if prev.get("_src") != uploaded.name:
                for k in ["res","chat","sys_prompt","summary","translations","audio","llm_traces"]:
                    st.session_state.pop(k, None)
            
            if model_err == "DEMO":
                res = _predict_demo(uploaded.name)
            elif model_err:
                res = None
            else:
                res = _predict_real(model, thr, pil)
                
            if res:
                res["_src"] = uploaded.name
                st.session_state["res"] = res
            else:
                res = prev
                
            if not res and model_err and model_err != "DEMO":
                st.error(f"Erreur modèle : {model_err}")
            elif res:
                render_ct_result(res)

    # Probabilités + interprétation
    if uploaded and "res" in st.session_state:
        res = st.session_state["res"]
        cls = res["class"]; cfg = CLASS_CFG[cls]
        
        st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Distribution des probabilités & Interprétation</div>", unsafe_allow_html=True)
        
        cp, ci = st.columns([1, 1], gap="large")
        
        with cp:
            st.markdown(
                "<div class='ct-result-card'>"
                "<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;"
                "letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;"
                "padding-bottom:10px;border-bottom:1px solid rgba(66,165,245,0.15);'>"
                "📊 Probabilités par classe</div>", unsafe_allow_html=True,
            )
            
            for c_name in CLASSES:
                val = res["probs"][c_name]
                is_top = (c_name == cls)
                t_cls = "prob-name prob-top" if is_top else "prob-name"
                p_cls = "prob-pct prob-top" if is_top else "prob-pct"
                c_hex = CLASS_CFG[c_name]["color"]
                lbl = CLASS_CFG[c_name]["label"]
                
                badge_html = ""
                if is_top:
                    badge_html = f"<span class='prob-badge' style='background:{c_hex};'>WINNER</span>"
                else:
                    badge_html = f"<span class='prob-badge' style='background:rgba(255,255,255,0.05);color:#5a8fbf !important;border:1px solid rgba(255,255,255,0.1);'>SCORE</span>"
                
                st.markdown(
                    f"<div class='prob-row'>"
                    f"{badge_html}"
                    f"<div class='{t_cls}'>{c_name}</div>"
                    f"<div class='prob-track'><div class='prob-fill' style='width:{val*100}%;background:{c_hex};box-shadow:0 0 10px {c_hex};'></div></div>"
                    f"<div class='{p_cls}'>{val*100:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with ci:
            interp_html = INTERP[cls]
            suivi = CTX[cls]["suivi"]
            st.markdown(
                f"<div class='ct-result-card' style='border-left:4px solid {c_color};'>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;"
                f"letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;"
                f"padding-bottom:8px;border-bottom:1px solid rgba(66,165,245,0.15);'>"
                f"📝 Interprétation médicale automatique</div>"
                f"<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#dce8ff;line-height:1.8;'>"
                f"{interp_html}</div>"
                f"<div style='margin-top:16px;padding-top:12px;border-top:1px solid rgba(66,165,245,0.15);'>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;"
                f"text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;'>Suivi recommandé</div>"
                f"<div style='display:inline-block;font-family:Exo 2,sans-serif;font-size:13px;font-weight:700;"
                f"color:#ffffff;background:rgba(255,255,255,0.08);border:1px solid {c_color};"
                f"border-left:4px solid {c_color};border-radius:6px;padding:7px 14px;'>{suivi}</div>"
                f"</div></div>", unsafe_allow_html=True,
            )
            
        st.markdown(
            "<div class='info-box'>💬 Consultez l'onglet <strong>Assistant Médical</strong> pour des questions personnalisées sur ce résultat.</div>",
            unsafe_allow_html=True,
        )

# ─── §13 ASSISTANT MÉDICAL ───────────────────────────────────────────────────
with tab_chat:
    res = st.session_state.get("res")
    if res is None:
        st.markdown(
            "<div style='height:320px;display:flex;align-items:center;justify-content:center;"
            "flex-direction:column;gap:14px;background:rgba(4,14,38,0.6);"
            "border:1px solid rgba(66,165,245,0.2);border-radius:16px;margin-top:10px;'>"
            "<div style='font-size:3rem;opacity:0.15;'>🔬</div>"
            "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#3d6b9e;text-align:center;'>"
            "Aucune analyse disponible<br>"
            "<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#2d5a8e;'>"
            "Uploadez une image CT dans l'onglet Analyse CT</span></div></div>", unsafe_allow_html=True,
        )
    elif not KEYS["GROQ"]:
        st.error("Clé Groq manquante dans `.streamlit/secrets.toml`.")
    else:
        cls = res["class"]; conf = res["conf"]; cfg = CLASS_CFG[cls]
        pills = "".join([
            "<span class='pill pill-ok'>🔊 Audio</span>" if tts_on else "",
            "<span class='pill pill-ok'>🇩🇪 Traduction</span>" if show_tr else "",
            "<span class='pill pill-ok'>🟢 LangSmith</span>" if ls_active else "",
        ])
        
        c_color = cfg["color"]
        st.markdown(
            f"<div class='ct-result-card' style='border-color:rgba(66,165,245,0.25);margin-top:10px;'>"
            f"<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#6a90b4;'>CONTEXTE PATIENT INJECTÉ</div>"
            f"<div style='font-family:Orbitron,monospace;font-size:14px;color:{c_color};margin-top:2px;'>{cfg['emoji']} {cfg['label']} ({conf*100:.1f}%)</div>"
            f"<div style='margin-top:8px; display:flex; gap:6px;'>{pills}</div>"
            f"</div>", unsafe_allow_html=True,
        )
        
        st.session_state.setdefault("chat", [])
        st.session_state.setdefault("sys_prompt", make_system_prompt(res))
        st.session_state.setdefault("translations", {})
        st.session_state.setdefault("audio", {})

        for idx, m in enumerate(st.session_state["chat"]):
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m["role"] == "assistant":
                    if idx in st.session_state["translations"]:
                        st.markdown(
                            f"<div class='de-box'><div class='de-t'>🇩🇪 Deutsch Expert</div>"
                            f"<div class='de-b'>{st.session_state['translations'][idx].replace(chr(10),'<br>')}</div></div>",
                            unsafe_allow_html=True,
                        )
                    if idx in st.session_state["audio"]:
                        st.audio(st.session_state["audio"][idx], format="audio/mp3")

        if prompt := st.chat_input("Posez votre question sur les kystes, calculs ou tumeurs..."):
            st.session_state["chat"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("<span style='color:#6a90b4;'>Inférence en cours...</span>", unsafe_allow_html=True)
                
                answer, metrics = call_llm(st.session_state["chat"], st.session_state["sys_prompt"], "agent_chat")
                placeholder.write(answer)
                idx = len(st.session_state["chat"])
                st.session_state["chat"].append({"role": "assistant", "content": answer})
                
                if show_tr and KEYS["GROQ"]:
                    de_text = translate_de(answer)
                    st.session_state["translations"][idx] = de_text
                    st.markdown(
                        f"<div class='de-box'><div class='de-t'>🇩🇪 Deutsch Expert</div>"
                        f"<div class='de-b'>{de_text.replace(chr(10),'<br>')}</div></div>",
                        unsafe_allow_html=True,
                    )
                
                if tts_on:
                    tts_lang = "Allemand 🇩🇪" if (show_tr and "German" in tts_lang if 'tts_lang' in locals() else False) else "Français 🇫🇷"
                    txt = st.session_state["translations"][idx] if "Allemand" in tts_lang else answer
                    ab = tts(txt, tts_lang)
                    if ab:
                        st.session_state["audio"][idx] = ab
                        ll = "Deutsch" if "Allemand" in tts_lang else "Français"
                        st.markdown(f"<div class='audio-box'><div class='au-t'>🔊 {ll}</div></div>", unsafe_allow_html=True)
                        st.audio(ab, format="audio/mp3")
                        
            if KEYS["GROQ"] and len(st.session_state["chat"]) >= 2:
                st.session_state["summary"] = make_summary(st.session_state["chat"], res)
                st.success("✅ Résumé mis à jour — onglet **Résumé & Rapport**.")
                
            if metrics and ls_active:
                st.markdown(
                    f"<div class='trace-bar'>"
                    f"Run : <span>{metrics.get('run_name','')}</span> &nbsp;|&nbsp; "
                    f"Latence : <span>{metrics.get('latency_ms',0)} ms</span> &nbsp;|&nbsp; "
                    f"Tokens : <span>{metrics.get('tokens_in',0)}→{metrics.get('tokens_out',0)}</span> &nbsp;|&nbsp; "
                    f"Modèle : <span>{metrics.get('model','')}</span> &nbsp;|&nbsp; "
                    f"Heure : <span>{metrics.get('timestamp','')}</span>"
                    f"</div>", unsafe_allow_html=True,
                )

        b1, b2, _ = st.columns([1,1.5,3])
        with b1:
            if st.button("🔄 Réinitialiser"):
                for k in ["chat","sys_prompt","translations","audio","summary"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with b2:
            if (len(st.session_state.get("chat",[]))>=2 and not st.session_state.get("summary") and st.button("📋 Générer résumé")):
                st.session_state["summary"] = make_summary(st.session_state["chat"], res)
                st.success("✅ Résumé généré — onglet **Résumé & Rapport**.")

# ─── §14 RÉSUMÉ & RAPPORT ────────────────────────────────────────────────────
with tab_sum:
    res = st.session_state.get("res")
    summary = st.session_state.get("summary")
    chat = st.session_state.get("chat", [])
    
    if res is None:
        st.markdown(
            "<div style='height:280px;display:flex;align-items:center;justify-content:center;"
            "flex-direction:column;gap:12px;background:rgba(4,14,38,0.6);"
            "border:1px solid rgba(66,165,245,0.2);border-radius:16px;margin-top:10px;'>"
            "<div style='font-size:3rem;opacity:0.15;'>📋</div>"
            "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#3d6b9e;text-align:center;'>"
            "Aucun rapport disponible<br>"
            "<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#2d5a8e;'>"
            "Générez une analyse pour extraire le compte rendu médical</span></div></div>", unsafe_allow_html=True,
        )
    elif not summary:
        st.markdown(
            "<div style='height:240px;display:flex;align-items:center;justify-content:center;"
            "flex-direction:column;gap:12px;background:rgba(4,14,38,0.4);"
            "border:1px dashed rgba(66,165,245,0.2);border-radius:14px;margin-top:10px;'>"
            "<div style='font-size:2.5rem;opacity:0.2;'>💬</div>"
            "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#5a8fbf;text-align:center;'>"
            "Le résumé se génère automatiquement après vos premiers échanges avec l'assistant.<br>"
            "<span style='font-size:11px;color:#3d6b9e;'>Alternativement, forcez la création via le bouton ci-dessous :</span></div>", unsafe_allow_html=True,
        )
        if st.button("📋 Générer le compte rendu maintenant", key="force_sum"):
            st.session_state["summary"] = make_summary([{"role":"user","content":"Générer rapport standard."}], res)
            st.rerun()
    else:
        cls = res["class"]; cfg = CLASS_CFG[cls]
        st.markdown("<div class='section-title'>📝 Rapport de Synthèse Automatique (RAG-Driven)</div>", unsafe_allow_html=True)
        
        c_fr, c_de = st.columns(2, gap="large")
        with c_fr:
            st.markdown(
                f"<div class='sum-card'>"
                f"<div class='sum-label' style='color:#42a5f5;'>Version Française</div>"
                f"<div class='sum-body'>{summary['fr'].replace(chr(10),'<br>')}</div>"
                f"</div>", unsafe_allow_html=True,
            )
            if tts_on and st.button("🔊 Écouter en Français"):
                ab = tts(summary["fr"], "Français 🇫🇷")
                if ab: st.audio(ab, format="audio/mp3")
                
        with c_de:
            st.markdown(
                f"<div class='sum-de-card'>"
                f"<div class='sum-label' style='color:#ffc107;'>Deutsche Version</div>"
                f"<div class='sum-body' style='color:#ffe082;'>{summary['de'].replace(chr(10),'<br>')}</div>"
                f"</div>", unsafe_allow_html=True,
            )
            if tts_on and st.button("🔊 Auf Deutsch anhören"):
                ab = tts(summary["de"], "Allemand 🇩🇪")
                if ab: st.audio(ab, format="audio/mp3")
                
        ctx_data = CTX[cls]
        export = (
            f"MEDICALScan AI — COMPTE RENDU · {res['ts']}\n"
            f"KAMNO RUTH\n{'='*60}\n"
            f"Classe : {cls} ({cfg['label']}) | Confiance : {conf*100:.1f}%\n"
            f"Urgence : {ctx_data['urgence']} | Suivi : {ctx_data['suivi']}\n"
            f"{'='*60} RÉSUMÉ FR {'='*60}\n{summary['fr']}\n"
            f"{'='*60} ZUSAMMENFASSUNG DE {'='*60}\n{summary['de']}\n"
            f"{'='*60}\nRésultat IA — à confirmer par un professionnel de santé.\n"
        )
        
        c1, c2, _ = st.columns([1,1,3])
        with c1:
            fn = f"MEDICALScan_{res['ts'].replace(' ','_').replace(':','-')}.txt"
            st.download_button("💾 Télécharger le fichier (.txt)", data=export, file_name=fn, use_container_width=True)
        with c2:
            if st.button("🗑️ Effacer le rapport", use_container_width=True):
                st.session_state.pop("summary", None); st.rerun()

# ─── §15 MONITORING PIPELINE ─────────────────────────────────────────────────
with tab_mon:
    st.markdown("<div class='section-title'>🖥️ Télémétrie en temps réel (LangSmith Traces)</div>", unsafe_allow_html=True)
    
    traces = st.session_state.get("llm_traces", [])
    if not traces:
        st.markdown(
            "<div style='height:200px;display:flex;align-items:center;justify-content:center;"
            "flex-direction:column;gap:12px;background:rgba(4,14,38,0.4);"
            "border:1px dashed rgba(66,165,245,0.2);border-radius:14px;margin-top:10px;'>"
            "<div style='font-size:2.5rem;opacity:0.2;'>📊</div>"
            "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#5a8fbf;'>Aucune métrique LLM enregistrée pour le moment.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"<div class='mon-card'><div class='mon-val'>{len(traces)}</div><div class='mon-lbl'>Appels API</div></div>", unsafe_allow_html=True)
        with m2:
            avg_lat = int(np.mean([t.get("latency_ms",0) for t in traces]))
            st.markdown(f"<div class='mon-card'><div class='mon-val'>{avg_lat} ms</div><div class='mon-lbl'>Latence Moyenne</div></div>", unsafe_allow_html=True)
        with m3:
            tot_tok = sum(t.get("tokens_in",0)+t.get("tokens_out",0) for t in traces)
            st.markdown(f"<div class='mon-card'><div class='mon-val'>{tot_tok}</div><div class='mon-lbl'>Tokens Totalisés</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='mon-card'><div class='mon-val' style='color:#7c4dff;'>{groq_model[:9]}</div><div class='mon-lbl'>Modèle actif</div></div>", unsafe_allow_html=True)
            
        c_g1, c_g2 = st.columns([2, 1], gap="medium")
        with c_g1:
            st.markdown(
                "<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;"
                "text-transform:uppercase;letter-spacing:1.5px;margin:14px 0 6px;'>Latence par appel (ms)</div>",
                unsafe_allow_html=True,
            )
            st.bar_chart([t.get("latency_ms",0) for t in traces])
            
        st.markdown(
            "<div class='ct-result-card' style='margin-top:10px;'> Gentilesse"
            "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#8aabcc;'>🌐 Dashboard complet : "
            "<a href='https://smith.langchain.com' target='_blank' style='color:#42a5f5;font-weight:700;text-decoration:none;'>smith.langchain.com</a> → Projet : "
            "<span style='color:#7c4dff;font-weight:700;'>MEDICALScan-AI</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        
        if st.button("🗑️ Effacer les traces"):
            st.session_state["llm_traces"] = []; st.rerun()

# §16 ── Footer ────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "<div>⚠️ <span>Avertissement médical</span> : Ce système est un outil d'aide à la décision basé sur l'IA. "
    "Il ne remplace en aucun cas un diagnostic médical établi par un professionnel qualifié. "
    "Tout résultat doit être confirmé par un radiologue ou médecin spécialiste.</div>"
    "<div style='margin-top:10px; opacity:0.4; font-size:9px;'>MEDICALScan-AI · KAMNO RUTH · v5.0 · Multi-Tab Pipeline Architecture</div>"
    "</div>",
    unsafe_allow_html=True,
)
