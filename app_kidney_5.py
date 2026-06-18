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
        "color": "#42a5f5", "bg": "rgba(13,71,161,0.15)", "border": "rgba(66,165,245,0.3)",
        "label": "Kyste rénal", "urgence": "Faible", "emoji": "💧", "neon": "rgba(66,165,245,0.4)"
    },
    "Normal": {
        "color": "#00e676", "bg": "rgba(0,100,50,0.15)", "border": "rgba(0,230,118,0.3)",
        "label": "Rein normal", "urgence": "Aucune", "emoji": "✅", "neon": "rgba(0,230,118,0.4)"
    },
    "Stone": {
        "color": "#ff9800", "bg": "rgba(100,60,0,0.15)", "border": "rgba(255,152,0,0.3)",
        "label": "Lithiase rénale (calcul)", "urgence": "Modérée", "emoji": "🪨", "neon": "rgba(255,152,0,0.4)"
    },
    "Tumor": {
        "color": "#ff5252", "bg": "rgba(120,0,0,0.15)", "border": "rgba(255,82,82,0.3)",
        "label": "Tumeur rénale", "urgence": "Élevée ⚠️", "emoji": "🔴", "neon": "rgba(255,82,82,0.4)"
    }
}

INTERP: dict = {
    "Normal": "L'analyse ne révèle <strong>aucune anomalie rénale significative</strong>.",
    "Cyst": "L'analyse identifie une <strong>formation kystique rénale</strong>.",
    "Stone": "L'analyse détecte la <strong>présence de calculs rénaux</strong>.",
    "Tumor": "L'analyse identifie une <strong>masse rénale suspecte nécessitant une évaluation urgente</strong>."
}

CTX: dict = {
    "Cyst":   {"urgence": "Faible à modérée", "suivi": "Échographie à 6-12 mois"},
    "Normal": {"urgence": "Aucune", "suivi": "Contrôle de routine"},
    "Stone":  {"urgence": "Modérée", "suivi": "Consultation urologique"},
    "Tumor":  {"urgence": "⚠️ ÉLEVÉE", "suivi": "IRM + avis urologique urgent"}
}

# §5 ── Design System CSS (Retour au Style Initial + Corrections ciblées) ──
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── 1. UNIFORMISATION STRICTE DU FOND DE PAGE ET DE LA SIDEBAR ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stAppViewBlockContainer"], section[data-testid="stSidebar"] {
    margin: 0; padding: 0;
    background-color: #020818 !important;
    background: #020818 !important; /* Même couleur de fond partout */
    font-family: 'Exo 2', sans-serif;
    color: #E5E3F5 !important;
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

/* Ajustements Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(66,165,245,0.2) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6);
    z-index: 50;
}
[data-testid="stSidebar"] * { color: #c8deff !important; }
[data-testid="stSidebar"] label p {
    color: #90caf9 !important;
    font-weight: 600;
}

/* ── 2. CORRECTION DES INPUTS, SELECTBOX ET CHAT_INPUT (TEXTE VISIBLE) ── */
.stTextInput input, 
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
[data-baseweb="select"] *,
[data-testid="stChatInput"] textarea,
.stChatInput textarea,
div[data-baseweb="input"] input {
    background-color: #0d1f3c !important;
    background: #0d1f3c !important;
    color: #ffffff !important; /* ← S'assurer que c'est bien #ffffff */
    border: 1px solid rgba(66,165,245,0.3) !important;
    border-radius: 10px !important;
    -webkit-text-fill-color: #ffffff !important; /* ← AJOUTER OU FIXER CECI EN #ffffff POUR LA SAISIE UTILSATEUR */
}

/* Menu déroulant des selectbox */
ul[role="listbox"] li, [data-baseweb="popover"] * {
    background-color: #0d1f3c !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* ── 3. CORRECTION DU SOFTMAX (TEXTE VISIBLE POUR TOUTES LES CLASSES) ── */
.prob-row { display:flex; align-items:center; gap:10px; margin:8px 0; }

.prob-name {
    font-family:'Share Tech Mono',monospace; font-size:13px;
    color: #ffffff !important; /* Forcé en blanc pur pour toutes les classes */
    -webkit-text-fill-color: #ffffff !important;
    width:75px; flex-shrink:0; letter-spacing:0.5px;
    font-weight: 600;
}
.prob-track { flex:1; height:8px; background:rgba(255,255,255,0.12); border-radius:4px; overflow:hidden; border:1px solid rgba(255,255,255,0.15); }
.prob-fill  { height:100%; border-radius:4px; transition: width 0.8s ease; }
.prob-pct   {
    font-family:'Share Tech Mono',monospace; font-size:13px;
    color: #42a5f5 !important; /* Forcé en bleu clair visible pour toutes les classes */
    -webkit-text-fill-color: #42a5f5 !important;
    width:50px; text-align:right; flex-shrink:0; font-weight:600;
}

/* ── 4. PANNEAUX, MÉTRIQUES ET DESIGN INITIAL RELESTÉ ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(4,15,40,0.55);
    backdrop-filter: blur(8px);
    border-radius: 16px;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px; background: rgba(2,8,24,0.75); backdrop-filter: blur(16px);
    border-radius: 14px; padding: 6px; border: 1px solid rgba(66,165,245,0.2);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 22px;
    font-family: 'Exo 2', sans-serif; font-weight: 700; font-size: 13px;
    color: #5a8fbf !important; background: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg,#0d47a1 0%,#1565c0 50%,#1e88e5 100%) !important;
    color: #ffffff !important;
}

/* Boutons */
.stButton > button {
    background: linear-gradient(135deg,#0a2a5e 0%,#0d47a1 40%,#1976d2 80%,#42a5f5 100%) !important;
    color: white !important;
    border: 1px solid rgba(66,165,245,0.4) !important;
    border-radius: 12px; padding: 12px 32px;
    font-family: 'Exo 2', sans-serif; font-weight: 700; text-transform: uppercase;
    box-shadow: 0 0 30px rgba(13,71,161,0.5);
}
.stButton > button:hover {
    box-shadow: 0 0 50px rgba(66,165,245,0.7);
    border-color: rgba(66,165,245,0.8) !important;
}

/* Métriques */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,rgba(13,71,161,0.22) 0%,rgba(5,20,50,0.7) 100%) !important;
    border: 1px solid rgba(66,165,245,0.25) !important;
    border-radius: 14px; padding: 18px 20px;
}
[data-testid="metric-container"] label {
    color: #5a8fbf !important; font-size: 11px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #42a5f5 !important; font-family: 'Orbitron', monospace !important;
    font-size: 20px !important; font-weight: 700;
}

/* Cartes de résultats RAG et IA */
.ct-result-card, .info-box, .sum-card, .sum-de-card, .de-box {
    background: #0d1f3c !important;
    border: 1px solid rgba(66,165,245,0.35) !important;
    border-radius: 16px; padding: 20px 18px;
    color: #e0eaff !important;
}

.stChatMessage {
    background: rgba(4,15,40,0.7) !important;
    border: 1px solid rgba(66,165,245,0.2) !important;
    border-radius: 12px !important;
}

/* Titres & Dégradés animés */
.hero-title {
    font-family: 'Orbitron', monospace; font-size: 48px; font-weight: 900;
    background: linear-gradient(90deg,#0d47a1,#42a5f5,#7c4dff,#00b4d8,#42a5f5,#0d47a1);
    background-size: 400% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite; text-align: center; margin-bottom: 6px;
}
.hero-subtitle { color: #5a8fbf; text-align: center; font-size: 11px; letter-spacing: 5px; font-family: 'Share Tech Mono', monospace; }
.section-title {
    font-family: 'Orbitron', monospace; font-size: 20px; font-weight: 700;
    background: linear-gradient(90deg,#42a5f5,#7c4dff,#00b4d8,#42a5f5); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

@keyframes shimmer { 0% { background-position: 0% center; } 100% { background-position: 300% center; } }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)
# §6 ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 12px 14px; position:relative;'>
        <div style='font-family:Share Tech Mono,monospace; font-size:9px;
                    color:rgba(66,165,245,0.4); letter-spacing:3px; margin-bottom:8px;'>
            ◈ SYSTEM ONLINE ◈
        </div>
        <div style='font-family:Orbitron,monospace; font-size:22px; font-weight:900;
                    background:linear-gradient(135deg,#42a5f5,#7c4dff,#00b4d8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    filter:drop-shadow(0 0 12px rgba(66,165,245,0.4));'>
            🏥 MEDICALScan AI
        </div>
        <div style='color:#2d5a8e; font-size:10px; letter-spacing:3px; margin-top:4px;
                    font-family:Share Tech Mono,monospace;'>
            RENAL CT ANALYSIS
        </div>
        <div style='margin-top:10px; display:flex; justify-content:center; align-items:center; gap:6px;'>
            <span class='pulse-dot'></span>
            <span style='font-size:10px; color:#00e676; font-family:Share Tech Mono,monospace;'>
                AI READY
            </span>
        </div>
    </div>
    <div style='height:1px; background:linear-gradient(90deg,transparent,rgba(66,165,245,0.4),transparent); margin:4px 0 16px;'></div>
    """, unsafe_allow_html=True)

    groq_ok = bool(KEYS["GROQ"])
    ls_ok   = bool(KEYS["LS"])

    st.markdown(
        f"<div style='padding:0 6px 8px;'>"
        f"<div style='font-family:Share Tech Mono,monospace; font-size:10px; color:#5a8fbf; letter-spacing:2px; text-transform:uppercase; margin-bottom:8px;'>Statut des services</div>"
        f"<div style='display:flex; gap:6px; flex-wrap:wrap;'>"
        f"<span class='pill {'pill-ok' if groq_ok else 'pill-no'}'>{'✓' if groq_ok else '✗'} Groq</span>"
        f"<span class='pill {'pill-ok' if ls_ok else 'pill-no'}'>{'✓' if ls_ok else '✗'} LangSmith</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#5a8fbf;"
        "letter-spacing:2px;text-transform:uppercase;padding:0 6px;margin:12px 0 6px;'>Modèle LLM</div>",
        unsafe_allow_html=True,
    )

    # ── CORRECTION : suppression des modèles désactivés (mixtral, gemma2) ──
    groq_model: str = st.selectbox(
        "groq_model",
        [
            "llama-3.3-70b-versatile",   # meilleur choix par défaut
            "llama-3.1-8b-instant",       # rapide, léger
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#5a8fbf;letter-spacing:2px;text-transform:uppercase;padding:0 6px;margin:12px 0 6px;'>Monitoring</div>", unsafe_allow_html=True)
    langsmith_on: bool = st.toggle("LangSmith actif", value=ls_ok)

    st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#5a8fbf;letter-spacing:2px;text-transform:uppercase;padding:0 6px;margin:12px 0 6px;'>Synthèse vocale</div>", unsafe_allow_html=True)
    tts_on:   bool = st.toggle("Audio TTS", value=True)
    tts_lang: str  = st.selectbox("tts_lang", ["Français 🇫🇷","Allemand 🇩🇪"], label_visibility="collapsed")

    st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#5a8fbf;letter-spacing:2px;text-transform:uppercase;padding:0 6px;margin:12px 0 6px;'>Options</div>", unsafe_allow_html=True)
    show_tr:  bool = st.toggle("Traduction 🇩🇪",       value=True)
    auto_sum: bool = st.toggle("Résumé automatique", value=True)

    st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#5a8fbf;letter-spacing:2px;text-transform:uppercase;padding:0 6px;margin:12px 0 6px;'>Applications</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='padding:0 6px 14px;'>"
        "<a href='https://stocksightaistockprediction-rrceguvir9vxa9tmappwkps.streamlit.app/'"
        " target='_blank' class='ext-link'>📈 Application financière StockSight</a>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='padding:14px;margin-top:40px;border-top:1px solid rgba(66,165,245,0.15);'>"
        "<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#2d5a8e;line-height:1.8;text-align:center;'>"
        "<span style='color:#42a5f5;font-weight:700;'>Groupe 2 · M2 IABD</span><br>"
        "KAMNO<br>"
        "KidneyClassifier v5 · AUC 1.00"
        "</div></div>",
        unsafe_allow_html=True,
    )

# §7 ── Service LLM ────────────────────────────────────────────────────────────
ls_active: bool = langsmith_on and bool(KEYS["LS"])

if ls_active:
    os.environ.update({
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY":    KEYS["LS"],
        "LANGCHAIN_PROJECT":    "MEDICALScan-AI",
        "LANGCHAIN_ENDPOINT":   "https://api.smith.langchain.com",
    })


def _langsmith_log(run_name: str, inputs: dict, outputs: dict, meta: dict) -> dict:
    if not ls_active:
        return {}
    try:
        from langsmith import Client
        c   = Client(api_key=KEYS["LS"])
        rid = c.create_run(
            name=run_name, run_type="llm", inputs=inputs,
            project_name="MEDICALScan-AI", extra={"metadata": meta},
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
            model=groq_model, messages=full, max_tokens=max_tok, temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        lat  = int((datetime.datetime.now()-t0).total_seconds()*1000)
        meta = {
            "model": groq_model,
            "tokens_in":  getattr(resp.usage,"prompt_tokens",0),
            "tokens_out": getattr(resp.usage,"completion_tokens",0),
            "latency_ms": lat,
            "run_name":   run_name,
            "timestamp":  datetime.datetime.now().strftime("%H:%M:%S"),
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
    probs  = model.predict(x, verbose=0)[0]
    scores = probs - thr
    above  = np.where(scores > 0)[0]
    idx    = int(above[np.argmax(probs[above])]) if len(above)>1 else (int(above[0]) if len(above)==1 else int(np.argmax(probs)))
    return _build_result(CLASSES[idx], float(probs[idx]), {c:float(v) for c,v in zip(CLASSES,probs)})


def _predict_demo(filename):
    import hashlib
    cls = CLASSES[int(hashlib.md5(filename.encode()).hexdigest(),16)%4]
    c   = {"Tumor":0.87,"Stone":0.92,"Cyst":0.78,"Normal":0.95}[cls]
    raw = {k:0.02 for k in CLASSES}; raw[cls]=c
    tot = sum(raw.values())
    return _build_result(cls, c, {k:v/tot for k,v in raw.items()})


def _build_result(cls, conf, probs):
    return {"class":cls,"conf":conf,"probs":probs,"ts":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# §9 ── Fonctions utilitaires ──────────────────────────────────────────────────
def tts(text, lang):
    try:
        import re
        from gtts import gTTS
        clean = re.sub(r"\*+|#+\s*","",text)
        clean = re.sub(r"\n+"," ",clean).strip()
        buf = io.BytesIO()
        gTTS(text=clean, lang="de" if "Allemand" in lang else "fr", slow=False).write_to_fp(buf)
        buf.seek(0); return buf.read()
    except Exception:
        return None


def translate_de(text_fr):
    out, _ = call_llm(
        messages=[{"role":"user","content":text_fr}],
        system="Traduis ce texte médical du français vers l'allemand. Réponds UNIQUEMENT avec la traduction.",
        run_name="translation_fr_de", max_tok=800,
    )
    return out


def make_summary(history, res):
    cls = res["class"]
    tr  = "\n".join(f"{'Patient' if m['role']=='user' else 'Médecin IA'}: {m['content']}" for m in history)
    fr, _ = call_llm(
        messages=[{"role":"user","content":f"Conversation :\n{tr}"}],
        system=(
            f"Résume en 5 points structurés (max 250 mots) :\n"
            f"Résultat IA : {cls} ({CTX[cls]['urgence']}) | Confiance : {res['conf']*100:.1f}%\n"
            f"1. Résultat  2. Points clés  3. Recommandations  4. Urgence  5. Avertissement IA\n"
            f"Réponds UNIQUEMENT avec le résumé structuré."
        ),
        run_name="summary_generation", max_tok=600,
    )
    return {"fr":fr,"de":translate_de(fr)}


def make_system_prompt(res):
    cls=res["class"]; cfg=CLASS_CFG[cls]
    prob="\n".join(f"  - {c} : {p*100:.1f}%" for c,p in res["probs"].items())
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
    cls      = res["class"]
    conf     = res["conf"]
    cfg      = CLASS_CFG[cls]
    c_color  = cfg["color"]
    c_border = cfg["border"]
    c_neon   = cfg["neon"]
    c_emoji  = cfg["emoji"]
    c_label  = cfg["label"]
    c_urg    = cfg["urgence"]
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
        f"<span style='font-family:Orbitron,monospace;font-weight:900;font-size:16px;color:{c_color};"
        f"text-shadow:0 0 15px {c_neon};'>{conf_pct}%</span></div>"
        f"<div style='height:10px;background:rgba(13,71,161,0.2);border-radius:5px;overflow:hidden;"
        f"border:1px solid rgba(66,165,245,0.15);'>"
        f"<div style='height:100%;width:{conf_pct}%;background:linear-gradient(90deg,{c_color},{c_neon});"
        f"border-radius:5px;box-shadow:0 0 10px {c_neon};'></div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Métriques 3 colonnes
    c1, c2, c3 = st.columns(3)
    lbl_style = "font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;letter-spacing:1.5px;text-transform:uppercase;margin-top:6px;font-weight:600;"
    with c1:
        st.markdown(
            f"<div class='ct-result-card' style='text-align:center;--card-accent:{c_color};border-color:{c_border};padding:14px;'>"
            f"<div style='font-family:Orbitron,monospace;font-size:22px;font-weight:900;"
            f"color:{c_color};text-shadow:0 0 15px {c_neon};'>{conf_pct}%</div>"
            f"<div style='{lbl_style}'>Confiance</div></div>", unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='ct-result-card' style='text-align:center;--card-accent:{c_color};border-color:{c_border};padding:14px;'>"
            f"<div style='font-family:Exo 2,sans-serif;font-size:13px;font-weight:700;"
            f"color:#ffffff;'>{c_urg}</div>"
            f"<div style='{lbl_style}'>Urgence</div></div>", unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='ct-result-card' style='text-align:center;--card-accent:#42a5f5;padding:14px;'>"
            f"<div style='font-family:Share Tech Mono,monospace;font-size:14px;font-weight:700;"
            f"color:#90caf9;'>{ts_short}</div>"
            f"<div style='{lbl_style}'>Horodatage</div></div>", unsafe_allow_html=True,
        )

    # Alerte contextuelle dark
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if cls == "Tumor" and conf > 0.70:
        st.markdown(
            "<div class='al al-r'><div class='al-t'>🚨 CAS CRITIQUE DÉTECTÉ</div>"
            "<div class='al-b'>Tumeur rénale confirmée · Consultation oncologique urgente requise.</div></div>",
            unsafe_allow_html=True,
        )
    elif cls == "Tumor":
        st.markdown(
            "<div class='al al-r'><div class='al-t'>⚠️ Tumeur rénale détectée</div>"
            "<div class='al-b'>Confirmation par IRM et avis spécialisé requis sans délai.</div></div>",
            unsafe_allow_html=True,
        )
    elif cls == "Stone":
        st.markdown(
            "<div class='al al-o'><div class='al-t'>🪨 Lithiase rénale détectée</div>"
            "<div class='al-b'>Consultation urologique recommandée pour évaluation complète.</div></div>",
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
groq_status = "● GROQ OK"    if bool(KEYS["GROQ"]) else "○ GROQ OFFLINE"
ls_status   = "● LANGSMITH"  if ls_active          else "○ LANGSMITH OFF"

st.markdown(f"""
<div style='background:linear-gradient(135deg,#020818 0%,#0a1e35 40%,#0d2a45 100%);
            padding:20px 32px; display:flex; align-items:center; justify-content:space-between;
            border-bottom:1px solid rgba(66,165,245,0.3);
            box-shadow:0 4px 30px rgba(0,0,0,0.5), 0 0 60px rgba(13,71,161,0.2);
            position:relative; overflow:hidden;'>
  <div style='position:absolute;top:0;left:0;right:0;height:2px;
              background:linear-gradient(90deg,transparent,#42a5f5,#7c4dff,#00b4d8,transparent);'></div>
  <div style='display:flex; align-items:center; gap:16px;'>
    <div style='width:48px;height:48px;border-radius:12px;
                background:rgba(66,165,245,0.1);border:1px solid rgba(66,165,245,0.3);
                display:flex;align-items:center;justify-content:center;font-size:24px;
                box-shadow:0 0 20px rgba(66,165,245,0.2);'>🏥</div>
    <div>
      <div style='font-family:Orbitron,monospace;font-size:22px;font-weight:900;
                  background:linear-gradient(90deg,#42a5f5,#7c4dff,#00b4d8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  filter:drop-shadow(0 0 10px rgba(66,165,245,0.3));'>
        MEDICALScan AI
      </div>
      <div style='font-family:Share Tech Mono,monospace;font-size:10px;
                  color:rgba(66,165,245,0.6);letter-spacing:3px;text-transform:uppercase;margin-top:3px;'>
        Renal CT Scan Analysis · Groupe 2 · M2 IABD
      </div>
    </div>
  </div>
  <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
    <span style='font-family:Share Tech Mono,monospace;font-size:10px;padding:5px 12px;
                 border-radius:5px;border:1px solid {'rgba(0,230,118,0.4)' if bool(KEYS['GROQ']) else 'rgba(255,82,82,0.3)'};
                 color:{'#00e676' if bool(KEYS['GROQ']) else '#ff5252'};
                 background:{'rgba(0,80,40,0.3)' if bool(KEYS['GROQ']) else 'rgba(120,0,0,0.2)'};
                 letter-spacing:0.5px;'>{groq_status}</span>
    <span style='font-family:Share Tech Mono,monospace;font-size:10px;padding:5px 12px;
                 border-radius:5px;border:1px solid {'rgba(0,230,118,0.4)' if ls_active else 'rgba(66,165,245,0.2)'};
                 color:{'#00e676' if ls_active else '#5a8fbf'};
                 background:{'rgba(0,80,40,0.3)' if ls_active else 'rgba(13,71,161,0.1)'};'>{ls_status}</span>
    <span style='font-family:Share Tech Mono,monospace;font-size:10px;padding:5px 12px;
                 border-radius:5px;border:1px solid rgba(66,165,245,0.2);color:#5a8fbf;
                 background:rgba(13,71,161,0.1);'>KidneyClassifier v5 · AUC 1.00</span>
    <span style='font-family:Share Tech Mono,monospace;font-size:10px;padding:5px 12px;
                 border-radius:5px;border:1px solid rgba(66,165,245,0.2);color:#5a8fbf;
                 background:rgba(13,71,161,0.1);'>{datetime.datetime.now().strftime("%d/%m/%Y")}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# §12-15 ── Onglets ────────────────────────────────────────────────────────────
tab_scan, tab_chat, tab_sum, tab_mon = st.tabs([
    "🔬  Analyse CT",
    "💬  Assistant Médical",
    "📋  Résumé & Rapport",
    "📊  Monitoring",
])

# ─── §12  ANALYSE CT ──────────────────────────────────────────────────────────
with tab_scan:
    model, thr, model_err = _load_model(KEYS["MP"], KEYS["TP"])

    st.markdown("<div class='section-title'>🩻 Import de l'image CT</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        uploaded = st.file_uploader(
            "Importer une image CT rénale",
            type=["jpg","jpeg","png"],
            label_visibility="visible",
            key="ct_upload",
        )

        if uploaded:
            pil = Image.open(uploaded)
            st.markdown(
                f"<div class='ct-result-card' style='padding:12px 16px; margin-bottom:10px;'>"
                f"<div style='display:flex;align-items:center;gap:12px;'>"
                f"<div style='width:36px;height:36px;border-radius:8px;background:rgba(13,71,161,0.4);"
                f"border:1px solid rgba(66,165,245,0.3);display:flex;align-items:center;justify-content:center;font-size:18px;'>🩻</div>"
                f"<div style='flex:1;min-width:0;'>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:12px;color:#90caf9;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{uploaded.name}</div>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#5a8fbf;margin-top:2px;'>"
                f"{pil.size[0]}×{pil.size[1]} px · {pil.mode} · {uploaded.size//1024} Ko</div>"
                f"</div>"
                f"<div style='background:rgba(0,80,40,0.4);border:1px solid rgba(0,230,118,0.4);border-radius:6px;"
                f"padding:3px 10px;display:flex;align-items:center;gap:4px;'>"
                f"<span class='pulse-dot' style='width:6px;height:6px;margin:0;'></span>"
                f"<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#00e676;'>PRÊT</span>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )
            st.image(pil, caption="", use_container_width=True)
        else:
            st.markdown(
                "<div style='height:220px;display:flex;align-items:center;justify-content:center;"
                "flex-direction:column;gap:12px;background:rgba(4,14,38,0.6);"
                "border:1.5px dashed rgba(66,165,245,0.25);border-radius:14px;margin-top:4px;'>"
                "<div style='font-size:3rem;opacity:0.2;'>🩻</div>"
                "<div style='text-align:center;'>"
                "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#5a8fbf;font-weight:600;'>Aucune image chargée</div>"
                "<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#2d5a8e;margin-top:4px;'>Utilisez la zone d'import ci-dessus</div>"
                "</div></div>",
                unsafe_allow_html=True,
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
                "Importez un scan pour démarrer l'analyse</span></div></div>",
                unsafe_allow_html=True,
            )
        else:
            prev = st.session_state.get("res", {})
            if prev.get("_src") != uploaded.name:
                for k in ["res","chat","sys_prompt","summary","translations","audio","llm_traces"]:
                    st.session_state.pop(k, None)
                if   model_err == "DEMO": res = _predict_demo(uploaded.name)
                elif model_err:           res = None
                else:                     res = _predict_real(model, thr, pil)
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
                "📊 Probabilités par classe</div>",
                unsafe_allow_html=True,
            )

            # ── CORRECTION BARRES : opacity=1 partout, couleurs toujours visibles ──
            for c, p in sorted(res["probs"].items(), key=lambda x: -x[1]):
                cc       = CLASS_CFG[c]
                ip       = c == cls          # est-ce la classe prédite ?
                cc_color = cc["color"]
                cc_neon  = cc["neon"]
                cc_brd   = cc["border"]
                cc_emo   = cc["emoji"]

                # Nom de classe : couleur de la classe pour la gagnante, blanc cassé pour les autres
                nm_class  = "prob-top" if ip else ""
                # Pourcentage : idem
                pct_class = "prob-top" if ip else ""
                # Badge : TOP pour la gagnante, emoji sinon
                badge_txt = "TOP" if ip else cc_emo
                # Surbrillance de la ligne gagnante, fond blanc pour les autres
                row_style = (
                    "background:rgba(13,71,161,0.18);border-radius:6px;"
                    "padding:4px 8px;margin:6px -8px;"
                ) if ip else (
                    "background:rgba(58,109,207,0.92);border-radius:6px;"
                    "padding:4px 8px;margin:3px -8px;"
                )

                p_pct      = f"{p*100:.1f}"
                name_color = cc_color if ip else "#1a202c"
                pct_color  = cc_color if ip else "#1a202c"
                st.markdown(
                    f"<div class='prob-row' style='{row_style}'>"
                    f"<div class='prob-name {nm_class}' style='color:{name_color};'>{c}</div>"
                    f"<div class='prob-track'>"
                    f"<div class='prob-fill' style='width:{p_pct}%;"
                    f"background:linear-gradient(90deg,{cc_color},{cc_neon});"
                    f"box-shadow:0 0 8px {cc_neon};'></div>"
                    f"</div>"
                    f"<div class='prob-pct {pct_class}' style='color:{pct_color};'>{p_pct}%</div>"
                    f"<div class='prob-badge' style='background:{cc_color};color:#ffffff;"
                    f"border:1px solid {cc_brd};'>{badge_txt}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with ci:
            c_color  = cfg["color"]
            c_border = cfg["border"]
            suivi    = CTX[cls]["suivi"]
            interp_html = INTERP[cls].replace(
                "<strong>", "<strong style='color:#ffffff;font-weight:700;'>"
            )
            st.markdown(
                f"<div class='ct-result-card' style='border-color:{c_border};'>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;"
                f"letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;"
                f"padding-bottom:8px;border-bottom:1px solid rgba(66,165,245,0.2);'>"
                f"📝 Interprétation médicale automatique</div>"
                f"<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#dce8ff;line-height:1.8;'>"
                f"{interp_html}</div>"
                f"<div style='margin-top:16px;padding-top:12px;border-top:1px solid rgba(66,165,245,0.15);'>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;"
                f"text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;'>Suivi recommandé</div>"
                f"<div style='display:inline-block;font-family:Exo 2,sans-serif;font-size:13px;font-weight:700;"
                f"color:#ffffff;background:rgba(255,255,255,0.08);border:1px solid {c_color};"
                f"border-left:4px solid {c_color};border-radius:6px;padding:7px 14px;'>{suivi}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='info-box'>💬 Consultez l'onglet <strong>Assistant Médical</strong> pour des questions personnalisées sur ce résultat.</div>",
            unsafe_allow_html=True,
        )

# ─── §13  ASSISTANT MÉDICAL ───────────────────────────────────────────────────
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
            "Uploadez une image CT dans l'onglet Analyse CT</span></div></div>",
            unsafe_allow_html=True,
        )
    elif not KEYS["GROQ"]:
        st.error("Clé Groq manquante dans `.streamlit/secrets.toml`.")
    else:
        cls = res["class"]; conf = res["conf"]; cfg = CLASS_CFG[cls]

        pills = "".join([
            "<span class='pill pill-ok'>🔊 Audio</span>"      if tts_on    else "",
            "<span class='pill pill-ok'>🇩🇪 Traduction</span>"  if show_tr   else "",
            "<span class='pill pill-ok'>🟢 LangSmith</span>"   if ls_active else "",
        ])
        c_color  = cfg["color"]
        c_neon   = cfg["neon"]
        c_emoji  = cfg["emoji"]
        c_label  = cfg["label"]
        conf_pct = f"{conf*100:.1f}"
        st.markdown(
            f"<div style='background:rgba(4,14,38,0.7);border:1px solid rgba(66,165,245,0.2);"
            f"border-left:3px solid {c_color};border-radius:10px;padding:12px 16px;"
            f"display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:14px;'>"
            f"<div>"
            f"<div style='font-family:Orbitron,monospace;font-size:14px;font-weight:700;"
            f"color:#ffffff;'>{c_emoji} {cls} — {c_label}</div>"
            f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;margin-top:3px;'>"
            f"Confiance {conf_pct}% · {res['ts']} · {groq_model}</div></div>"
            f"<div style='display:flex;gap:6px;'>{pills}</div></div>",
            unsafe_allow_html=True,
        )

        for k, v in [("chat",[]),("translations",{}),("audio",{})]:
            st.session_state.setdefault(k, v)
        st.session_state.setdefault("sys_prompt", make_system_prompt(res))

        if not st.session_state["chat"]:
            welcome, _ = call_llm(
                messages=[{"role":"user","content":"Bonjour, je viens de recevoir le résultat de mon scanner rénal."}],
                system=st.session_state["sys_prompt"],
                run_name="welcome_message",
            )
            st.session_state["chat"].append({"role":"assistant","content":welcome})
            if show_tr:
                st.session_state["translations"][0] = translate_de(welcome)
            if tts_on:
                ab = tts(st.session_state["translations"].get(0,welcome) if "Allemand" in tts_lang else welcome, tts_lang)
                if ab: st.session_state["audio"][0] = ab

        ai = 0
        for msg in st.session_state["chat"]:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🏥"):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if show_tr and (de := st.session_state["translations"].get(ai,"")):
                        st.markdown(
                            f"<div class='de-box'><div class='de-t'>🇩🇪 Deutsche Übersetzung</div>"
                            f"<div class='de-b'>{de}</div></div>",
                            unsafe_allow_html=True,
                        )
                    if tts_on and (ab := st.session_state["audio"].get(ai)):
                        ll = "Deutsch" if "Allemand" in tts_lang else "Français"
                        st.markdown(f"<div class='audio-box'><div class='au-t'>🔊 Audio — {ll}</div></div>", unsafe_allow_html=True)
                        st.audio(ab, format="audio/mp3")
                    ai += 1

        if user_in := st.chat_input("Posez votre question sur ce résultat CT..."):
            st.session_state["chat"].append({"role":"user","content":user_in})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_in)

            with st.chat_message("assistant", avatar="🏥"):
                answer, metrics = call_llm(
                    messages=st.session_state["chat"],
                    system=st.session_state["sys_prompt"],
                    run_name="chatbot_answer",
                )
                st.markdown(answer)
                st.session_state["chat"].append({"role":"assistant","content":answer})
                idx = sum(1 for m in st.session_state["chat"] if m["role"]=="assistant") - 1

                if show_tr:
                    de = translate_de(answer)
                    st.session_state["translations"][idx] = de
                    st.markdown(f"<div class='de-box'><div class='de-t'>🇩🇪 Deutsche Übersetzung</div><div class='de-b'>{de}</div></div>", unsafe_allow_html=True)

                if tts_on:
                    txt = st.session_state["translations"].get(idx,answer) if "Allemand" in tts_lang else answer
                    ab  = tts(txt, tts_lang)
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
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        b1, b2, _ = st.columns([1,1.5,3])
        with b1:
            if st.button("🔄 Réinitialiser"):
                for k in ["chat","sys_prompt","translations","audio","summary"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with b2:
            if (len(st.session_state.get("chat",[]))>=2
                and not st.session_state.get("summary")
                and st.button("📋 Générer résumé")):
                st.session_state["summary"] = make_summary(st.session_state["chat"], res)
                st.success("✅ Résumé généré — onglet **Résumé & Rapport**.")

# ─── §14  RÉSUMÉ & RAPPORT ────────────────────────────────────────────────────
with tab_sum:
    res     = st.session_state.get("res")
    summary = st.session_state.get("summary")
    chat    = st.session_state.get("chat", [])

    if res is None:
        st.markdown(
            "<div style='height:280px;display:flex;align-items:center;justify-content:center;"
            "flex-direction:column;gap:12px;background:rgba(4,14,38,0.6);"
            "border:1px solid rgba(66,165,245,0.2);border-radius:16px;margin-top:10px;'>"
            "<div style='font-size:3rem;opacity:0.15;'>📋</div>"
            "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#3d6b9e;'>Aucun résultat disponible</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif len(chat) < 2:
        st.markdown(
            "<div class='info-box'>💬 Utilisez l'assistant médical pour générer un résumé de consultation.</div>",
            unsafe_allow_html=True,
        )
    else:
        if summary is None and auto_sum and KEYS["GROQ"]:
            summary = make_summary(chat, res)
            st.session_state["summary"] = summary

        if summary:
            cls      = res["class"]; conf = res["conf"]
            cfg      = CLASS_CFG[cls]
            c_color  = cfg["color"]
            c_neon   = cfg["neon"]
            c_border = cfg["border"]
            c_emoji  = cfg["emoji"]
            c_label  = cfg["label"]
            c_urg    = cfg["urgence"]
            conf_pct = f"{conf*100:.1f}"

            st.markdown("<div class='section-title'>📋 Compte Rendu de Consultation</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div class='ct-result-card' style='border-color:{c_border};padding:20px;margin-bottom:18px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px;'>"
                f"<div>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#90caf9;"
                f"text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;'>Compte Rendu — MEDICALScan AI</div>"
                f"<div style='font-family:Orbitron,monospace;font-size:20px;font-weight:900;"
                f"color:{c_color};text-shadow:0 0 15px {c_neon};'>"
                f"{c_emoji} {cls} — {c_label}</div>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#7aabd4;margin-top:4px;'>"
                f"Généré le {res['ts']} · Groupe 2 · M2 IABD</div>"
                f"</div>"
                f"<div style='display:flex;gap:10px;'>"
                f"<div style='text-align:center;background:rgba(13,71,161,0.2);border:1px solid {c_border};"
                f"border-radius:10px;padding:10px 16px;'>"
                f"<div style='font-family:Orbitron,monospace;font-size:18px;font-weight:900;"
                f"color:{c_color};text-shadow:0 0 12px {c_neon};'>{conf_pct}%</div>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#7aabd4;"
                f"text-transform:uppercase;letter-spacing:1px;margin-top:4px;'>Confiance</div></div>"
                f"<div style='text-align:center;background:rgba(13,71,161,0.2);border:1px solid {c_border};"
                f"border-radius:10px;padding:10px 16px;'>"
                f"<div style='font-family:Exo 2,sans-serif;font-size:13px;font-weight:700;color:#ffffff;'>{c_urg}</div>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#7aabd4;"
                f"text-transform:uppercase;letter-spacing:1px;margin-top:4px;'>Urgence</div></div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

            col_fr, col_de = st.columns([1,1], gap="large")

            with col_fr:
                st.markdown(
                    f"<div class='sum-card'>"
                    f"<div class='sum-label' style='color:#42a5f5;'>🇫🇷 Résumé médical — Français</div>"
                    f"<div class='sum-body'>{summary['fr'].replace(chr(10),'<br>')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if tts_on and st.button("🔊 Écouter en français"):
                    ab = tts(summary["fr"], "Français 🇫🇷")
                    if ab: st.audio(ab, format="audio/mp3")

            with col_de:
                st.markdown(
                    f"<div class='sum-de-card'>"
                    f"<div class='sum-label' style='color:#ffc107;'>🇩🇪 Zusammenfassung — Deutsch</div>"
                    f"<div class='sum-body' style='color:#ffe082;'>{summary['de'].replace(chr(10),'<br>')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if tts_on and st.button("🔊 Auf Deutsch anhören"):
                    ab = tts(summary["de"], "Allemand 🇩🇪")
                    if ab: st.audio(ab, format="audio/mp3")

            ctx_data = CTX[cls]
            export = (
                f"MEDICALScan AI — COMPTE RENDU · {res['ts']}\n"
                f"Groupe 2 · M2 IABD · KAMNO\n{'='*60}\n"
                f"Classe : {cls} ({cfg['label']}) | Confiance : {conf*100:.1f}%\n"
                f"Urgence : {ctx_data['urgence']} | Suivi : {ctx_data['suivi']}\n"
                f"{'='*60} RÉSUMÉ FR {'='*60}\n{summary['fr']}\n"
                f"{'='*60} ZUSAMMENFASSUNG DE {'='*60}\n{summary['de']}\n"
                f"{'='*60}\nRésultat IA — à confirmer par un professionnel de santé.\n"
            )
            c1, c2, _ = st.columns([1,1,3])
            with c1:
                fn = f"MEDICALScan_{res['ts'].replace(' ','_').replace(':','-')}.txt"
                st.download_button("⬇️ Télécharger .txt", data=export.encode("utf-8"), file_name=fn, mime="text/plain")
            with c2:
                if st.button("🔄 Régénérer"):
                    st.session_state.pop("summary",None); st.rerun()

# ─── §15  MONITORING ──────────────────────────────────────────────────────────
with tab_mon:
    traces = st.session_state.get("llm_traces", [])

    st.markdown("<div class='section-title'>📊 Monitoring LangSmith</div>", unsafe_allow_html=True)

    if not ls_active:
        st.markdown(
            "<div style='height:280px;display:flex;align-items:center;justify-content:center;"
            "flex-direction:column;gap:14px;background:rgba(4,14,38,0.6);"
            "border:1px solid rgba(66,165,245,0.2);border-radius:16px;'>"
            "<div style='font-size:3rem;opacity:0.15;'>📡</div>"
            "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#3d6b9e;text-align:center;'>"
            "LangSmith désactivé<br>"
            "<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#2d5a8e;'>"
            "Activez dans la barre latérale · Clé gratuite sur <strong style='color:#42a5f5;'>smith.langchain.com</strong>"
            "</span></div></div>",
            unsafe_allow_html=True,
        )
    elif not traces:
        st.markdown(
            "<div class='info-box'>Aucune trace — utilisez l'assistant médical pour commencer.</div>",
            unsafe_allow_html=True,
        )
    else:
        n_calls = len(traces)
        avg_lat = int(np.mean([t.get("latency_ms",0) for t in traces]))
        tot_tok = sum(t.get("tokens_in",0)+t.get("tokens_out",0) for t in traces)
        tot_out = sum(t.get("tokens_out",0) for t in traces)

        for col, val, lbl in zip(
            st.columns(4),
            [n_calls, f"{avg_lat} ms", f"{tot_tok:,}", tot_out],
            ["Appels LLM","Latence moy.","Tokens total","Tokens générés"],
        ):
            with col:
                st.markdown(
                    f"<div class='mon-card'>"
                    f"<div class='mon-val'>{val}</div>"
                    f"<div class='mon-lbl'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        import pandas as pd
        st.dataframe(
            pd.DataFrame([{
                "Heure":        t.get("timestamp",""),
                "Run":          t.get("run_name",""),
                "Modèle":       t.get("model",""),
                "Latence (ms)": t.get("latency_ms",0),
                "Tokens IN":    t.get("tokens_in",0),
                "Tokens OUT":   t.get("tokens_out",0),
            } for t in traces]),
            use_container_width=True, hide_index=True,
        )

        if len(traces) > 1:
            st.markdown(
                "<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#5a8fbf;"
                "text-transform:uppercase;letter-spacing:1.5px;margin:14px 0 6px;'>Latence par appel (ms)</div>",
                unsafe_allow_html=True,
            )
            st.bar_chart([t.get("latency_ms",0) for t in traces])

        st.markdown(
            "<div class='ct-result-card' style='margin-top:10px;'>"
            "<div style='font-family:Exo 2,sans-serif;font-size:13px;color:#8aabcc;'>"
            "🌐 Dashboard complet : "
            "<a href='https://smith.langchain.com' target='_blank'"
            " style='color:#42a5f5;font-weight:700;text-decoration:none;'>"
            "smith.langchain.com</a> → Projet : "
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
    "<div style='margin-top:8px;'>"
    "MEDICALScan AI · <span>KidneyClassifier v5</span> · MobileNetV2 · "
    "<span>Groupe 2 · M2 IABD · 2026</span> · KAMNO"
    "</div></div>",
    unsafe_allow_html=True,
)
