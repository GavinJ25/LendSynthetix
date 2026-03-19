"""
app.py — LendSynthetix + BioSentinel
─────────────────────────────────────────────────────────────────────
Single Streamlit entry point.

BioSentinel modes (auto-detected at startup):
  LIVE   — trained model found in biosentinel/saved_model/
           Real LSTM inference runs on session feature inputs.
  MANUAL — model not yet trained.
           Fraud score slider shown for demo / hackathon use.

Langflow:
  One flow, one POST, JSON response split into 4 agent panels.
"""

import streamlit as st
import requests
import json
import time
import numpy as np
from datetime import datetime
from app_logger import log_application, load_all

# ─────────────────────────────────────────────────────────────────────
# BIOSENTINEL — lazy import (model may not be trained yet)
# ─────────────────────────────────────────────────────────────────────
_BS_AVAILABLE = False
_BS_SCORER    = None

try:
    from biosentinel.inference import BioSentinelScorer, _demo_score, FEATURE_COLUMNS
    _scorer_instance = BioSentinelScorer()
    # probe — will raise FileNotFoundError if model not trained
    _scorer_instance._load()
    _BS_SCORER    = _scorer_instance
    _BS_AVAILABLE = True
except Exception:
    try:
        from biosentinel.inference import _demo_score, FEATURE_COLUMNS
    except Exception:
        FEATURE_COLUMNS = []
        def _demo_score(score=0.18):
            return {
                "fraud_score":     score,
                "signal":          "CLEAN" if score < 0.3 else "SUSPICIOUS",
                "emoji":           "🟢"    if score < 0.3 else "🟡",
                "predicted_class": "human",
                "class_probs":     {"human": 1-score, "bot": score*0.5,
                                    "duress": score*0.3, "coached": score*0.2},
                "flags":           ["✅ No behavioral anomalies detected"],
            }

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LendSynthetix | AI Underwriting",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────
if "dark_mode"    not in st.session_state: st.session_state.dark_mode    = True
if "bio_result"   not in st.session_state: st.session_state.bio_result   = None
if "last_results" not in st.session_state: st.session_state.last_results = None

def toggle_theme(): st.session_state.dark_mode = not st.session_state.dark_mode
dark = st.session_state.dark_mode

# ─────────────────────────────────────────────────────────────────────
# THEME TOKENS
# ─────────────────────────────────────────────────────────────────────
T = {
    "bg_base":       "#0A0C10"  if dark else "#F4F1EA",
    "bg_grad":       ("radial-gradient(ellipse at 20% 0%, #0F1A2E 0%, #0A0C10 50%)"
                      if dark else
                      "radial-gradient(ellipse at 20% 0%, #E8E0D0 0%, #F4F1EA 50%)"),
    "bg_card":       "#0D111A"  if dark else "#FFFFFF",
    "bg_input":      "#131720"  if dark else "#FAF8F5",
    "bg_step_wait":  "#0D111A"  if dark else "#F0EDE6",
    "bg_step_act":   "#1A1F2E"  if dark else "#FFF8E8",
    "bg_step_done":  "#0F1E14"  if dark else "#EEF7F1",
    "border":        "#1E2535"  if dark else "#D8D2C5",
    "border_gold":   "#C9A84C33" if dark else "#C9A84C55",
    "text_primary":  "#E8E4DA"  if dark else "#1A1A1A",
    "text_muted":    "#5A6478"  if dark else "#7A7A7A",
    "text_sub":      "#9AA3B5"  if dark else "#555555",
    "text_label":    "#9AA3B5"  if dark else "#444444",
    "gold":          "#C9A84C",
    "gold_light":    "#E8C87A",
    "scroll_track":  "#0A0C10"  if dark else "#E8E4DA",
    "aria_bg":       ("linear-gradient(135deg,#1A3A2A,#0F2A1A)"
                      if dark else "linear-gradient(135deg,#D4F5E2,#B8EDD0)"),
    "rex_bg":        ("linear-gradient(135deg,#3A1A1A,#2A0F0F)"
                      if dark else "linear-gradient(135deg,#FFE0E0,#FFD0D0)"),
    "clara_bg":      ("linear-gradient(135deg,#1A2A3A,#0F1A2A)"
                      if dark else "linear-gradient(135deg,#D8EAFF,#C5DDFF)"),
    "oracle_bg":     ("linear-gradient(135deg,#0D111A 0%,#111520 50%,#0D111A 100%)"
                      if dark else
                      "linear-gradient(135deg,#FFFFFF 0%,#FAFAF7 50%,#FFFFFF 100%)"),
    "metric_bg":     "#0D111A"  if dark else "#FFFFFF",
    "bio_panel_bg":  "#0D111A"  if dark else "#FFFFFF",
    "step_wait_text":"#5A6478"  if dark else "#8A8A8A",
    "step_act_text": "#C9A84C",
    "step_done_text":"#4ADE80"  if dark else "#16A34A",
    "step_wait_bdr": "#1E2535"  if dark else "#D8D2C5",
    "step_act_bdr":  "#C9A84C44",
    "step_done_bdr": "#4ADE8033" if dark else "#16A34A33",
    "shadow":        "0 4px 24px #00000030" if dark else "0 4px 24px #00000010",
    "shadow_lg":     "0 8px 40px #00000040" if dark else "0 8px 40px #00000015",
}

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

*,*::before,*::after{{box-sizing:border-box;}}

html,body,[data-testid="stAppViewContainer"]{{
    background:{T['bg_base']} !important;
    color:{T['text_primary']} !important;
    font-family:'DM Sans',sans-serif !important;
}}
[data-testid="stAppViewContainer"]{{
    background:{T['bg_grad']} !important;
}}
#MainMenu,footer{{visibility:hidden;}}
[data-testid="stToolbar"]{{display:none;}}
[data-testid="stHeader"]{{background:transparent !important;border-bottom:none !important;}}
[data-testid="stHeader"] [data-testid="stDecoration"]{{display:none;}}
[data-testid="stHeader"] [data-testid="stStatusWidget"]{{display:none;}}
.block-container{{padding:2rem 3rem !important;max-width:1600px !important;}}

::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{T['scroll_track']};}}
::-webkit-scrollbar-thumb{{background:{T['gold']};border-radius:3px;}}

/* ── HEADER ── */
.ls-header{{
    display:flex;align-items:center;justify-content:space-between;
    padding:1.5rem 2rem;border-bottom:1px solid {T['border']};
    margin-bottom:2rem;background:{T['bg_card']};
    border-radius:12px;position:relative;overflow:hidden;
    box-shadow:{T['shadow']};
}}
.ls-header::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,{T['gold']},{T['gold_light']},{T['gold']});
}}
.ls-logo{{font-family:'DM Serif Display',serif;font-size:2rem;color:{T['gold']};letter-spacing:-0.5px;}}
.ls-logo span{{color:{T['text_primary']};font-style:italic;}}
.ls-tagline{{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:{T['text_muted']};
             letter-spacing:0.15em;text-transform:uppercase;margin-top:0.2rem;}}
.ls-status-live{{display:flex;align-items:center;gap:0.5rem;
                 font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                 color:#4ADE80;letter-spacing:0.1em;}}
.live-dot{{width:8px;height:8px;background:#4ADE80;border-radius:50%;
           animation:pulse 2s ease-in-out infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:0.4;transform:scale(0.8);}}}}

/* ── SECTION TITLE ── */
.section-title{{
    font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
    letter-spacing:0.2em;text-transform:uppercase;color:{T['text_muted']};
    margin-bottom:1rem;display:flex;align-items:center;gap:0.8rem;
}}
.section-title::after{{content:'';flex:1;height:1px;background:{T['border']};}}

/* ── CARDS ── */
.form-card,.agent-card,.oracle-card,.bio-panel{{
    background:{T['bg_card']};border:1px solid {T['border']};
    border-radius:16px;padding:1.5rem;position:relative;
    overflow:hidden;box-shadow:{T['shadow']};
}}
.form-card{{padding:2rem;margin-bottom:1.5rem;}}
.form-card::before,.oracle-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,#C9A84C55,transparent);
}}

/* ── INPUTS ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
textarea{{
    background:{T['bg_input']} !important;border:1px solid {T['border']} !important;
    border-radius:8px !important;color:{T['text_primary']} !important;
    font-family:'DM Sans',sans-serif !important;font-size:0.9rem !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus{{
    border-color:{T['gold']} !important;box-shadow:0 0 0 2px #C9A84C22 !important;
}}
[data-testid="stSelectbox"]>div>div{{
    background:{T['bg_input']} !important;border:1px solid {T['border']} !important;
    border-radius:8px !important;color:{T['text_primary']} !important;
}}
label{{color:{T['text_label']} !important;font-size:0.8rem !important;
       font-family:'IBM Plex Mono',monospace !important;letter-spacing:0.05em !important;}}

/* ── BIOSENTINEL PANEL ── */
.bio-panel{{margin-bottom:1.5rem;}}
.bio-mode-badge{{
    display:inline-flex;align-items:center;gap:0.4rem;
    padding:0.25rem 0.7rem;border-radius:20px;
    font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1rem;
}}
.badge-live   {{background:#4ADE8022;color:#4ADE80;border:1px solid #4ADE8044;}}
.badge-manual {{background:#FBBF2422;color:#FBBF24;border:1px solid #FBBF2444;}}

.bio-score-ring{{
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;padding:1.2rem 0;
    border-radius:12px;background:{T['bg_input']};
    border:1px solid {T['border']};margin-bottom:1rem;
}}
.bio-score-label{{font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                  color:{T['text_muted']};letter-spacing:0.2em;text-transform:uppercase;
                  margin-bottom:0.4rem;}}
.bio-score-value{{font-family:'DM Serif Display',serif;font-size:2.8rem;line-height:1;margin-bottom:0.3rem;}}
.bio-score-tag{{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                padding:0.2rem 0.8rem;border-radius:20px;display:inline-block;letter-spacing:0.1em;}}

.score-clean .bio-score-value,.score-clean .bio-score-tag{{color:#4ADE80;}}
.score-clean .bio-score-tag{{background:#4ADE8022;border:1px solid #4ADE8044;}}
.score-suspicious .bio-score-value,.score-suspicious .bio-score-tag{{color:#FBBF24;}}
.score-suspicious .bio-score-tag{{background:#FBBF2422;border:1px solid #FBBF2444;}}
.score-high .bio-score-value,.score-high .bio-score-tag{{color:#F97316;}}
.score-high .bio-score-tag{{background:#F9731622;border:1px solid #F9731644;}}
.score-critical .bio-score-value,.score-critical .bio-score-tag{{color:#EF4444;}}
.score-critical .bio-score-tag{{background:#EF444422;border:1px solid #EF444444;}}

.bio-prob-row{{
    display:flex;align-items:center;gap:0.6rem;
    padding:0.35rem 0;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
    color:{T['text_sub']};border-bottom:1px solid {T['border']};
}}
.bio-prob-row:last-child{{border-bottom:none;}}
.bio-prob-label{{width:70px;flex-shrink:0;}}
.bio-prob-bar-bg{{flex:1;height:5px;background:{T['border']};border-radius:3px;overflow:hidden;}}
.bio-prob-bar{{height:100%;border-radius:3px;transition:width 0.4s ease;}}
.bar-human  {{background:#4ADE80;}}
.bar-bot    {{background:#EF4444;}}
.bar-duress {{background:#FBBF24;}}
.bar-coached{{background:#F97316;}}
.bio-prob-pct{{width:38px;text-align:right;flex-shrink:0;}}

.bio-flag{{
    display:flex;align-items:flex-start;gap:0.5rem;
    padding:0.5rem 0.6rem;border-radius:6px;margin-top:0.4rem;
    font-family:'IBM Plex Mono',monospace;font-size:0.63rem;
    line-height:1.5;color:{T['text_sub']};
    background:{T['bg_input']};border:1px solid {T['border']};
}}

.feature-grid{{
    display:grid;grid-template-columns:1fr 1fr;
    gap:0.3rem;margin-top:0.6rem;
}}
.feature-item{{
    font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
    color:{T['text_muted']};padding:0.3rem 0.5rem;
    background:{T['bg_input']};border-radius:4px;
    border:1px solid {T['border']};
}}
.feature-item span{{color:{T['text_sub']};float:right;}}

/* ── STEP ROWS ── */
.step-row{{
    display:flex;align-items:center;gap:1rem;padding:0.75rem 1rem;
    border-radius:8px;margin-bottom:0.4rem;
    font-family:'IBM Plex Mono',monospace;font-size:0.72rem;letter-spacing:0.05em;
}}
.step-waiting{{background:{T['bg_step_wait']};color:{T['step_wait_text']};border:1px solid {T['step_wait_bdr']};}}
.step-active {{background:{T['bg_step_act']};color:{T['step_act_text']};border:1px solid {T['step_act_bdr']};
               animation:stepglow 1.5s ease-in-out infinite;}}
.step-done   {{background:{T['bg_step_done']};color:{T['step_done_text']};border:1px solid {T['step_done_bdr']};}}
@keyframes stepglow{{0%,100%{{box-shadow:0 0 0 0 #C9A84C00;}}50%{{box-shadow:0 0 12px 2px #C9A84C22;}}}}
.step-icon{{font-size:1rem;min-width:1.2rem;text-align:center;}}

/* ── AGENT CARDS ── */
.agent-card{{border-radius:16px;padding:1.5rem;transition:border-color 0.2s ease;}}
.agent-card:hover{{border-color:{"#2A3348" if dark else "#B8A882"};}}
.agent-header{{display:flex;align-items:center;gap:1rem;margin-bottom:1.2rem;
               padding-bottom:1rem;border-bottom:1px solid {T['border']};}}
.agent-avatar{{width:42px;height:42px;border-radius:10px;display:flex;
               align-items:center;justify-content:center;font-size:1.2rem;
               font-family:'DM Serif Display',serif;font-weight:bold;flex-shrink:0;}}
.aria-avatar {{background:{T['aria_bg']};color:#4ADE80;border:1px solid #4ADE8033;}}
.rex-avatar  {{background:{T['rex_bg']};color:#EF4444;border:1px solid #EF444433;}}
.clara-avatar{{background:{T['clara_bg']};color:#60A5FA;border:1px solid #60A5FA33;}}
.agent-name  {{font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:{T['text_primary']};font-weight:500;letter-spacing:0.05em;}}
.agent-role  {{font-size:0.7rem;color:{T['text_muted']};margin-top:0.1rem;}}

.verdict-badge{{
    display:inline-flex;align-items:center;gap:0.4rem;padding:0.3rem 0.8rem;
    border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
    font-weight:500;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:1rem;
}}
.badge-approve{{background:#4ADE8022;color:#4ADE80;border:1px solid #4ADE8033;}}
.badge-risk   {{background:#FBBF2422;color:#FBBF24;border:1px solid #FBBF2433;}}
.badge-clear  {{background:#60A5FA22;color:#60A5FA;border:1px solid #60A5FA33;}}

.agent-output{{
    font-family:'IBM Plex Mono',monospace;font-size:0.72rem;line-height:1.7;
    color:{T['text_sub']};white-space:pre-wrap;word-break:break-word;
    max-height:540px;overflow-y:auto;padding-right:0.5rem;
}}
.agent-output::-webkit-scrollbar{{width:4px;}}
.agent-output::-webkit-scrollbar-thumb{{background:#C9A84C55;border-radius:2px;}}

/* ── ORACLE ── */
.oracle-card{{
    background:{T['oracle_bg']};border:1px solid {T['border_gold']};
    border-radius:20px;padding:2.5rem;margin-top:2rem;
    box-shadow:{T['shadow_lg']};
}}
.oracle-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,{T['gold']},transparent);
}}
.oracle-card::after{{
    content:'ORACLE';position:absolute;top:1.5rem;right:2rem;
    font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
    letter-spacing:0.3em;color:#C9A84C22;
}}
.oracle-title   {{font-family:'DM Serif Display',serif;font-size:1.5rem;color:{T['gold']};margin-bottom:0.3rem;}}
.oracle-subtitle{{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:{T['text_muted']};
                  letter-spacing:0.15em;text-transform:uppercase;margin-bottom:1.5rem;}}

.decision-banner{{border-radius:12px;padding:1.5rem 2rem;margin:1.5rem 0;display:flex;align-items:center;gap:1rem;}}
.decision-approved{{background:linear-gradient(135deg,#0F2A1A,#162A1F);border:1px solid #4ADE8044;}}
.decision-rejected{{background:linear-gradient(135deg,#2A0F0F,#2A1515);border:1px solid #EF444444;}}
.decision-counter {{background:linear-gradient(135deg,#2A1F0F,#2A1A0F);border:1px solid #FBBF2444;}}
.decision-escalate{{background:linear-gradient(135deg,#0F1A2A,#0F1F2A);border:1px solid #60A5FA44;}}
.decision-emoji{{font-size:2rem;}}
.decision-text{{font-family:'DM Serif Display',serif;font-size:1.8rem;}}
.decision-approved .decision-text{{color:#4ADE80;}}
.decision-rejected .decision-text{{color:#EF4444;}}
.decision-counter  .decision-text{{color:#FBBF24;}}
.decision-escalate .decision-text{{color:#60A5FA;}}
.oracle-output{{
    font-family:'IBM Plex Mono',monospace;font-size:0.73rem;line-height:1.8;
    color:{T['text_sub']};white-space:pre-wrap;word-break:break-word;
}}

/* ── METRICS ── */
.metric-strip{{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:2rem;}}
.metric-box{{
    background:{T['metric_bg']};border:1px solid {T['border']};
    border-radius:10px;padding:1rem 1.2rem;text-align:center;
    box-shadow:{T['shadow']};
}}
.metric-val  {{font-family:'DM Serif Display',serif;font-size:1.6rem;color:{T['text_primary']};line-height:1;margin-bottom:0.3rem;}}
.metric-label{{font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:{T['text_muted']};letter-spacing:0.1em;text-transform:uppercase;}}
.metric-fraud-clean     .metric-val{{color:#4ADE80;}}
.metric-fraud-suspicious.metric-val{{color:#FBBF24;}}
.metric-fraud-high      .metric-val{{color:#F97316;}}
.metric-fraud-critical  .metric-val{{color:#EF4444;}}

/* ── BUTTON ── */
[data-testid="stButton"]>button{{
    background:linear-gradient(135deg,#C9A84C,#B8922A) !important;
    color:#0A0C10 !important;font-family:'IBM Plex Mono',monospace !important;
    font-size:0.8rem !important;font-weight:500 !important;letter-spacing:0.1em !important;
    text-transform:uppercase !important;border:none !important;border-radius:8px !important;
    padding:0.7rem 2rem !important;width:100% !important;transition:all 0.2s ease !important;
}}
[data-testid="stButton"]>button:hover{{transform:translateY(-1px) !important;box-shadow:0 8px 24px #C9A84C33 !important;}}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"]{{
    background:{T['bg_card']};border-radius:10px;padding:0.3rem;border:1px solid {T['border']};
}}
[data-testid="stTabs"] [role="tab"]{{
    font-family:'IBM Plex Mono',monospace !important;font-size:0.7rem !important;
    letter-spacing:0.08em !important;color:{T['text_muted']} !important;
    border-radius:8px !important;padding:0.5rem 1rem !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{{
    background:{T['bg_input']} !important;color:{T['gold']} !important;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{{background:{T['bg_card']} !important;border-right:1px solid {T['border']} !important;}}
[data-testid="stSidebarNav"]{{display:block !important;visibility:visible !important;}}
[data-testid="stSidebarNav"] a{{font-family:'IBM Plex Mono',monospace !important;font-size:0.75rem !important;color:{T['text_muted']} !important;letter-spacing:0.05em !important;}}
[data-testid="stSidebarNav"] a:hover{{color:{T['gold']} !important;}}
[data-testid="stSidebarNav"] [aria-selected="true"]{{color:{T['gold']} !important;background:{T['bg_input']} !important;border-radius:6px !important;}}

hr{{border-color:{T['border']} !important;}}
[data-testid="stPageLink"] a{{font-family:'IBM Plex Mono',monospace !important;font-size:0.7rem !important;letter-spacing:0.08em !important;text-transform:uppercase !important;border:1px solid {T['border']} !important;border-radius:6px !important;padding:0.4rem 1rem !important;color:{T['text_muted']} !important;text-decoration:none !important;transition:all 0.2s ease !important;}}
[data-testid="stPageLink"] a:hover{{color:{T['gold']} !important;border-color:{T['gold']} !important !important;}}

.nav-bar{{
    display:flex;align-items:center;gap:0.5rem;
    margin-bottom:1.5rem;
}}
.nav-btn-active{{
    font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
    letter-spacing:0.1em;text-transform:uppercase;
    background:{T['gold']};color:#0A0C10;
    border:none;border-radius:6px;padding:0.45rem 1.2rem;
    cursor:pointer;font-weight:600;
}}
.nav-btn{{
    font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
    letter-spacing:0.1em;text-transform:uppercase;
    background:{T['bg_card']};color:{T['text_muted']};
    border:1px solid {T['border']};border-radius:6px;
    padding:0.45rem 1.2rem;cursor:pointer;
}}
.nav-btn:hover{{color:{T['gold']};border-color:{T['gold']};}}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def get_fraud_css(score: float) -> str:
    if score <= 0.30: return "score-clean"
    if score <= 0.60: return "score-suspicious"
    if score <= 0.80: return "score-high"
    return "score-critical"

def get_fraud_signal(score: float) -> tuple:
    if score <= 0.30: return "CLEAN",      "🟢", "metric-fraud-clean"
    if score <= 0.60: return "SUSPICIOUS", "🟡", "metric-fraud-suspicious"
    if score <= 0.80: return "HIGH RISK",  "🟠", "metric-fraud-high"
    return               "CRITICAL",   "🔴", "metric-fraud-critical"

def parse_decision_type(text: str) -> str:
    u = text.upper()
    if "REJECTED"    in u or "REJECT"       in u: return "rejected"
    if "COUNTER-OFFER" in u or "COUNTER OFFER" in u: return "counter"
    if "ESCALATED"   in u or "ESCALATE"     in u: return "escalate"
    return "approved"

def clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.lower().startswith("json"): raw = raw[4:]
    return raw.strip()

def call_all_agents(flow_id: str, application_text: str, api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input_value": application_text, "output_type": "chat",
               "input_type": "chat", "tweaks": {}}
    fallback = lambda msg: {k: msg for k in ("aria","rex","clara","oracle")}
    try:
        resp = requests.post(
            f"https://api.langflow.astra.datastax.com/lf/{flow_id}/api/v1/run/chat",
            headers=headers, json=payload, timeout=240)
        resp.raise_for_status()
        data = resp.json()
        raw  = data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        raw  = clean_json(raw)
        parsed = json.loads(raw)
        return {k: parsed.get(k, f"⚠️ {k.upper()} key missing.") for k in ("aria","rex","clara","oracle")}
    except json.JSONDecodeError:
        preview = raw[:800] if "raw" in dir() else "No output"
        return {"aria": f"⚠️ JSON parse failed.\n\nRaw output:\n\n{preview}",
                "rex":"Awaiting JSON fix...","clara":"Awaiting JSON fix...","oracle":"Awaiting JSON fix..."}
    except requests.exceptions.Timeout:
        return fallback("⚠️ Timed out (240s). Agents may be cold-starting — try again.")
    except requests.exceptions.HTTPError as e:
        return fallback(f"⚠️ HTTP {resp.status_code}: {e}")
    except requests.exceptions.RequestException as e:
        return fallback(f"⚠️ Network error: {e}")
    except (KeyError, IndexError):
        snippet = json.dumps(data, indent=2)[:600] if "data" in dir() else "No data"
        return fallback(f"⚠️ Unexpected response:\n\n{snippet}")


# ─────────────────────────────────────────────────────────────────────
# BIOSENTINEL UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────

# Colour map for each class — used in progress bar CSS overrides
_BAR_COLOURS = {
    "human":   "#4ADE80",
    "bot":     "#EF4444",
    "duress":  "#FBBF24",
    "coached": "#F97316",
}

def _score_card_html(score: float, signal: str, emoji: str, css: str, mode_live: bool) -> str:
    """Single-line HTML for the score ring — no indentation so markdown can't misread it."""
    badge = '<span class="bio-mode-badge badge-live">⚡ LIVE MODEL</span>' if mode_live else '<span class="bio-mode-badge badge-manual">🎛 MANUAL MODE</span>'
    return (f'<div class="bio-score-ring {css}">'
            f'{badge}'
            f'<div class="bio-score-label" style="margin-top:0.6rem;">BioSentinel Fraud Score</div>'
            f'<div class="bio-score-value">{score:.2f}</div>'
            f'<span class="bio-score-tag">{emoji} {signal}</span>'
            f'</div>')


def render_bio_panel(bio: dict, mode_live: bool):
    """
    Render BioSentinel panel using native Streamlit components.
    Avoids the indented-HTML-in-f-string markdown code-block bug.
    """
    score  = bio["fraud_score"]
    signal = bio["signal"]
    emoji  = bio["emoji"]
    probs  = bio["class_probs"]   # dict: {human, bot, duress, coached}
    flags  = bio["flags"]
    css    = get_fraud_css(score)

    # ── Score ring (minimal single-line HTML, safe from markdown parser) ──
    st.markdown(_score_card_html(score, signal, emoji, css, mode_live),
                unsafe_allow_html=True)

    # ── Class probabilities — native Streamlit progress bars ──
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;"
        f"color:{T['text_muted']};letter-spacing:0.1em;text-transform:uppercase;"
        f"margin:0.8rem 0 0.5rem 0;'>Class Probabilities</div>",
        unsafe_allow_html=True
    )

    for cls, p in probs.items():
        pct     = int(round(p * 100))
        colour  = _BAR_COLOURS.get(cls, "#C9A84C")
        label_c, pct_c = st.columns([3, 1])

        with label_c:
            # Coloured label + progress bar
            st.markdown(
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:{colour};letter-spacing:0.06em;'>{cls}</span>",
                unsafe_allow_html=True
            )
            # Override Streamlit's progress bar colour via a one-liner wrapper
            st.markdown(
                f"<div style='height:6px;background:{T['border']};border-radius:3px;overflow:hidden;margin-top:2px;'>"
                f"<div style='height:100%;width:{pct}%;background:{colour};border-radius:3px;transition:width 0.4s ease;'></div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with pct_c:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:{colour};text-align:right;padding-top:2px;'>{pct}%</div>",
                unsafe_allow_html=True
            )

    # ── Behavioral flags — native Streamlit alerts ──
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;"
        f"color:{T['text_muted']};letter-spacing:0.1em;text-transform:uppercase;"
        f"margin:1rem 0 0.4rem 0;'>Behavioral Flags</div>",
        unsafe_allow_html=True
    )

    for flag in flags:
        if flag.startswith("🔴") or flag.startswith("❌"):
            st.error(flag, icon=None)
        elif flag.startswith("⚠️") or flag.startswith("🟡") or flag.startswith("👥") or flag.startswith("🤖"):
            st.warning(flag, icon=None)
        elif flag.startswith("✅"):
            st.success(flag, icon=None)
        else:
            st.info(flag, icon=None)


def render_feature_inputs():
    """
    Live mode: show a form to manually enter the 16 behavioral features.
    In production replace this with real JS-captured data.
    """
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                color:{T['text_muted']};letter-spacing:0.1em;text-transform:uppercase;
                margin-bottom:0.8rem;">
        Session Feature Inputs
    </div>
    """, unsafe_allow_html=True)

    defaults = {
        "inter_key_delay": 120.0, "key_hold_duration": 80.0,
        "keystroke_variance": 40.0, "backspace_rate": 0.05,
        "typing_speed_wpm": 45.0, "mouse_velocity": 300.0,
        "mouse_acceleration": 50.0, "click_pressure_var": 60.0,
        "scroll_jitter": 25.0, "avg_field_dwell": 8000.0,
        "field_dwell_var": 3000.0, "tab_order_deviations": 0.0,
        "copy_paste_count": 0.0, "total_session_time": 180.0,
        "hesitation_count": 2.0, "first_field_latency": 4.0,
    }

    features = {}
    cols = st.columns(2)
    for i, (key, default) in enumerate(defaults.items()):
        with cols[i % 2]:
            features[key] = st.number_input(
                key.replace("_", " ").title(),
                value=default, step=1.0,
                key=f"feat_{key}"
            )
    return features


# ─────────────────────────────────────────────────────────────────────
# MOCK OUTPUTS
# ─────────────────────────────────────────────────────────────────────
MOCK = {
"aria": """════════════════════════════════════════
        ARIA — SALES ASSESSMENT REPORT
════════════════════════════════════════
APPLICANT: Priya Sharma
ASSESSED BY: ARIA, Senior Loan Sales Specialist

SECTION 1: INCOME & REPAYMENT ANALYSIS
Monthly Net Income       : Rs 95,000
Requested Loan Amount    : Rs 400,000
Estimated Monthly EMI    : Rs 8,624  (@ 10.5% p.a., 60 months)
EMI-to-Income Ratio      : 9.1%   -> HEALTHY
Combined EMI Burden      : Rs 16,624 (17.5% of income)
Loan-to-Annual-Income    : 3.5x   -> GREEN

SECTION 2: LOAN PURPOSE
Stated Purpose           : Home Renovation
Purpose Category         : Productive
Viability Score          : 9/10

SECTION 3: EMPLOYMENT
Employment Type          : Salaried | 8 years
Employer Stability       : STABLE

SECTION 4: SCORECARD
Income Adequacy          : 24/25
Loan Purpose             : 18/20
Employment Stability     : 25/25
Repayment Capacity       : 28/30
TOTAL SALES SCORE        : 95/100

VERDICT: APPROVE  |  Confidence: 9/10
Strengths: Strong income cushion, asset-backed purpose,
8-year employment tenure, conservative LTI of 3.5x.
════════════════════════════════════════""",

"rex": """════════════════════════════════════════
         REX — RISK ASSESSMENT REPORT
════════════════════════════════════════
APPLICANT: Priya Sharma
ASSESSED BY: REX, Chief Credit Risk Officer

SECTION 1: CREDITWORTHINESS
Credit Score             : 740 (GOOD — STANDARD weight)

SECTION 2: DEBT BURDEN
Current EMIs             : Rs 8,000
New EMI                  : Rs 8,624
Total Post-Loan EMI      : Rs 16,624
DTI                      : 17.5% -> LOW / ACCEPTABLE

SECTION 3: BEHAVIORAL FRAUD SIGNAL
BioSentinel Score        : 0.18 / 1.00  (CLEAN)
Predicted Class          : human  (P=0.82)
Bot Probability          : 0.05  — no automation detected
Duress Probability       : 0.08  — no coercion signals
Coached Probability      : 0.05  — no third-party involvement
Behavioral Flags         : None detected

SECTION 4: DEFAULT PROBABILITY
Estimated PD             : 2.8%  (ACCEPTABLE)
  Credit Score           : -3.2%
  DTI                    : -2.1%
  Employment             : -1.8%
  Purpose                : +0.5%
  Fraud Signal           : -0.6%

SECTION 5: RED FLAGS
No red flags identified.

SCORECARD
Credit Health            : 22/25
Debt Sustainability      : 24/25
Behavioral Signal        : 25/25
Employment Risk          : 24/25
TOTAL RISK SCORE         : 95/100

VERDICT: LOW RISK  |  Escalation: NO
════════════════════════════════════════""",

"clara": """════════════════════════════════════════
      CLARA — COMPLIANCE REPORT
════════════════════════════════════════
APPLICANT: Priya Sharma
FRAMEWORK: RBI | PMLA 2002 | Fair Practices Code

KYC STATUS               : COMPLETE (ID, Address, PAN confirmed)
DTI vs RBI Limit         : 17.5% vs 50% limit  -> WITHIN LIMIT
Fair Lending             : COMPLIANT (financial metrics only)
AML Screening            : LOW RISK
  BioSentinel Score 0.18 below PMLA threshold 0.70
  Mandatory escalation   : NO
  Loan purpose clarity   : CLEAR

FINDINGS
  CLEAR — KYC complete
  CLEAR — DTI within RBI S4.2 threshold
  CLEAR — Fair lending upheld
  CLEAR — No AML/PMLA triggers

Total BLOCKs             : 0
Total FLAGs              : 0
COMPLIANCE STATUS        : CLEAR
Defensibility Rating     : HIGH
════════════════════════════════════════""",

"oracle": """╔══════════════════════════════════════════════════════╗
║          ORACLE — FINAL LOAN DECISION MEMO           ║
╚══════════════════════════════════════════════════════╝
APPLICANT   : Priya Sharma
APP ID      : LN-48291037
AGENTS      : ARIA | REX | CLARA

AGENT VERDICTS
  ARIA    APPROVE    95/100   9/10
  REX     LOW RISK   95/100   9/10
  CLARA   CLEAR      N/A      10/10
Conflicts   : NONE — Full consensus

KEY METRICS
  Monthly Income       : Rs 95,000
  Loan Amount          : Rs 400,000
  Credit Score         : 740
  Post-Loan DTI        : 17.5%
  BioSentinel Score    : 0.18 (CLEAN)
  Probability of Default: 2.8%
  Compliance           : CLEAR

FINAL DECISION: APPROVED
  Decision Driver      : CONSENSUS (all agents aligned)
  Decision Trigger     : Low DTI + Clean BioSentinel + Score 740
  Approved Amount      : Rs 400,000
  Interest Rate        : 10.5% p.a.
  Tenure               : 60 months
  Monthly EMI          : Rs 8,624
  Disbursement         : KYC verification on file

ORACLE REASONING
All agents aligned with no conflicts. ARIA scored 95/100
on repayment capacity and employment stability. REX confirmed
PD of 2.8% supported by DTI of 17.5% and a clean BioSentinel
behavioral score of 0.18 — genuine human interaction confirmed.
CLARA returned zero compliance impediments. BioSentinel clean
signal was decisive for REX, removing any residual fraud risk.
Full approval at requested amount. High confidence.

AUDIT TRAIL
  Regulations: RBI S4.2 | PMLA 2002 S12 | Fair Practices Code
  Defensibility: HIGH | Human Override: NO
  Valid for 30 days from date of issuance.

APPLICANT COMMUNICATION
Dear Priya, your loan of Rs 4,00,000 is approved at 10.5%
p.a. over 60 months (EMI: Rs 8,624/month). Please submit
KYC documents to your branch to initiate disbursement.
════════════════════════════════════════════════════════"""
}


# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([4, 1])
with h1:
    st.markdown(f"""
    <div class="ls-header">
        <div>
            <div class="ls-logo">Lend<span>Synthetix</span></div>
            <div class="ls-tagline">
                ⚖️ AI Underwriting War Room &nbsp;|&nbsp; Powered by ORACLE
                &nbsp;|&nbsp;
                {"⚡ BioSentinel LIVE" if _BS_AVAILABLE else "🎛 BioSentinel MANUAL"}
            </div>
        </div>
        <div class="ls-status-live"><div class="live-dot"></div>AGENTS ONLINE</div>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    st.button(
        "☀️  Light" if dark else "🌙  Dark",
        on_click=toggle_theme, key="theme_btn"
    )



# ─────────────────────────────────────────────────────────────────────
# PAGE NAVIGATION
# ─────────────────────────────────────────────────────────────────────
nav_cols = st.columns([1, 1, 6])
with nav_cols[0]:
    st.page_link("app.py",                        label="⚖️  Underwriting", use_container_width=True)
with nav_cols[1]:
    st.page_link("pages/1_Analytics.py",       label="📊  Analytics",    use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.65rem;"
                f"letter-spacing:0.2em;color:{T['text_muted']};text-transform:uppercase;"
                f"margin-bottom:1rem;\">⚙ Configuration</div>", unsafe_allow_html=True)

    try:
        default_key = st.secrets.get("LANGFLOW_API_KEY", "")
        default_fid = st.secrets.get("LANGFLOW_FLOW_ID", "")
    except Exception:
        default_key = default_fid = ""

    api_key = st.text_input("Langflow API Key", value=default_key,
                             type="password", placeholder="Auto-loaded from secrets.toml")
    flow_id = st.text_input("Flow ID", value=default_fid,
                             placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

    st.markdown("---")
    demo_mode = st.toggle("🎭 Demo Mode", value=True,
                           help="Use mock outputs — no API key needed")

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                color:{T['text_muted']};margin-top:1.5rem;line-height:2.2;opacity:0.6;">
    ARIA &nbsp;&nbsp;·&nbsp; Sales Agent<br>
    REX &nbsp;&nbsp;&nbsp;·&nbsp; Risk Agent<br>
    CLARA &nbsp;·&nbsp; Compliance Agent<br>
    ORACLE ·&nbsp; Decision Engine<br>
    ────────────────<br>
    BioSentinel: {"⚡ LIVE" if _BS_AVAILABLE else "🎛 MANUAL (train model)"}<br>
    1 Flow · 1 API call · JSON → 4 panels
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# MAIN FORM
# ─────────────────────────────────────────────────────────────────────
col_form, col_bio = st.columns([2, 1], gap="large")

with col_form:
    st.markdown('<div class="section-title">Loan Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        applicant_name = st.text_input("Full Name",            placeholder="e.g. Priya Sharma")
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, step=1000,  value=95000)
        loan_amount    = st.number_input("Loan Amount (₹)",    min_value=0, step=10000, value=400000)
        loan_purpose   = st.text_input("Loan Purpose",         placeholder="e.g. Home Renovation")
    with fc2:
        credit_score   = st.number_input("Credit Score",        min_value=300, max_value=900, value=740)
        existing_emis  = st.number_input("Existing EMIs (₹/mo)",min_value=0, step=500,  value=8000)
        emp_type       = st.selectbox("Employment Type", ["Salaried","Self-Employed","Business Owner"])
        years_employed = st.number_input("Years Employed",      min_value=0, max_value=40, value=8)
    notes = st.text_area("Additional Notes",
                          placeholder="Any extra context for the underwriting committee...",
                          height=65)
    st.markdown('</div>', unsafe_allow_html=True)

with col_bio:
    st.markdown('<div class="section-title">BioSentinel</div>', unsafe_allow_html=True)

    if _BS_AVAILABLE:
        # ── LIVE MODE — show feature inputs, score on submit ──
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                    color:{T['text_muted']};margin-bottom:0.8rem;line-height:1.7;">
        Model loaded. Enter behavioral features captured from the session,
        or use the JS capture script to auto-populate.
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📊 Session Feature Inputs", expanded=False):
            live_features = render_feature_inputs()

        # Show last result if available
        if st.session_state.bio_result:
            render_bio_panel(st.session_state.bio_result, mode_live=True)
        else:
            st.markdown(f"""
            <div class="bio-panel" style="text-align:center;padding:2rem 1rem;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">⚡</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                            color:{T['text_muted']};">Submit form to run BioSentinel</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # ── MANUAL MODE — slider ──
        st.markdown('<div class="bio-panel">', unsafe_allow_html=True)
        st.markdown(f"""
        <span class="bio-mode-badge badge-manual">🎛 MANUAL MODE</span>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                    color:{T['text_muted']};margin-bottom:0.8rem;line-height:1.7;">
        Train the model first to enable live scoring.<br>
        Run: <span style="color:{T['gold']};">python -m biosentinel.train</span>
        </div>
        """, unsafe_allow_html=True)
        fraud_score_manual = st.slider("Fraud Score Override", 0.0, 1.0, 0.18, 0.01,
                                        help="Simulates BioSentinel output for demo")
        manual_bio = _demo_score(fraud_score_manual)
        render_bio_panel(manual_bio, mode_live=False)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SUBMIT
# ─────────────────────────────────────────────────────────────────────
run = st.button("⚖️  Convene Underwriting Committee")

if run:
    # ── Resolve BioSentinel score ──
    if _BS_AVAILABLE:
        bio = _BS_SCORER.score_from_features(live_features)
        st.session_state.bio_result = bio
        # Re-render the panel with fresh result
        with col_bio:
            render_bio_panel(bio, mode_live=True)
    else:
        bio = _demo_score(fraud_score_manual)

    fraud_score = bio["fraud_score"]
    signal, emoji, fraud_css = get_fraud_signal(fraud_score)

    # ── Build application text for Langflow ──
    app_text = f"""
LOAN APPLICATION
================
Applicant Name        : {applicant_name or 'Anonymous'}
Monthly Income        : Rs {monthly_income:,}
Loan Amount           : Rs {loan_amount:,}
Loan Purpose          : {loan_purpose or 'Not specified'}
Credit Score          : {credit_score}
Existing EMIs         : Rs {existing_emis:,}/month
Employment Type       : {emp_type}
Years Employed        : {years_employed}
Additional Notes      : {notes or 'None'}

BIOSENTINEL ANALYSIS
====================
Fraud Score           : {fraud_score:.4f}
Signal                : {bio['signal']}
Predicted Class       : {bio['predicted_class']}
P(human)              : {bio['class_probs']['human']:.4f}
P(bot)                : {bio['class_probs']['bot']:.4f}
P(duress)             : {bio['class_probs']['duress']:.4f}
P(coached)            : {bio['class_probs']['coached']:.4f}
Behavioral Flags      :
{chr(10).join('  - ' + f for f in bio['flags'])}
""".strip()

    st.markdown("---")

    # ── Quick metrics (now 5 — includes fraud score) ──
    new_emi = round((loan_amount * (0.105/12)) / (1-(1+0.105/12)**-60)) if loan_amount else 0
    dti     = round(((existing_emis + new_emi) / monthly_income) * 100, 1) if monthly_income else 0
    lti     = round(loan_amount / (monthly_income * 12), 1) if monthly_income else 0

    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-box">
            <div class="metric-val">Rs {new_emi:,}</div>
            <div class="metric-label">Est. EMI / mo</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{dti}%</div>
            <div class="metric-label">Post-Loan DTI</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{lti}x</div>
            <div class="metric-label">Loan-to-Income</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{credit_score}</div>
            <div class="metric-label">Credit Score</div>
        </div>
        <div class="metric-box {fraud_css}">
            <div class="metric-val">{fraud_score:.2f}</div>
            <div class="metric-label">{emoji} Fraud Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fetch agent outputs ──
    if demo_mode:
        time.sleep(0.4)
        results = MOCK
    else:
        if not api_key or not flow_id:
            st.error("⚠️ Enter your Langflow API Key and Flow ID in the sidebar.")
            st.stop()

        st.markdown('<div class="section-title">Processing</div>', unsafe_allow_html=True)
        pb = st.empty()
        steps = [("🟢","ARIA  — Sales Assessment"),("🔴","REX   — Risk Analysis"),
                 ("🔵","CLARA — Compliance Audit"),("⚡","ORACLE — Final Decision")]

        def render_steps(done: int):
            html = ""
            for i,(icon,label) in enumerate(steps):
                cls = "step-done" if i<done else ("step-active" if i==done else "step-waiting")
                s   = "✅" if i<done else ("⏳" if i==done else "○")
                html += (f'<div class="step-row {cls}"><span class="step-icon">{icon}</span>'
                         f'{label}<span style="margin-left:auto">{s}</span></div>')
            pb.markdown(html, unsafe_allow_html=True)

        render_steps(0)
        results = call_all_agents(flow_id, app_text, api_key)
        render_steps(4)

    st.session_state.last_results = results

    # ── Persist to analytics log ──
    try:
        log_application(
            applicant_name=applicant_name,
            monthly_income=monthly_income,
            loan_amount=loan_amount,
            loan_purpose=loan_purpose,
            credit_score=credit_score,
            existing_emis=existing_emis,
            emp_type=emp_type,
            years_employed=years_employed,
            bio=bio,
            results=results,
            new_emi=new_emi,
            dti=dti,
            lti=lti,
        )
    except Exception:
        pass   # never break the main flow over logging

    aria_out   = results["aria"]
    rex_out    = results["rex"]
    clara_out  = results["clara"]
    oracle_out = results["oracle"]

    # ── Agent Cards ──
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Agent Assessments</div>',
                unsafe_allow_html=True)
    ca, cr, cc = st.columns(3, gap="medium")

    with ca:
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-header">
                <div class="agent-avatar aria-avatar">A</div>
                <div><div class="agent-name">ARIA</div>
                     <div class="agent-role">Senior Loan Sales Specialist</div></div>
            </div>
            <div class="verdict-badge badge-approve">✅ Sales Assessment</div>
            <div class="agent-output">{aria_out}</div>
        </div>""", unsafe_allow_html=True)

    with cr:
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-header">
                <div class="agent-avatar rex-avatar">R</div>
                <div><div class="agent-name">REX</div>
                     <div class="agent-role">Chief Credit Risk Officer</div></div>
            </div>
            <div class="verdict-badge badge-risk">📊 Risk Assessment</div>
            <div class="agent-output">{rex_out}</div>
        </div>""", unsafe_allow_html=True)

    with cc:
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-header">
                <div class="agent-avatar clara-avatar">C</div>
                <div><div class="agent-name">CLARA</div>
                     <div class="agent-role">Regulatory Compliance Officer</div></div>
            </div>
            <div class="verdict-badge badge-clear">⚖️ Compliance Status</div>
            <div class="agent-output">{clara_out}</div>
        </div>""", unsafe_allow_html=True)

    # ── ORACLE ──
    st.markdown('<div class="section-title" style="margin-top:2rem;">Final Decision</div>',
                unsafe_allow_html=True)
    d_type = parse_decision_type(oracle_out)
    d_map  = {"approved":("decision-approved","✅","LOAN APPROVED"),
              "rejected": ("decision-rejected","❌","LOAN REJECTED"),
              "counter":  ("decision-counter", "⚠️","COUNTER-OFFER"),
              "escalate": ("decision-escalate","🔍","ESCALATED FOR REVIEW")}
    d_cls, d_emoji, d_text = d_map[d_type]

    st.markdown(f"""
    <div class="oracle-card">
        <div class="oracle-title">ORACLE</div>
        <div class="oracle-subtitle">Chief Underwriting Engine — Final Determination</div>
        <div class="decision-banner {d_cls}">
            <div class="decision-emoji">{d_emoji}</div>
            <div class="decision-text">{d_text}</div>
        </div>
        <div class="oracle-output">{oracle_out}</div>
    </div>""", unsafe_allow_html=True)

    # ── Export ──
    st.markdown("---")
    with st.expander("📋  Export Raw Outputs"):
        t1,t2,t3,t4 = st.tabs(["ARIA","REX","CLARA","ORACLE"])
        with t1: st.code(aria_out,   language=None)
        with t2: st.code(rex_out,    language=None)
        with t3: st.code(clara_out,  language=None)
        with t4: st.code(oracle_out, language=None)

        bio_summary = (
            f"BioSentinel Score  : {bio['fraud_score']}\n"
            f"Signal             : {bio['signal']}\n"
            f"Predicted Class    : {bio['predicted_class']}\n"
            f"Class Probs        : {bio['class_probs']}\n"
            f"Flags              :\n" +
            "\n".join(f"  {f}" for f in bio["flags"])
        )
        full = "\n\n".join([
            f"BIOSENTINEL\n{'='*60}\n{bio_summary}",
            f"ARIA\n{'='*60}\n{aria_out}",
            f"REX\n{'='*60}\n{rex_out}",
            f"CLARA\n{'='*60}\n{clara_out}",
            f"ORACLE\n{'='*60}\n{oracle_out}",
        ])
        st.download_button("⬇️  Download Full Report (.txt)", data=full,
                           file_name=f"underwriting_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                           mime="text/plain")

# ── Idle state ──
else:
    bio_status = "⚡ BioSentinel LIVE — LSTM model loaded" if _BS_AVAILABLE else "🎛 BioSentinel MANUAL — run python -m biosentinel.train to enable live mode"
    bio_color  = "#4ADE80" if _BS_AVAILABLE else T['text_muted']
    st.markdown(f"""
    <div style="text-align:center;padding:4rem 2rem;border:1px dashed {T['border']};
                border-radius:16px;margin-top:1rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">⚖️</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;
                    color:{T['text_primary']};margin-bottom:0.5rem;">Awaiting Application</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                    color:{T['text_muted']};letter-spacing:0.1em;">
            Fill in the form above and convene the committee
        </div>
        <div style="display:flex;justify-content:center;gap:3rem;margin-top:2.5rem;">
            <div style="text-align:center;"><div style="font-size:1.4rem;">🟢</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                            color:{T['text_muted']};margin-top:0.4rem;">ARIA</div></div>
            <div style="text-align:center;"><div style="font-size:1.4rem;">🔴</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                            color:{T['text_muted']};margin-top:0.4rem;">REX</div></div>
            <div style="text-align:center;"><div style="font-size:1.4rem;">🔵</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                            color:{T['text_muted']};margin-top:0.4rem;">CLARA</div></div>
            <div style="text-align:center;"><div style="font-size:1.4rem;">⚡</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
                            color:{T['text_muted']};margin-top:0.4rem;">ORACLE</div></div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                    color:{bio_color};margin-top:2rem;letter-spacing:0.05em;">
            {bio_status}
        </div>
    </div>
    """, unsafe_allow_html=True)