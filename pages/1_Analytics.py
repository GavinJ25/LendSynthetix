"""
pages/1_Analytics.py
─────────────────────────────────────────────────────────────────────
LendSynthetix Analytics Dashboard

Reads from data/applications.json (written by app_logger.py).
Shows aggregate KPIs, decision breakdown, fraud signal distribution,
DTI / credit score charts, agent score trends, and full history table.

Seeded with 30 realistic synthetic records so the dashboard looks
live from day one — no real submissions required for demo.
"""

import streamlit as st
import json
import os
import sys
import random
import math
from datetime import datetime, timedelta

# ── Allow importing from project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app_logger import load_all, clear_log, LOG_PATH

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LendSynthetix | Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# THEME — inherit from session state if available
# ─────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

T = {
    "bg_base":    "#0A0C10"  if dark else "#F4F1EA",
    "bg_grad":    ("radial-gradient(ellipse at 20% 0%, #0F1A2E 0%, #0A0C10 50%)"
                   if dark else
                   "radial-gradient(ellipse at 20% 0%, #E8E0D0 0%, #F4F1EA 50%)"),
    "bg_card":    "#0D111A"  if dark else "#FFFFFF",
    "bg_input":   "#131720"  if dark else "#FAF8F5",
    "border":     "#1E2535"  if dark else "#D8D2C5",
    "text":       "#E8E4DA"  if dark else "#1A1A1A",
    "text_muted": "#5A6478"  if dark else "#7A7A7A",
    "text_sub":   "#9AA3B5"  if dark else "#555555",
    "gold":       "#C9A84C",
    "shadow":     "0 4px 24px #00000030" if dark else "0 4px 24px #00000010",
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
    color:{T['text']} !important;
    font-family:'DM Sans',sans-serif !important;
}}
[data-testid="stAppViewContainer"]{{background:{T['bg_grad']} !important;}}
#MainMenu,footer{{visibility:hidden;}}
[data-testid="stToolbar"]{{display:none;}}
[data-testid="stHeader"]{{background:transparent !important;border-bottom:none !important;}}
[data-testid="stHeader"] [data-testid="stDecoration"]{{display:none;}}
[data-testid="stHeader"] [data-testid="stStatusWidget"]{{display:none;}}
.block-container{{padding:2rem 3rem !important;max-width:1600px !important;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{T['bg_base']};}}
::-webkit-scrollbar-thumb{{background:{T['gold']};border-radius:3px;}}

.ls-header{{
    display:flex;align-items:center;justify-content:space-between;
    padding:1.5rem 2rem;border-bottom:1px solid {T['border']};
    margin-bottom:2rem;background:{T['bg_card']};
    border-radius:12px;position:relative;overflow:hidden;box-shadow:{T['shadow']};
}}
.ls-header::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,{T['gold']},{T['gold']},{T['gold']});
}}
.ls-logo{{font-family:'DM Serif Display',serif;font-size:2rem;color:{T['gold']};letter-spacing:-0.5px;}}
.ls-logo span{{color:{T['text']};font-style:italic;}}
.ls-tagline{{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:{T['text_muted']};
             letter-spacing:0.15em;text-transform:uppercase;margin-top:0.2rem;}}

.section-title{{
    font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
    letter-spacing:0.2em;text-transform:uppercase;color:{T['text_muted']};
    margin-bottom:1rem;display:flex;align-items:center;gap:0.8rem;
}}
.section-title::after{{content:'';flex:1;height:1px;background:{T['border']};}}

.kpi-card{{
    background:{T['bg_card']};border:1px solid {T['border']};
    border-radius:12px;padding:1.2rem 1.4rem;text-align:center;
    box-shadow:{T['shadow']};position:relative;overflow:hidden;
}}
.kpi-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,{T['gold']}55,transparent);
}}
.kpi-val  {{font-family:'DM Serif Display',serif;font-size:2.2rem;
            color:{T['text']};line-height:1;margin-bottom:0.3rem;}}
.kpi-label{{font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
            color:{T['text_muted']};letter-spacing:0.12em;text-transform:uppercase;}}
.kpi-delta{{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;margin-top:0.2rem;}}
.delta-up  {{color:#4ADE80;}}
.delta-down{{color:#EF4444;}}
.delta-neu {{color:{T['text_muted']};}}

.chart-card{{
    background:{T['bg_card']};border:1px solid {T['border']};
    border-radius:16px;padding:1.5rem;box-shadow:{T['shadow']};
    margin-bottom:1.5rem;
}}
.chart-title{{
    font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
    color:{T['text_sub']};letter-spacing:0.1em;text-transform:uppercase;
    margin-bottom:1rem;padding-bottom:0.6rem;border-bottom:1px solid {T['border']};
}}

.signal-pill{{
    display:inline-block;padding:0.15rem 0.6rem;border-radius:20px;
    font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:0.08em;
}}
.pill-clean      {{background:#4ADE8022;color:#4ADE80;border:1px solid #4ADE8033;}}
.pill-suspicious {{background:#FBBF2422;color:#FBBF24;border:1px solid #FBBF2433;}}
.pill-high       {{background:#F9731622;color:#F97316;border:1px solid #F9731633;}}
.pill-critical   {{background:#EF444422;color:#EF4444;border:1px solid #EF444433;}}
.pill-approved   {{background:#4ADE8022;color:#4ADE80;border:1px solid #4ADE8033;}}
.pill-rejected   {{background:#EF444422;color:#EF4444;border:1px solid #EF444433;}}
.pill-counter    {{background:#FBBF2422;color:#FBBF24;border:1px solid #FBBF2433;}}
.pill-escalated  {{background:#60A5FA22;color:#60A5FA;border:1px solid #60A5FA33;}}

.history-row{{
    display:grid;
    grid-template-columns:140px 130px 100px 80px 80px 80px 90px 100px;
    gap:0.5rem;align-items:center;
    padding:0.6rem 0.8rem;border-radius:6px;margin-bottom:0.3rem;
    font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
    border:1px solid {T['border']};background:{T['bg_input']};
    color:{T['text_sub']};
}}
.history-header{{
    color:{T['text_muted']};font-weight:500;
    background:{T['bg_card']};border:1px solid {T['border']};
    letter-spacing:0.08em;text-transform:uppercase;
}}
.history-name{{color:{T['text']};font-weight:500;}}

[data-testid="stButton"]>button{{
    background:linear-gradient(135deg,#C9A84C,#B8922A) !important;
    color:#0A0C10 !important;font-family:'IBM Plex Mono',monospace !important;
    font-size:0.75rem !important;font-weight:500 !important;letter-spacing:0.1em !important;
    text-transform:uppercase !important;border:none !important;border-radius:8px !important;
    padding:0.5rem 1.5rem !important;transition:all 0.2s ease !important;
}}
[data-testid="stButton"]>button:hover{{transform:translateY(-1px) !important;}}
[data-testid="stSidebar"]{{background:{T['bg_card']} !important;border-right:1px solid {T['border']} !important;}}
[data-testid="stSidebarNav"]{{display:block !important;visibility:visible !important;}}
hr{{border-color:{T['border']} !important;}}
[data-testid="stPageLink"] a{{font-family:'IBM Plex Mono',monospace !important;font-size:0.7rem !important;letter-spacing:0.08em !important;text-transform:uppercase !important;border:1px solid {T['border']} !important;border-radius:6px !important;padding:0.4rem 1rem !important;color:{T['text_muted']} !important;text-decoration:none !important;transition:all 0.2s ease !important;}}
[data-testid="stPageLink"] a:hover{{color:{T['gold']} !important;border-color:{T['gold']} !important !important;}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SYNTHETIC SEED DATA  — 30 realistic records for demo
# ─────────────────────────────────────────────────────────────────────
NAMES = [
    "Priya Sharma","Arjun Mehta","Sunita Reddy","Ravi Krishnamurthy",
    "Ananya Iyer","Deepak Patel","Kavya Nair","Suresh Gupta",
    "Meera Joshi","Vikram Singh","Pooja Pillai","Aakash Verma",
    "Divya Rao","Nikhil Desai","Shalini Kumar","Rohit Banerjee",
    "Anjali Saxena","Kiran Murthy","Aditya Shah","Swathi Menon",
    "Prakash Tiwari","Rekha Sinha","Manish Agarwal","Lakshmi Devi",
    "Rajesh Nair","Usha Krishnan","Sanjay Bose","Fatima Khan",
    "Harish Choudhary","Nandita Roy"
]
PURPOSES = ["Home Renovation","Vehicle Purchase","Education Loan","Medical Emergency",
            "Business Expansion","Debt Consolidation","Home Renovation","Wedding Expenses",
            "Travel","Equipment Purchase"]
EMP_TYPES = ["Salaried","Salaried","Salaried","Self-Employed","Business Owner"]

def _seed_record(i: int, rng: random.Random) -> dict:
    """Generate one realistic synthetic application record."""
    # Bias toward approvals (~60%) with some rejections/escalations
    profile = rng.choices(
        ["clean_approve","borderline","high_risk","fraud"],
        weights=[55, 25, 12, 8]
    )[0]

    income  = rng.randint(40, 200) * 1000
    loan    = rng.randint(2, 20) * 50000
    credit  = rng.randint(580, 820)
    emis    = rng.randint(0, int(income * 0.2))
    emp     = rng.choice(EMP_TYPES)
    yrs     = rng.randint(1, 15)
    purpose = rng.choice(PURPOSES)

    rate    = 0.105
    tenure  = 60
    new_emi = round((loan * (rate/12)) / (1-(1+rate/12)**-tenure)) if loan else 0
    dti     = round(((emis + new_emi) / income) * 100, 1) if income else 0
    lti     = round(loan / (income * 12), 1) if income else 0

    if profile == "clean_approve":
        fraud  = round(rng.uniform(0.02, 0.28), 4)
        signal = "CLEAN"
        pred   = "human"
        p_h    = round(rng.uniform(0.72, 0.95), 4)
        p_b    = round(rng.uniform(0.01, 0.12), 4)
        p_d    = round(rng.uniform(0.01, 0.08), 4)
        p_c    = round(1 - p_h - p_b - p_d, 4)
        decision = "Approved"
        aria_s = rng.randint(78, 98)
        rex_s  = rng.randint(75, 97)
    elif profile == "borderline":
        fraud  = round(rng.uniform(0.31, 0.58), 4)
        signal = "SUSPICIOUS"
        pred   = rng.choice(["human","coached"])
        p_h    = round(rng.uniform(0.42, 0.65), 4)
        p_b    = round(rng.uniform(0.08, 0.22), 4)
        p_d    = round(rng.uniform(0.04, 0.15), 4)
        p_c    = round(1 - p_h - p_b - p_d, 4)
        decision = rng.choice(["Approved","Counter-Offer","Escalated"])
        aria_s = rng.randint(55, 78)
        rex_s  = rng.randint(45, 72)
    elif profile == "high_risk":
        fraud  = round(rng.uniform(0.62, 0.79), 4)
        signal = "HIGH RISK"
        pred   = rng.choice(["bot","coached"])
        p_h    = round(rng.uniform(0.20, 0.38), 4)
        p_b    = round(rng.uniform(0.35, 0.55), 4)
        p_d    = round(rng.uniform(0.05, 0.15), 4)
        p_c    = round(1 - p_h - p_b - p_d, 4)
        decision = rng.choice(["Escalated","Rejected"])
        aria_s = rng.randint(30, 60)
        rex_s  = rng.randint(20, 45)
    else:  # fraud
        fraud  = round(rng.uniform(0.82, 0.97), 4)
        signal = "CRITICAL"
        pred   = rng.choice(["bot","coached"])
        p_h    = round(rng.uniform(0.02, 0.15), 4)
        p_b    = round(rng.uniform(0.60, 0.82), 4)
        p_d    = round(rng.uniform(0.02, 0.10), 4)
        p_c    = round(1 - p_h - p_b - p_d, 4)
        decision = "Rejected"
        aria_s = rng.randint(20, 50)
        rex_s  = rng.randint(5, 25)

    # Scatter timestamps over the past 14 days
    ts = datetime.now() - timedelta(
        days=rng.randint(0, 13),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59)
    )

    return {
        "id":              f"LN-SEED{i:04d}",
        "timestamp":       ts.isoformat(),
        "applicant_name":  NAMES[i % len(NAMES)],
        "monthly_income":  income,
        "loan_amount":     loan,
        "loan_purpose":    purpose,
        "credit_score":    credit,
        "existing_emis":   emis,
        "emp_type":        emp,
        "years_employed":  yrs,
        "new_emi":         new_emi,
        "dti":             max(dti, 0),
        "lti":             lti,
        "decision":        decision,
        "aria_score":      float(aria_s),
        "rex_score":       float(rex_s),
        "fraud_score":     fraud,
        "fraud_signal":    signal,
        "predicted_class": pred,
        "p_human":         max(p_h, 0),
        "p_bot":           max(p_b, 0),
        "p_duress":        max(p_d, 0),
        "p_coached":       max(p_c, 0),
    }


def _ensure_seed_data():
    """Write seed data if the log is empty or doesn't exist."""
    records = load_all()
    if len(records) == 0:
        rng = random.Random(42)
        seeds = [_seed_record(i, rng) for i in range(30)]
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w") as f:
            json.dump(seeds, f, indent=2)

_ensure_seed_data()


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def _signal_pill(signal: str) -> str:
    cls = {
        "CLEAN":      "pill-clean",
        "SUSPICIOUS": "pill-suspicious",
        "HIGH RISK":  "pill-high",
        "CRITICAL":   "pill-critical",
    }.get(signal, "pill-clean")
    return f'<span class="signal-pill {cls}">{signal}</span>'


def _decision_pill(decision: str) -> str:
    cls = {
        "Approved":     "pill-approved",
        "Rejected":     "pill-rejected",
        "Counter-Offer":"pill-counter",
        "Escalated":    "pill-escalated",
    }.get(decision, "pill-approved")
    return f'<span class="signal-pill {cls}">{decision}</span>'


def _safe_avg(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else 0.0


def _bar(value: float, max_val: float, colour: str, height: int = 120) -> str:
    """Render a single vertical bar as inline HTML."""
    pct = min(int((value / max_val) * 100), 100) if max_val else 0
    bar_h = int(height * pct / 100)
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px;">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.58rem;'
        f'color:{T["text_muted"]};">{value:.0f}</div>'
        f'<div style="width:28px;height:{height}px;background:{T["border"]};'
        f'border-radius:4px;display:flex;align-items:flex-end;overflow:hidden;">'
        f'<div style="width:100%;height:{bar_h}px;background:{colour};'
        f'border-radius:4px;transition:height 0.4s ease;"></div></div>'
        f'</div>'
    )


def _mini_line(values: list, colour: str, width: int = 200, height: int = 50) -> str:
    """SVG sparkline from a list of float values."""
    if len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    span = mx - mn if mx != mn else 1
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = int((i / (n - 1)) * width)
        y = height - int(((v - mn) / span) * (height - 8)) - 4
        pts.append(f"{x},{y}")
    polyline = " ".join(pts)
    last_val = values[-1]
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{polyline}" fill="none" stroke="{colour}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{int(((n-1)/(n-1))*width)}" cy="{height - int(((last_val - mn) / span) * (height - 8)) - 4}" '
        f'r="3" fill="{colour}"/>'
        f'</svg>'
    )


# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([4, 1])
with h1:
    st.markdown(f"""
    <div class="ls-header">
        <div>
            <div class="ls-logo">Lend<span>Synthetix</span></div>
            <div class="ls-tagline">📊 Analytics Dashboard &nbsp;|&nbsp; Underwriting Intelligence</div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                    color:{T['text_muted']};letter-spacing:0.1em;">
            Last updated: {datetime.now().strftime("%d %b %Y, %H:%M")}
        </div>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    st.button("☀️ Light" if dark else "🌙 Dark", on_click=toggle_theme)



# ─────────────────────────────────────────────────────────────────────
# PAGE NAVIGATION
# ─────────────────────────────────────────────────────────────────────
nav_cols = st.columns([1, 1, 6])
with nav_cols[0]:
    st.page_link("app.py",                        label="⚖️  Underwriting", use_container_width=True)
with nav_cols[1]:
    st.page_link("pages/1_Analytics.py",       label="📊  Analytics",    use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────
records = load_all()
# Sort newest first
records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

n = len(records)

# Derived lists
decisions    = [r["decision"]     for r in records]
fraud_scores = [r["fraud_score"]  for r in records]
dtis         = [r["dti"]          for r in records]
credits      = [r["credit_score"] for r in records]
aria_scores  = [r["aria_score"]   for r in records if r.get("aria_score", 0) > 0]
rex_scores   = [r["rex_score"]    for r in records if r.get("rex_score",  0) > 0]
signals      = [r["fraud_signal"] for r in records]
classes      = [r.get("predicted_class","human") for r in records]
incomes      = [r["monthly_income"] for r in records]
loan_amts    = [r["loan_amount"]    for r in records]

approved_n  = decisions.count("Approved")
rejected_n  = decisions.count("Rejected")
counter_n   = decisions.count("Counter-Offer")
escalated_n = decisions.count("Escalated")

approval_rate = round(approved_n / n * 100, 1) if n else 0
avg_fraud     = round(_safe_avg(fraud_scores), 3)
avg_dti       = round(_safe_avg(dtis), 1)
avg_credit    = round(_safe_avg(credits))
avg_loan      = round(_safe_avg(loan_amts))

# Trend vs previous half (simple split)
mid = n // 2
prev_approval = (decisions[:mid].count("Approved") / mid * 100) if mid else 0
curr_approval = (decisions[mid:].count("Approved") / max(n - mid, 1) * 100)


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                letter-spacing:0.2em;color:{T['text_muted']};text-transform:uppercase;
                margin-bottom:1rem;">📊 Filters</div>
    """, unsafe_allow_html=True)

    filter_decision = st.multiselect(
        "Decision", ["Approved","Rejected","Counter-Offer","Escalated"],
        default=["Approved","Rejected","Counter-Offer","Escalated"]
    )
    filter_signal = st.multiselect(
        "Fraud Signal", ["CLEAN","SUSPICIOUS","HIGH RISK","CRITICAL"],
        default=["CLEAN","SUSPICIOUS","HIGH RISK","CRITICAL"]
    )
    filter_emp = st.multiselect(
        "Employment Type", ["Salaried","Self-Employed","Business Owner"],
        default=["Salaried","Self-Employed","Business Owner"]
    )

    st.markdown("---")
    if st.button("🔄  Refresh Data"):
        st.rerun()

    st.markdown("---")
    if st.button("🗑  Reset to Seed Data"):
        clear_log()
        _ensure_seed_data()
        st.success("Log cleared. Seed data restored.")
        st.rerun()

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                color:{T['text_muted']};margin-top:1.5rem;line-height:2;opacity:0.5;">
    Total records    : {n}<br>
    Log path         : data/applications.json
    </div>
    """, unsafe_allow_html=True)

# Apply sidebar filters
records_f = [
    r for r in records
    if r["decision"]     in filter_decision
    and r["fraud_signal"] in filter_signal
    and r["emp_type"]     in filter_emp
]
nf = len(records_f)


# ─────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)

def kpi(col, val, label, delta=None, delta_label=""):
    if delta is not None:
        d_cls = "delta-up" if delta >= 0 else "delta-down"
        d_sym = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta {d_cls}">{d_sym} {abs(delta):.1f}% {delta_label}</div>'
    else:
        delta_html = f'<div class="kpi-delta delta-neu">— no trend data</div>'
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{val}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

with k1: kpi(k1, n,                  "Total Applications")
with k2: kpi(k2, f"{approval_rate}%","Approval Rate",
             delta=round(curr_approval - prev_approval, 1), delta_label="vs prev")
with k3: kpi(k3, f"{avg_fraud:.3f}", "Avg Fraud Score")
with k4: kpi(k4, f"{avg_dti}%",      "Avg Post-Loan DTI")
with k5: kpi(k5, str(avg_credit),    "Avg Credit Score")
with k6: kpi(k6, f"Rs {avg_loan:,}", "Avg Loan Amount")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# ROW 1  — Decision breakdown | Fraud signal breakdown
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Decision & Fraud Signal Distribution</div>',
            unsafe_allow_html=True)

r1a, r1b = st.columns(2, gap="large")

# ── Decision Donut (pure HTML/CSS) ──
with r1a:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Decision Breakdown</div>', unsafe_allow_html=True)

    dec_data = [
        ("Approved",     approved_n,  "#4ADE80"),
        ("Rejected",     rejected_n,  "#EF4444"),
        ("Counter-Offer",counter_n,   "#FBBF24"),
        ("Escalated",    escalated_n, "#60A5FA"),
    ]

    # Horizontal bar chart using native Streamlit
    for label, count, colour in dec_data:
        if n == 0:
            pct = 0
        else:
            pct = int(count / n * 100)
        lc, bc, pc = st.columns([2, 5, 1])
        with lc:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:{colour};padding-top:2px;'>{label}</div>",
                unsafe_allow_html=True
            )
        with bc:
            st.markdown(
                f"<div style='height:20px;background:{T['border']};border-radius:4px;"
                f"overflow:hidden;margin-top:2px;'>"
                f"<div style='height:100%;width:{pct}%;background:{colour};"
                f"border-radius:4px;'></div></div>",
                unsafe_allow_html=True
            )
        with pc:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:{T['text_sub']};text-align:right;padding-top:2px;'>"
                f"{count} ({pct}%)</div>",
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Fraud Signal Breakdown ──
with r1b:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">BioSentinel Signal Distribution</div>',
                unsafe_allow_html=True)

    sig_data = [
        ("CLEAN",      signals.count("CLEAN"),      "#4ADE80"),
        ("SUSPICIOUS", signals.count("SUSPICIOUS"),  "#FBBF24"),
        ("HIGH RISK",  signals.count("HIGH RISK"),   "#F97316"),
        ("CRITICAL",   signals.count("CRITICAL"),    "#EF4444"),
    ]

    for label, count, colour in sig_data:
        pct = int(count / n * 100) if n else 0
        lc, bc, pc = st.columns([2, 5, 1])
        with lc:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:{colour};padding-top:2px;'>{label}</div>",
                unsafe_allow_html=True
            )
        with bc:
            st.markdown(
                f"<div style='height:20px;background:{T['border']};border-radius:4px;"
                f"overflow:hidden;margin-top:2px;'>"
                f"<div style='height:100%;width:{pct}%;background:{colour};"
                f"border-radius:4px;'></div></div>",
                unsafe_allow_html=True
            )
        with pc:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:{T['text_sub']};text-align:right;padding-top:2px;'>"
                f"{count} ({pct}%)</div>",
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# ROW 2  — DTI histogram | Credit score histogram | Fraud score histogram
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Financial Profile Distributions</div>',
            unsafe_allow_html=True)

r2a, r2b, r2c = st.columns(3, gap="large")

def _histogram(col, values: list, bins: list, label: str,
               colour: str, x_labels: list = None):
    """
    Render a vertical bar histogram inside a chart-card.
    bins   : list of (lo, hi) tuples
    """
    counts = [sum(1 for v in values if lo <= v < hi) for lo, hi in bins]
    max_c  = max(counts) if counts else 1

    with col:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-title">{label}</div>', unsafe_allow_html=True)

        bars_html = '<div style="display:flex;align-items:flex-end;gap:6px;height:130px;padding-bottom:4px;">'
        for i, c in enumerate(counts):
            bar_h = int((c / max_c) * 110) if max_c else 0
            xl = x_labels[i] if x_labels else f"{bins[i][0]}"
            bars_html += (
                f'<div style="flex:1;display:flex;flex-direction:column;'
                f'align-items:center;gap:2px;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.52rem;'
                f'color:{T["text_muted"]};">{c}</div>'
                f'<div style="width:100%;height:{bar_h}px;background:{colour};'
                f'border-radius:3px 3px 0 0;min-height:2px;"></div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.48rem;'
                f'color:{T["text_muted"]};text-align:center;width:100%;'
                f'word-break:break-all;">{xl}</div>'
                f'</div>'
            )
        bars_html += '</div>'
        st.markdown(bars_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

_histogram(
    r2a, dtis,
    bins=[(0,15),(15,25),(25,35),(35,45),(45,55),(55,100)],
    x_labels=["<15","15-25","25-35","35-45","45-55","55+"],
    label="Post-Loan DTI Distribution (%)",
    colour="#60A5FA"
)

_histogram(
    r2b, credits,
    bins=[(300,600),(600,650),(650,700),(700,750),(750,800),(800,900)],
    x_labels=["<600","600-650","650-700","700-750","750-800","800+"],
    label="Credit Score Distribution",
    colour=T["gold"]
)

_histogram(
    r2c, fraud_scores,
    bins=[(0,0.15),(0.15,0.30),(0.30,0.50),(0.50,0.65),(0.65,0.80),(0.80,1.01)],
    x_labels=["0-0.15","0.15-0.3","0.3-0.5","0.5-0.65","0.65-0.8","0.8+"],
    label="Fraud Score Distribution",
    colour="#EF4444"
)


# ─────────────────────────────────────────────────────────────────────
# ROW 3  — BioSentinel class breakdown | Agent score comparison
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">BioSentinel & Agent Intelligence</div>',
            unsafe_allow_html=True)

r3a, r3b = st.columns(2, gap="large")

# ── BioSentinel Class Breakdown ──
with r3a:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Predicted Behavioral Class Distribution</div>',
                unsafe_allow_html=True)

    class_data = [
        ("human",   classes.count("human"),   "#4ADE80"),
        ("bot",     classes.count("bot"),     "#EF4444"),
        ("duress",  classes.count("duress"),  "#FBBF24"),
        ("coached", classes.count("coached"), "#F97316"),
    ]

    avg_probs = {
        "P(human)":   round(_safe_avg([r.get("p_human",0)   for r in records]), 3),
        "P(bot)":     round(_safe_avg([r.get("p_bot",0)     for r in records]), 3),
        "P(duress)":  round(_safe_avg([r.get("p_duress",0)  for r in records]), 3),
        "P(coached)": round(_safe_avg([r.get("p_coached",0) for r in records]), 3),
    }

    # Class count bars
    for cls, count, colour in class_data:
        pct = int(count / n * 100) if n else 0
        lc, bc, pc = st.columns([2, 5, 1])
        with lc:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:{colour};padding-top:2px;'>{cls}</div>",
                unsafe_allow_html=True
            )
        with bc:
            st.markdown(
                f"<div style='height:20px;background:{T['border']};border-radius:4px;"
                f"overflow:hidden;margin-top:2px;'>"
                f"<div style='height:100%;width:{pct}%;background:{colour};"
                f"border-radius:4px;'></div></div>",
                unsafe_allow_html=True
            )
        with pc:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:{T['text_sub']};text-align:right;padding-top:2px;'>"
                f"{count}</div>",
                unsafe_allow_html=True
            )

    st.markdown(f"<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;"
                f"color:{T['text_muted']};letter-spacing:0.1em;text-transform:uppercase;"
                f"margin-bottom:0.4rem;'>Portfolio Avg Class Probabilities</div>",
                unsafe_allow_html=True)

    prob_colours = {"P(human)":"#4ADE80","P(bot)":"#EF4444",
                    "P(duress)":"#FBBF24","P(coached)":"#F97316"}
    for label, avg in avg_probs.items():
        pct = int(avg * 100)
        lc, bc, pc = st.columns([2, 5, 1])
        colour = prob_colours[label]
        with lc:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;"
                f"color:{T['text_muted']};padding-top:2px;'>{label}</div>",
                unsafe_allow_html=True
            )
        with bc:
            st.markdown(
                f"<div style='height:14px;background:{T['border']};border-radius:3px;"
                f"overflow:hidden;margin-top:3px;'>"
                f"<div style='height:100%;width:{pct}%;background:{colour};"
                f"border-radius:3px;opacity:0.7;'></div></div>",
                unsafe_allow_html=True
            )
        with pc:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;"
                f"color:{T['text_sub']};text-align:right;padding-top:3px;'>"
                f"{avg:.3f}</div>",
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Agent Score Comparison ──
with r3b:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Agent Score Averages by Decision</div>',
                unsafe_allow_html=True)

    for decision, d_colour in [("Approved","#4ADE80"),("Rejected","#EF4444"),
                                ("Counter-Offer","#FBBF24"),("Escalated","#60A5FA")]:
        subset = [r for r in records if r["decision"] == decision]
        if not subset:
            continue
        a_avg = round(_safe_avg([r["aria_score"] for r in subset if r.get("aria_score",0)>0]))
        r_avg = round(_safe_avg([r["rex_score"]  for r in subset if r.get("rex_score",0)>0]))
        f_avg = round(_safe_avg([r["fraud_score"] for r in subset]) * 100)

        st.markdown(
            f"<div style='margin-bottom:0.8rem;padding:0.6rem 0.8rem;"
            f"background:{T['bg_input']};border-radius:8px;"
            f"border:1px solid {T['border']};'>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
            f"color:{d_colour};letter-spacing:0.08em;margin-bottom:0.4rem;'>"
            f"{decision} ({len(subset)})</div>"
            f"<div style='display:flex;gap:1.5rem;font-family:IBM Plex Mono,monospace;"
            f"font-size:0.6rem;color:{T['text_muted']};'>"
            f"<span>ARIA <span style='color:{T['text_sub']};'>{a_avg}/100</span></span>"
            f"<span>REX <span style='color:{T['text_sub']};'>{r_avg}/100</span></span>"
            f"<span>Fraud <span style='color:{T['text_sub']};'>{f_avg/100:.2f}</span></span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    # Avg scores overall
    st.markdown(
        f"<div style='margin-top:0.5rem;padding:0.6rem 0.8rem;"
        f"background:{T['bg_card']};border-radius:8px;"
        f"border:1px solid {T['border']};border-left:2px solid {T['gold']};'>"
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;"
        f"color:{T['gold']};letter-spacing:0.1em;margin-bottom:0.3rem;'>PORTFOLIO AVERAGES</div>"
        f"<div style='display:flex;gap:1.5rem;font-family:IBM Plex Mono,monospace;"
        f"font-size:0.65rem;color:{T['text_sub']};'>"
        f"<span>ARIA {round(_safe_avg(aria_scores))}/100</span>"
        f"<span>REX {round(_safe_avg(rex_scores))}/100</span>"
        f"<span>Fraud {avg_fraud:.3f}</span>"
        f"</div></div>",
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# ROW 4  — Fraud score vs Decision scatter (text-based) | Employment mix
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Risk Patterns</div>', unsafe_allow_html=True)

r4a, r4b = st.columns(2, gap="large")

# ── Fraud Score vs Decision Matrix ──
with r4a:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Fraud Score × Decision Matrix</div>',
                unsafe_allow_html=True)

    bands   = [("0.00–0.30","CLEAN"),("0.31–0.60","SUSPICIOUS"),
               ("0.61–0.80","HIGH RISK"),("0.81–1.00","CRITICAL")]
    d_types = ["Approved","Counter-Offer","Escalated","Rejected"]
    d_cols  = {"Approved":"#4ADE80","Counter-Offer":"#FBBF24",
               "Escalated":"#60A5FA","Rejected":"#EF4444"}

    # Header row
    header = (
        f"<div style='display:grid;grid-template-columns:120px repeat(4,1fr);"
        f"gap:4px;margin-bottom:4px;font-family:IBM Plex Mono,monospace;"
        f"font-size:0.55rem;color:{T['text_muted']};letter-spacing:0.06em;'>"
        f"<div></div>"
    )
    for d in d_types:
        header += f"<div style='text-align:center;'>{d[:8]}</div>"
    header += "</div>"
    st.markdown(header, unsafe_allow_html=True)

    for (band_label, _), (lo, hi) in zip(
        bands,
        [(0, 0.30),(0.30, 0.60),(0.60, 0.80),(0.80, 1.01)]
    ):
        row_recs = [r for r in records if lo <= r["fraud_score"] < hi]
        row_n    = len(row_recs)
        row_html = (
            f"<div style='display:grid;grid-template-columns:120px repeat(4,1fr);"
            f"gap:4px;margin-bottom:4px;align-items:center;'>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.58rem;"
            f"color:{T['text_sub']};'>{band_label}</div>"
        )
        for d in d_types:
            cnt  = sum(1 for r in row_recs if r["decision"] == d)
            pct  = int(cnt / row_n * 100) if row_n else 0
            intensity = min(int(pct * 2.5), 255)
            col_hex = d_cols[d]
            row_html += (
                f"<div style='text-align:center;padding:4px 2px;"
                f"background:{col_hex}{format(max(intensity,15),'02x')};"
                f"border-radius:4px;font-family:IBM Plex Mono,monospace;"
                f"font-size:0.6rem;color:{T['text']};'>"
                f"{cnt}<span style='font-size:0.48rem;color:{T['text_muted']};'> ({pct}%)</span>"
                f"</div>"
            )
        row_html += "</div>"
        st.markdown(row_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Employment Mix ──
with r4b:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Employment Type vs Approval Rate</div>',
                unsafe_allow_html=True)

    emp_types_all = ["Salaried","Self-Employed","Business Owner"]
    emp_colours   = ["#60A5FA","#C9A84C","#A78BFA"]

    for emp, colour in zip(emp_types_all, emp_colours):
        subset = [r for r in records if r["emp_type"] == emp]
        if not subset:
            continue
        total    = len(subset)
        approved = sum(1 for r in subset if r["decision"] == "Approved")
        apr_rate = int(approved / total * 100) if total else 0
        avg_f    = round(_safe_avg([r["fraud_score"] for r in subset]), 3)
        avg_c    = round(_safe_avg([r["credit_score"] for r in subset]))

        st.markdown(
            f"<div style='margin-bottom:0.8rem;padding:0.8rem;"
            f"background:{T['bg_input']};border-radius:8px;"
            f"border:1px solid {T['border']};'>"

            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;margin-bottom:0.5rem;'>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
            f"color:{colour};'>{emp}</span>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;"
            f"color:{T['text_muted']};'>{total} apps</span>"
            f"</div>"

            f"<div style='height:12px;background:{T['border']};border-radius:3px;"
            f"overflow:hidden;margin-bottom:0.4rem;'>"
            f"<div style='height:100%;width:{apr_rate}%;background:{colour};"
            f"border-radius:3px;'></div></div>"

            f"<div style='display:flex;gap:1.2rem;font-family:IBM Plex Mono,monospace;"
            f"font-size:0.58rem;color:{T['text_muted']};'>"
            f"<span>Approval {apr_rate}%</span>"
            f"<span>Avg Credit {avg_c}</span>"
            f"<span>Avg Fraud {avg_f}</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# APPLICATION HISTORY TABLE
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Application History</div>', unsafe_allow_html=True)

# Column headers
st.markdown(f"""
<div class="history-row history-header">
    <div>Applicant</div>
    <div>Timestamp</div>
    <div>Loan Amt</div>
    <div>Credit</div>
    <div>DTI%</div>
    <div>Fraud</div>
    <div>Signal</div>
    <div>Decision</div>
</div>
""", unsafe_allow_html=True)

display_records = records_f[:50]   # cap at 50 rows
for r in display_records:
    ts_str  = r["timestamp"][:16].replace("T", " ")
    signal  = r.get("fraud_signal", "CLEAN")
    decision= r.get("decision", "")

    sig_css = {
        "CLEAN":      "pill-clean",
        "SUSPICIOUS": "pill-suspicious",
        "HIGH RISK":  "pill-high",
        "CRITICAL":   "pill-critical",
    }.get(signal, "pill-clean")

    dec_css = {
        "Approved":     "pill-approved",
        "Rejected":     "pill-rejected",
        "Counter-Offer":"pill-counter",
        "Escalated":    "pill-escalated",
    }.get(decision, "pill-approved")

    st.markdown(f"""
<div class="history-row">
    <div class="history-name">{r['applicant_name']}</div>
    <div>{ts_str}</div>
    <div>Rs {r['loan_amount']:,}</div>
    <div>{r['credit_score']}</div>
    <div>{r['dti']}%</div>
    <div>{r['fraud_score']:.3f}</div>
    <div><span class="signal-pill {sig_css}">{signal}</span></div>
    <div><span class="signal-pill {dec_css}">{decision}</span></div>
</div>
""", unsafe_allow_html=True)

if nf == 0:
    st.markdown(
        f"<div style='text-align:center;padding:2rem;font-family:IBM Plex Mono,monospace;"
        f"font-size:0.7rem;color:{T['text_muted']};'>No records match the current filters.</div>",
        unsafe_allow_html=True
    )
elif nf > 50:
    st.markdown(
        f"<div style='text-align:center;padding:0.5rem;font-family:IBM Plex Mono,monospace;"
        f"font-size:0.6rem;color:{T['text_muted']};'>Showing 50 of {nf} filtered records.</div>",
        unsafe_allow_html=True
    )

# ── CSV export ──
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
if records_f:
    import csv, io
    buf = io.StringIO()
    cols = ["id","timestamp","applicant_name","monthly_income","loan_amount",
            "loan_purpose","credit_score","existing_emis","emp_type","years_employed",
            "new_emi","dti","lti","decision","aria_score","rex_score",
            "fraud_score","fraud_signal","predicted_class",
            "p_human","p_bot","p_duress","p_coached"]
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records_f)
    st.download_button(
        "⬇️  Export Filtered Records (.csv)",
        data=buf.getvalue(),
        file_name=f"lendsynthetix_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )