import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

# Optional dependency fallback
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Digital Twin Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use a NEW file name so old corrupted CSV does not break the app
HISTORY_PATH = os.path.join(BASE_DIR, "history_clean.csv")

# =========================================================
# THEME / CSS
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0, 212, 255, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(124, 58, 237, 0.10), transparent 22%),
            linear-gradient(180deg, #070b16 0%, #0b1020 100%);
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(168,85,247,0.16));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 35px rgba(0,0,0,0.28);
        border-radius: 24px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.12;
        color: #f8fafc;
    }

    .hero-subtitle {
        margin-top: 0.45rem;
        color: #cbd5e1;
        font-size: 0.98rem;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.74);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        backdrop-filter: blur(10px);
    }

    .metric-wrap {
        background: linear-gradient(180deg, rgba(30,41,59,0.98), rgba(15,23,42,0.96));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.85rem 1rem;
        text-align: left;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
        min-height: 120px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.84rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-note {
        margin-top: 0.35rem;
        color: #cbd5e1;
        font-size: 0.86rem;
    }

    .status-ok {
        background: linear-gradient(135deg, rgba(34,197,94,0.22), rgba(34,197,94,0.10));
        border: 1px solid rgba(34,197,94,0.45);
        color: #d1fae5;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 8px 22px rgba(0,0,0,0.15);
    }

    .status-bad {
        background: linear-gradient(135deg, rgba(239,68,68,0.22), rgba(239,68,68,0.10));
        border: 1px solid rgba(239,68,68,0.50);
        color: #fee2e2;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 8px 22px rgba(0,0,0,0.15);
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.86rem;
    }

    .section-title {
        margin: 0.1rem 0 0.6rem 0;
        font-size: 1.1rem;
        font-weight: 800;
        color: #e2e8f0;
    }

    .footer-note {
        margin-top: 1rem;
        color: #94a3b8;
        font-size: 0.83rem;
        text-align: center;
    }

    .stSidebar {
        background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(2,6,23,0.98));
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }

    .subtle-chip {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.08);
        color: #cbd5e1;
        font-size: 0.78rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        background: rgba(15, 23, 42, 0.55);
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user" not in st.session_state:
    st.session_state["user"] = ""

# =========================================================
# AUTH
# =========================================================
def login_screen():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">   🏭 Digital Twin Monitoring Platform</div>
            <div class="hero-subtitle">
                Secure access for predictive maintenance simulation and AI-based health monitoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.25, 1])

    with center:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Login")

        username = st.text_input("Username", placeholder="admin", key="login_user")
        password = st.text_input("Password", type="password", placeholder="1234", key="login_pass")

        c1, c2 = st.columns(2)
        with c1:
            login_clicked = st.button("Login", use_container_width=True)
        with c2:
            st.caption("Default: admin / 1234")

        if login_clicked:
            user_ok = os.getenv("APP_USER", "admin")
            pass_ok = os.getenv("APP_PASS", "1234")

            if username == user_ok and password == pass_ok:
                st.session_state["logged_in"] = True
                st.session_state["user"] = username
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown(
            """
            <div class="small-muted">
                This dashboard simulates a mechanical digital twin, logs history, and highlights anomalies in real time.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state["logged_in"]:
    login_screen()
    st.stop()

# =========================================================
# SIMULATION
# =========================================================
def simulate_digital_twin(n_samples=500, fault_strength=2.0):
    seed = int(time.time()) % 1_000_000
    rng = np.random.default_rng(seed)

    t = np.arange(n_samples)
    vibration = np.sin(0.02 * t) + 0.5 * rng.normal(size=n_samples)
    temperature = 50 + 0.01 * t + rng.normal(size=n_samples)
    load = 10 + rng.normal(size=n_samples)

    fault_start = int(n_samples * 0.6)
    vibration[fault_start:] += fault_strength
    temperature[fault_start:] += fault_strength * 5

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    return pd.DataFrame(
        {
            "time": t,
            "timestamp": timestamp,
            "vibration": vibration,
            "temperature": temperature,
            "load": load,
        }
    )


def process_data(df):
    features = df[["vibration", "temperature", "load"]].copy()

    pca = PCA(n_components=1)
    out = df.copy()
    out["health_score"] = pca.fit_transform(features)

    mean_score = out["health_score"].mean()
    std_score = out["health_score"].std(ddof=0) or 1e-9

    out["z_score"] = (out["health_score"] - mean_score) / std_score
    out["anomaly"] = out["z_score"].abs() > 2.0
    out["health_index"] = np.clip(100 - out["z_score"].abs() * 20, 0, 100)

    latest = out.iloc[-1]
    health_index = float(out["health_index"].iloc[-1])

    return out, health_index


def save_latest_row(df, path):
    latest_row = df.tail(1).copy()

    # fixed schema from now on
    columns = [
        "time",
        "timestamp",
        "vibration",
        "temperature",
        "load",
        "health_score",
        "z_score",
        "anomaly",
        "health_index",
    ]
    latest_row = latest_row[columns]

    if not os.path.exists(path):
        latest_row.to_csv(path, index=False)
    else:
        latest_row.to_csv(path, mode="a", header=False, index=False)


def load_history(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        hist = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

    expected = [
        "time",
        "timestamp",
        "vibration",
        "temperature",
        "load",
        "health_score",
        "z_score",
        "anomaly",
        "health_index",
    ]

    hist = hist[[c for c in hist.columns if c in expected]].copy()

    if "timestamp" in hist.columns:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")

    return hist


def card_html(label, value, note):
    return f"""
    <div class="metric-wrap">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def build_gauge(value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number={"suffix": " / 100"},
            delta={"reference": 80, "increasing": {"color": "#22c55e"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.18)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.28)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.22)"},
                    {"range": [70, 100], "color": "rgba(34,197,94,0.20)"},
                ],
                "threshold": {
                    "line": {"color": "#f97316", "width": 4},
                    "thickness": 0.75,
                    "value": 60,
                },
            },
            title={"text": "Health Index", "font": {"size": 18, "color": "#e2e8f0"}},
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Controls</div>', unsafe_allow_html=True)
    st.caption("Adjust the simulation and refresh speed.")

    if st.button("Logout", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["user"] = ""
        st.rerun()

    n_samples = st.slider("Data points", 100, 2000, 500, step=50)
    fault_strength = st.slider("Fault strength", 0.5, 4.0, 2.0, step=0.1)
    refresh_ms = st.slider("Refresh interval (ms)", 1000, 5000, 2000, step=500)

    st.markdown("---")
    st.markdown("### System Info")
    st.write(f"User: `{st.session_state.get('user', 'admin')}`")
    st.write(f"Refresh: `{refresh_ms} ms`")
    st.write("Mode: `Simulation + AI`")
    st.markdown(
        """
        <div class="small-muted">
            Tip: keep the app open for a few refresh cycles to build a useful history log.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Auto-refresh
st_autorefresh(interval=refresh_ms, key="refresh")

# =========================================================
# DASHBOARD DATA
# =========================================================
df = simulate_digital_twin(n_samples=n_samples, fault_strength=fault_strength)
df, health_index = process_data(df)
latest = df.iloc[-1]

save_latest_row(df, HISTORY_PATH)
history_df = load_history(HISTORY_PATH)

anomaly_now = bool(latest["anomaly"])
risk_label = "Critical" if health_index < 40 else ("Warning" if health_index < 70 else "Healthy")

prev = df.iloc[-2] if len(df) > 1 else latest
delta_temp = latest["temperature"] - prev["temperature"]
delta_vib = latest["vibration"] - prev["vibration"]
delta_load = latest["load"] - prev["load"]

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🏭 AI-Based Digital Twin System for Predictive Maintenance</div>
        <div class="hero-subtitle">
            Real-time simulation • anomaly detection • sensor history • interactive monitoring
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chips = st.columns(4)
with chips[0]:
    st.markdown('<span class="subtle-chip">Simulation</span>', unsafe_allow_html=True)
with chips[1]:
    st.markdown('<span class="subtle-chip">PCA-based scoring</span>', unsafe_allow_html=True)
with chips[2]:
    st.markdown('<span class="subtle-chip">History logging</span>', unsafe_allow_html=True)
with chips[3]:
    st.markdown('<span class="subtle-chip">Live refresh</span>', unsafe_allow_html=True)

# =========================================================
# TOP KPI CARDS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        card_html(
            "Health Index",
            f"{health_index:.1f}",
            f"Condition: {risk_label}",
        ),
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        card_html(
            "Temperature (°C)",
            f"{latest['temperature']:.2f}",
            f"Δ {delta_temp:+.2f} since last sample",
        ),
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        card_html(
            "Vibration",
            f"{latest['vibration']:.2f}",
            f"Δ {delta_vib:+.2f} since last sample",
        ),
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        card_html(
            "Load",
            f"{latest['load']:.2f}",
            f"Δ {delta_load:+.2f} since last sample",
        ),
        unsafe_allow_html=True,
    )

st.markdown("")

# =========================================================
# STATUS BANNER
# =========================================================
if anomaly_now:
    st.markdown(
        """
        <div class="status-bad">
            🔴 CRITICAL: Anomaly Detected — mechanical condition is outside the normal envelope.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="status-ok">
            🟢 SYSTEM OPERATING NORMALLY — no strong anomaly detected in the current window.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Signal Analysis", "🗂️ History & Data"])

# ------------------ OVERVIEW ------------------
with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Health Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(build_gauge(health_index), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Latest Sample Snapshot</div>', unsafe_allow_html=True)

        snap1, snap2 = st.columns(2)
        snap1.metric("Time Index", int(latest["time"]))
        snap2.metric("Health Z-Score", f"{float(latest['z_score']):.2f}")

        st.markdown(
            f"""
            <div class="small-muted">
                Timestamp: <b>{latest['timestamp']}</b><br>
                Current state: <b>{risk_label}</b><br>
                The health index is derived from PCA-based feature compression plus anomaly scoring.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Trend Summary")
        summary_fig = px.line(
            df,
            x="time",
            y="health_index",
            title="Health Index Trend",
            template="plotly_dark",
        )
        summary_fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(summary_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ SIGNAL ANALYSIS ------------------
with tab2:
    s1, s2 = st.columns(2)

    fig_vib = px.line(
        df,
        x="time",
        y="vibration",
        title="Vibration Signal",
        template="plotly_dark",
    )
    fig_vib.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    fig_vib.update_traces(line=dict(width=2.8))

    fig_temp = px.line(
        df,
        x="time",
        y="temperature",
        title="Temperature Signal",
        template="plotly_dark",
    )
    fig_temp.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    fig_temp.update_traces(line=dict(width=2.8))

    with s1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_vib, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with s2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_temp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    col_a, col_b = st.columns([1.15, 0.85])

    with col_a:
        fig_load = px.line(
            df,
            x="time",
            y="load",
            title="Load Profile",
            template="plotly_dark",
        )
        fig_load.update_layout(height=350, margin=dict(l=10, r=10, t=45, b=10))
        fig_load.update_traces(line=dict(width=2.8))

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_load, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        corr = df[["vibration", "temperature", "load", "health_score"]].corr(numeric_only=True)
        heat = px.imshow(
            corr,
            text_auto=True,
            title="Feature Correlation",
            template="plotly_dark",
            aspect="auto",
        )
        heat.update_layout(height=350, margin=dict(l=10, r=10, t=45, b=10))

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(heat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    fig_health = px.line(
        df,
        x="time",
        y="health_score",
        title="PCA-Based Health Score",
        template="plotly_dark",
    )
    fig_health.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
    fig_health.update_traces(line=dict(width=3))
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_health, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ HISTORY & DATA ------------------
with tab3:
    h1, h2 = st.columns([1, 1])

    with h1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Historical Sensor Trends</div>', unsafe_allow_html=True)

        if not history_df.empty:
            x_axis = "timestamp" if "timestamp" in history_df.columns else history_df.index
            hist_plot = px.line(
                history_df,
                x=x_axis,
                y=["vibration", "temperature", "load"] if all(
                    c in history_df.columns for c in ["vibration", "temperature", "load"]
                ) else history_df.columns,
                title="Logged Sensor History",
                template="plotly_dark",
            )
            hist_plot.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(hist_plot, use_container_width=True)
        else:
            st.info("No history found yet. Keep the app running for a few refresh cycles.")

        st.markdown("</div>", unsafe_allow_html=True)

    with h2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Data Export</div>', unsafe_allow_html=True)
        st.write("Download the current dashboard data for analysis or reporting.")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download current data as CSV",
            data=csv_data,
            file_name="digital_twin_current_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.write("")
        st.caption("Latest recorded rows")

        show_cols = ["time", "vibration", "temperature", "load", "health_score", "z_score", "anomaly", "health_index"]
        st.dataframe(df[show_cols].tail(20), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 📋 Stored History Preview")
    if not history_df.empty:
        st.dataframe(history_df.tail(25), use_container_width=True)
    else:
        st.info("history_clean.csv is empty right now.")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer-note">
        AI-based digital twin simulation with PCA-driven anomaly scoring and time-series logging.
    </div>
    """,
    unsafe_allow_html=True,
)