import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FloodSense — Disaster Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Black & Wine/Red Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background-color: #080808;
        color: #f0e6e6;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #110000 0%, #080808 100%);
        border-right: 1px solid #2a0a0a;
    }

    .stRadio > div { gap: 4px; }
    .stRadio label {
        color: #9a7070 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    .stRadio label:hover {
        background: #1a0505 !important;
        color: #e8a0a0 !important;
    }

    .main-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4rem;
        letter-spacing: 4px;
        background: linear-gradient(135deg, #c0392b, #922b21, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1;
    }

    .sub-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        color: #6b3030;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 6px;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f0505 0%, #1a0808 100%);
        border: 1px solid #2a0f0f;
        border-top: 2px solid #7b1c1c;
        border-radius: 12px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-top-color: #c0392b;
        box-shadow: 0 8px 32px rgba(192, 57, 43, 0.15);
    }

    .metric-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 2px;
        color: #c0392b;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.7rem;
        color: #6b3030;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 6px;
    }

    .flood-alert {
        background: linear-gradient(135deg, #1a0000, #2d0505);
        border: 1px solid #c0392b;
        border-left: 5px solid #c0392b;
        border-radius: 10px;
        padding: 22px 28px;
        color: #ff8a80;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.6rem;
        letter-spacing: 3px;
    }

    .safe-alert {
        background: linear-gradient(135deg, #000d0a, #001a12);
        border: 1px solid #1e7e5a;
        border-left: 5px solid #27ae60;
        border-radius: 10px;
        padding: 22px 28px;
        color: #69f0ae;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.6rem;
        letter-spacing: 3px;
    }

    .section-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.4rem;
        letter-spacing: 4px;
        color: #c0392b;
        border-bottom: 1px solid #2a0f0f;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    .info-box {
        background: linear-gradient(135deg, #0f0505, #1a0808);
        border: 1px solid #2a0f0f;
        border-radius: 12px;
        padding: 24px;
        color: #9a7070;
        line-height: 1.9;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7b1c1c, #c0392b);
        color: #fff0f0;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.1rem;
        letter-spacing: 3px;
        border: none;
        border-radius: 8px;
        padding: 14px 32px;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #922b21, #e74c3c);
        box-shadow: 0 4px 20px rgba(192, 57, 43, 0.4);
    }

    .stSlider > div > div > div {
        background: #c0392b !important;
    }

    div[data-testid="stMetric"] {
        background: #0f0505;
        border: 1px solid #2a0f0f;
        border-radius: 10px;
        padding: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #6b3030;
        letter-spacing: 1px;
    }

    .stTabs [aria-selected="true"] {
        color: #c0392b !important;
        border-bottom-color: #c0392b !important;
    }

    .stDataFrame { border: 1px solid #2a0f0f !important; border-radius: 10px !important; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #080808; }
    ::-webkit-scrollbar-thumb { background: #3d0f0f; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #7b1c1c; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MATPLOTLIB THEME CONSTANTS
# ─────────────────────────────────────────────
BG    = '#080808'
BG2   = '#0f0505'
RED   = '#c0392b'
RED2  = '#e74c3c'
WINE  = '#7b1c1c'
MUTED = '#6b3030'
TEXT  = '#9a7070'
GREEN = '#27ae60'

def style_ax(fig, ax):
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a0f0f')


# ─────────────────────────────────────────────
# LOAD & TRAIN MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("flood.csv")
    df['Flood'] = (df['FloodProbability'] > 0.5).astype(int)
    X = df.drop(columns=['Flood', 'FloodProbability'])
    y = df['Flood']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, df, X, acc, cm, report, feature_imp, X_test, y_test, y_pred

model, df, X, acc, cm, report, feature_imp, X_test, y_test, y_pred = load_and_train()
FEATURES = list(X.columns)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='padding: 16px 0 28px 0;'>
            <div style='font-family: Bebas Neue, sans-serif; font-size: 1.9rem; letter-spacing: 5px;
                        background: linear-gradient(135deg, #c0392b, #ff6b6b);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                🌊 FLOODSENSE
            </div>
            <div style='font-size: 0.65rem; color: #4a1a1a; letter-spacing: 3px; text-transform: uppercase; margin-top: 4px;'>
                Disaster Prediction System
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>NAVIGATE</div>", unsafe_allow_html=True)
    page = st.radio("", ["🏠  Overview", "🔮  Predict", "📊  Dashboard", "📁  Dataset"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='background:#0f0505; border:1px solid #2a0f0f; border-top:2px solid #c0392b;
                    border-radius:10px; padding:18px; text-align:center;'>
            <div style='font-family: Bebas Neue, sans-serif; font-size:2rem; letter-spacing:2px; color:#c0392b;'>{acc*100:.1f}%</div>
            <div style='font-size:0.65rem; color:#6b3030; text-transform:uppercase; letter-spacing:3px; margin-top:4px;'>Model Accuracy</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    flood_count = int(df['Flood'].sum())
    safe_count  = len(df) - flood_count
    st.markdown(f"""
        <div style='background:#0f0505; border:1px solid #2a0f0f; border-radius:10px; padding:16px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                <span style='font-size:0.7rem; color:#6b3030; text-transform:uppercase; letter-spacing:2px;'>Flood Cases</span>
                <span style='font-family: Bebas Neue; color:#c0392b; letter-spacing:1px;'>{flood_count:,}</span>
            </div>
            <div style='display:flex; justify-content:space-between;'>
                <span style='font-size:0.7rem; color:#6b3030; text-transform:uppercase; letter-spacing:2px;'>Safe Cases</span>
                <span style='font-family: Bebas Neue; color:#27ae60; letter-spacing:1px;'>{safe_count:,}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size:0.68rem; color:#2a0a0a; text-align:center; letter-spacing:1px; line-height:2;'>
            Random Forest · 100 Trees<br>
            20 Features · 50,000 Records
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# OVERVIEW PAGE
# ─────────────────────────────────────────────
if "Overview" in page:
    st.markdown("<p class='main-title'>FLOODSENSE</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Disaster Prediction & Resilience System</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{len(df):,}", "Total Records"),
        (f"{len(FEATURES)}", "Input Features"),
        (f"{acc*100:.1f}%", "Model Accuracy"),
        (f"{df['Flood'].mean()*100:.1f}%", "Flood Cases"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("<div class='section-header'>TOP RISK FACTORS</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        style_ax(fig, ax)
        top10  = feature_imp.head(10)
        colors = [RED if i == 0 else RED2 if i < 3 else WINE for i in range(len(top10))]
        ax.barh(top10.index[::-1], top10.values[::-1], color=colors[::-1], height=0.55)
        ax.set_xlabel('Importance Score', color=MUTED, fontsize=9)
        ax.xaxis.label.set_color(MUTED)
        for i, (val, name) in enumerate(zip(top10.values[::-1], top10.index[::-1])):
            ax.text(val + 0.0003, i, f'{val:.4f}', va='center', color=TEXT, fontsize=7.5)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("<div class='section-header'>FLOOD DISTRIBUTION</div>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        style_ax(fig2, ax2)
        sizes = df['Flood'].value_counts()
        wedges, texts, autotexts = ax2.pie(
            sizes,
            labels=['No Flood', 'Flood'],
            colors=[WINE, RED],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'color': TEXT, 'fontsize': 10, 'fontfamily': 'Outfit'},
            wedgeprops={'linewidth': 3, 'edgecolor': BG},
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color('#f0e6e6')
            at.set_fontsize(11)
        centre = plt.Circle((0, 0), 0.5, fc=BG2)
        ax2.add_patch(centre)
        ax2.text(0, 0, f"{df['Flood'].mean()*100:.0f}%\nFLOOD", ha='center', va='center',
                 color=RED, fontsize=12, fontweight='bold', fontfamily='Outfit')
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>ABOUT THIS SYSTEM</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-box'>
        This system uses a <strong style='color:#c0392b;'>Random Forest classifier</strong> trained on
        <strong style='color:#e8b0b0;'>50,000 flood event records</strong> to predict the likelihood of flooding
        based on <strong style='color:#e8b0b0;'>20 environmental and socio-economic factors</strong> including
        monsoon intensity, deforestation, urbanisation, drainage quality, and more.
        <br><br>
        The model achieves <strong style='color:#c0392b;'>{acc*100:.1f}% accuracy</strong> and is designed to assist
        disaster response teams and local authorities in early warning and preparedness planning.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PREDICT PAGE
# ─────────────────────────────────────────────
elif "Predict" in page:
    st.markdown("<p class='main-title'>PREDICTION</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Adjust environmental parameters & run analysis</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>ENVIRONMENTAL PARAMETERS</div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size:0.8rem; color:#6b3030; margin-bottom:20px; letter-spacing:1px;'>
            All values are scored 1–10. Higher scores indicate greater risk contribution.
        </div>
    """, unsafe_allow_html=True)

    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            input_data[feature] = st.slider(
                feature.replace('_', ' '),
                min_value=1, max_value=10, value=5,
                key=feature
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡  ANALYSE FLOOD RISK"):
        input_df    = pd.DataFrame([input_data])
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"""
                <div class='flood-alert'>
                    ⚠️ &nbsp; FLOOD RISK DETECTED &nbsp;·&nbsp; CONFIDENCE: {probability[1]*100:.1f}%
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='safe-alert'>
                    ✅ &nbsp; NO FLOOD RISK DETECTED &nbsp;·&nbsp; CONFIDENCE: {probability[0]*100:.1f}%
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:#c0392b;'>{probability[1]*100:.1f}%</div>
                <div class='metric-label'>Flood Probability</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:#27ae60;'>{probability[0]*100:.1f}%</div>
                <div class='metric-label'>Safe Probability</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            risk_level = "HIGH" if probability[1] > 0.7 else "MEDIUM" if probability[1] > 0.5 else "LOW"
            risk_color = "#c0392b" if risk_level == "HIGH" else "#e67e22" if risk_level == "MEDIUM" else "#27ae60"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:{risk_color};'>{risk_level}</div>
                <div class='metric-label'>Risk Level</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>RISK GAUGE</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 1.4))
        style_ax(fig, ax)
        prob = probability[1]
        ax.barh(0, 1, color='#1a0505', height=0.55, zorder=1)
        bar_color = RED if prob > 0.7 else '#e67e22' if prob > 0.5 else GREEN
        ax.barh(0, prob, color=bar_color, height=0.55, zorder=2)
        ax.axvline(0.5, color='#3d0f0f', linestyle='--', linewidth=1.5, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%\nThreshold', '75%', '100%'], color=TEXT, fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>TOP CONTRIBUTING FACTORS</div>", unsafe_allow_html=True)
        contrib = pd.Series(input_data) * feature_imp
        contrib = contrib.sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 2.5))
        style_ax(fig, ax)
        ax.barh(contrib.index[::-1], contrib.values[::-1], color=RED, height=0.5)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a0f0f')
        plt.tight_layout()
        st.pyplot(fig)


# ─────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────
elif "Dashboard" in page:
    st.markdown("<p class='main-title'>DASHBOARD</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Model Performance & Data Insights</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    r = report
    c1, c2, c3, c4 = st.columns(4)
    perf = [
        (f"{acc*100:.1f}%", "Accuracy"),
        (f"{r['weighted avg']['precision']*100:.1f}%", "Precision"),
        (f"{r['weighted avg']['recall']*100:.1f}%", "Recall"),
        (f"{r['weighted avg']['f1-score']*100:.1f}%", "F1-Score"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], perf):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        style_ax(fig, ax)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['No Flood', 'Flood'],
                    yticklabels=['No Flood', 'Flood'],
                    ax=ax, linewidths=2, linecolor=BG,
                    annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
        ax.tick_params(colors=TEXT)
        ax.set_xlabel('Predicted', color=MUTED, fontsize=9)
        ax.set_ylabel('Actual', color=MUTED, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("<div class='section-header'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        style_ax(fig, ax)
        fi = feature_imp.sort_values()
        colors = [RED if v == fi.max() else RED2 if v >= fi.quantile(0.75) else WINE for v in fi.values]
        fi.plot(kind='barh', ax=ax, color=colors)
        ax.tick_params(colors=TEXT, labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a0f0f')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>FLOOD PROBABILITY DISTRIBUTION</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    style_ax(fig, ax)
    ax.hist(df[df['Flood'] == 0]['FloodProbability'], bins=50, alpha=0.75, color=WINE, label='No Flood')
    ax.hist(df[df['Flood'] == 1]['FloodProbability'], bins=50, alpha=0.75, color=RED,  label='Flood')
    ax.axvline(0.5, color='#ff6b6b', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax.legend(facecolor=BG2, labelcolor=TEXT, edgecolor='#2a0f0f')
    ax.set_xlabel('Flood Probability', color=MUTED, fontsize=9)
    ax.set_ylabel('Count', color=MUTED, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>PER-CLASS METRICS</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for cls_key, cls_label, col in [('0', 'No Flood', col1), ('1', 'Flood', col2)]:
        with col:
            st.markdown(f"""
            <div class='info-box' style='text-align:center;'>
                <div style='font-family: Bebas Neue; font-size:1.2rem; letter-spacing:3px; color:#c0392b; margin-bottom:12px;'>{cls_label}</div>
                <div style='display:flex; justify-content:space-around;'>
                    <div>
                        <div style='font-family: Bebas Neue; font-size:1.6rem; color:#c0392b;'>{r[cls_key]['precision']*100:.1f}%</div>
                        <div style='font-size:0.65rem; color:#6b3030; letter-spacing:2px;'>PRECISION</div>
                    </div>
                    <div>
                        <div style='font-family: Bebas Neue; font-size:1.6rem; color:#c0392b;'>{r[cls_key]['recall']*100:.1f}%</div>
                        <div style='font-size:0.65rem; color:#6b3030; letter-spacing:2px;'>RECALL</div>
                    </div>
                    <div>
                        <div style='font-family: Bebas Neue; font-size:1.6rem; color:#c0392b;'>{r[cls_key]['f1-score']*100:.1f}%</div>
                        <div style='font-size:0.65rem; color:#6b3030; letter-spacing:2px;'>F1-SCORE</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATASET PAGE
# ─────────────────────────────────────────────
elif "Dataset" in page:
    st.markdown("<p class='main-title'>DATASET</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Explore the raw data</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, label) in zip([c1, c2, c3, c4], [
        (f"{len(df):,}", "Total Rows"),
        (f"{len(df.columns)}", "Columns"),
        ("0", "Missing Values"),
        (f"{len(df)*len(df.columns):,}", "Total Cells"),
    ]):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>DATA PREVIEW</div>", unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>STATISTICAL SUMMARY</div>", unsafe_allow_html=True)
    st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>CORRELATION HEATMAP</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    style_ax(fig, ax)
    corr = df[FEATURES].corr()
    sns.heatmap(corr, ax=ax, cmap='Reds', center=0,
                linewidths=0.5, linecolor=BG,
                annot=False, square=True)
    ax.tick_params(colors=TEXT, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
