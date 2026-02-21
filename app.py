"""
Sri Lanka Rainfall Prediction — Streamlit App
=============================================
Setup:
    pip install streamlit scikit-learn pandas numpy matplotlib

Run:
    streamlit run app.py

Make sure your dataset CSV is at:  data/sri-lanka-weather-dataset.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
import os
from datetime import date

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Rainfall Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    /* Main header */
    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .hero h1 { font-size: 2.6rem; margin: 0; color: white; }
    .hero p  { font-size: 1rem; opacity: 0.75; margin-top: 0.5rem; }

    /* Prediction result cards */
    .pred-card {
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        font-family: 'DM Serif Display', serif;
    }
    .pred-high   { background: linear-gradient(135deg, #1565C0, #42A5F5); }
    .pred-medium { background: linear-gradient(135deg, #2E7D32, #66BB6A); }
    .pred-low    { background: linear-gradient(135deg, #BF360C, #FF7043); }

    /* Input section card */
    .input-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Metric chips */
    .metric-chip {
        display: inline-block;
        background: #e8f4fd;
        border: 1px solid #90cdf4;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: #1a4a6e;
        margin: 3px;
        font-weight: 500;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #0f2027 !important; }

    /* Divider */
    hr { border-color: #e2e8f0; }

    /* Footer */
    .footer { text-align:center; color:#94a3b8; font-size:0.8rem; padding:2rem 0 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────────
DATA_PATH = "data/sri-lanka-weather-dataset.csv"

@st.cache_data(show_spinner="Loading & processing dataset…")
def load_and_prepare(path):
    df_raw = pd.read_csv(path)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    col_map = {
        "city":                       "district",
        "time":                       "time",
        "temperature_2m_mean":        "temperature",
        "precipitation_sum":          "daily_precip_mm",
        "windspeed_10m_max":          "wind_speed",
        "shortwave_radiation_sum":    "radiation",
        "et0_fao_evapotranspiration": "evapotranspiration",
        "winddirection_10m_dominant": "wind_direction",
        "weathercode":                "weather_code",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
    df_raw["time"]  = pd.to_datetime(df_raw["time"], errors="coerce")
    df_raw          = df_raw.dropna(subset=["time"])
    df_raw["month"] = df_raw["time"].dt.month
    df_raw["year"]  = df_raw["time"].dt.year

    df = (
        df_raw
        .groupby(["district", df_raw["time"].dt.to_period("M")])
        .agg(
            month               = ("month", "first"),
            year                = ("year", "first"),
            temperature         = ("temperature", "mean"),
            temp_max            = ("temperature_2m_max", "mean"),
            temp_min            = ("temperature_2m_min", "mean"),
            monthly_rainfall_mm = ("daily_precip_mm", "sum"),
            rain_days           = ("precipitation_hours", "sum"),
            wind_speed          = ("wind_speed", "mean"),
            wind_direction      = ("wind_direction", "mean"),
            radiation           = ("radiation", "mean"),
            evapotranspiration  = ("evapotranspiration", "mean"),
            latitude            = ("latitude", "first"),
            longitude           = ("longitude", "first"),
            elevation           = ("elevation", "first"),
        )
        .reset_index()
        .drop(columns=["time"])
    )

    p_max = df["monthly_rainfall_mm"].quantile(0.95)
    df["humidity"] = (55 + 40 * (df["monthly_rainfall_mm"] / (p_max + 1e-9))).clip(40, 100).round(1)

    df["rainfall_class"] = pd.cut(
        df["monthly_rainfall_mm"],
        bins=[-1, 60, 160, 9999],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    return df


@st.cache_resource(show_spinner="Training model…")
def train_model(path):
    df = load_and_prepare(path)

    le_district = LabelEncoder()
    le_label    = LabelEncoder()
    df["district_enc"]       = le_district.fit_transform(df["district"].astype(str))
    df["rainfall_class_enc"] = le_label.fit_transform(df["rainfall_class"].astype(str))

    features = [
        "district_enc", "month", "temperature", "temp_max", "temp_min",
        "wind_speed", "wind_direction",
        "rain_days", "radiation", "evapotranspiration",
        "latitude", "longitude", "elevation"
    ]

    df_clean = df[features + ["rainfall_class_enc"]].dropna()
    X = df_clean[features]
    y = df_clean["rainfall_class_enc"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_split=5, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    return rf, le_district, le_label, features, df, acc, X_test, y_test, y_pred


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌧️ Sri Lanka Rainfall Predictor</h1>
    <p>Enter district & date details — get an instant AI rainfall forecast</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(f"📂 Dataset not found at `{DATA_PATH}`.")
    st.info("Make sure `sri-lanka-weather-dataset.csv` is in a `data/` folder next to `app.py`.")
    st.stop()

with st.spinner("Setting up — this takes ~30 sec on first run…"):
    rf, le_district, le_label, features, df, acc, X_test, y_test, y_pred = train_model(DATA_PATH)

districts    = sorted(le_district.classes_.tolist())
class_names  = le_label.classes_


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Model info
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Info")
    st.metric("Algorithm",      "Random Forest")
    st.metric("Training Rows",  f"{len(df):,}")
    st.metric("Test Accuracy",  f"{acc*100:.2f}%")
    st.metric("Features Used",  str(len(features)))
    st.markdown("---")
    st.markdown("**Cities covered**")
    st.caption("\n".join([f"• {d}" for d in districts]))
    st.markdown("---")
    st.markdown("**Rainfall Classes**")
    st.markdown("🔵 **High** — > 160 mm/month")
    st.markdown("🟢 **Medium** — 60–160 mm/month")
    st.markdown("🔴 **Low** — < 60 mm/month")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Model Performance", "📈 Data Explorer"])


# ═════════════════════════════════════════════════════════
# TAB 1 — PREDICTION FORM
# ═════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Weather Details")
    st.caption("Fill in the fields below and click **Predict** to get a rainfall forecast.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📍 Location & Date**")
            district      = st.selectbox("District / City", districts)
            selected_date = st.date_input("Date", value=date.today(),
                                          min_value=date(2010, 1, 1),
                                          max_value=date(2030, 12, 31))
            month = selected_date.month

            # Auto-fill lat/lon/elevation from dataset for selected district
            dist_row  = df[df["district"] == district].iloc[0] if district in df["district"].values else None
            latitude  = float(dist_row["latitude"])  if dist_row is not None else 7.87
            longitude = float(dist_row["longitude"]) if dist_row is not None else 80.77
            elevation = float(dist_row["elevation"]) if dist_row is not None else 50.0

            st.caption(f"📌 Lat: {latitude:.3f} | Lon: {longitude:.3f} | Elev: {elevation:.0f}m")

        with col2:
            st.markdown("**🌡️ Temperature (°C)**")
            temperature = st.slider("Mean Temperature",   18.0, 38.0, 27.0, 0.1)
            temp_max    = st.slider("Max Temperature",    temperature, 42.0, min(temperature + 4, 42.0), 0.1)
            temp_min    = st.slider("Min Temperature",    10.0, temperature, max(temperature - 5, 10.0), 0.1)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**💨 Wind**")
            wind_speed     = st.slider("Wind Speed (km/h)",         0.0, 80.0, 20.0, 0.5)
            wind_direction = st.slider("Wind Direction (degrees)",  0,   360,  180,  5)

        with col4:
            st.markdown("**🌧️ Rainfall & Other**")
            monthly_rainfall_mm = st.slider("Expected Monthly Rainfall (mm)", 0.0, 600.0, 100.0, 5.0)
            rain_days           = st.slider("Rainy Hours (hrs/month)",         0,   300,   60,    5)
            radiation           = st.slider("Solar Radiation (MJ/m²)",         5.0, 30.0,  15.0, 0.5)
            evapotranspiration  = st.slider("Evapotranspiration (mm)",         1.0, 8.0,   3.5,  0.1)

        # Derived humidity
        p_max    = df["monthly_rainfall_mm"].quantile(0.95)
        humidity = float(np.clip(55 + 40 * (monthly_rainfall_mm / (p_max + 1e-9)), 40, 100))

        submitted = st.form_submit_button("🔮 Predict Rainfall", use_container_width=True, type="primary")

    # ── Run prediction ──────────────────────────────────
    if submitted:
        dist_enc = le_district.transform([district])[0]

        input_data = pd.DataFrame([{
            "district_enc":        dist_enc,
            "month":               month,
            "temperature":         temperature,
            "temp_max":            temp_max,
            "temp_min":            temp_min,
            "wind_speed":          wind_speed,
            "wind_direction":      wind_direction,
            "monthly_rainfall_mm": monthly_rainfall_mm,
            "rain_days":           rain_days,
            "radiation":           radiation,
            "evapotranspiration":  evapotranspiration,
            "latitude":            latitude,
            "longitude":           longitude,
            "elevation":           elevation,
            "humidity":            humidity,
        }])[features]

        prediction  = rf.predict(input_data)[0]
        probas      = rf.predict_proba(input_data)[0]
        label       = le_label.inverse_transform([prediction])[0]
        confidence  = probas.max() * 100

        class_style = {"High": "pred-high", "Medium": "pred-medium", "Low": "pred-low"}
        class_emoji = {"High": "🌊", "Medium": "🌦️", "Low": "☀️"}
        class_desc  = {
            "High":   "Expect heavy rainfall this month. Consider flood preparedness.",
            "Medium": "Moderate rainfall expected. Good conditions for most activities.",
            "Low":    "Dry conditions expected. Monitor for drought risk.",
        }

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            st.markdown(f"""
            <div class="pred-card {class_style[label]}">
                <div style="font-size:3.5rem">{class_emoji[label]}</div>
                <div style="font-size:2rem; font-weight:bold">{label} Rainfall</div>
                <div style="opacity:0.85; font-size:0.95rem; margin-top:0.4rem">{district} · {selected_date.strftime('%B %Y')}</div>
                <div style="margin-top:1rem; font-family:'DM Sans',sans-serif; font-size:0.9rem; opacity:0.9">{class_desc[label]}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.metric("Confidence", f"{confidence:.1f}%")
            st.metric("Month",      selected_date.strftime("%B"))
            st.metric("Est. Humidity", f"{humidity:.0f}%")

        with r3:
            # Probability bar chart
            fig, ax = plt.subplots(figsize=(3, 2.5))
            bar_colors = ["#2196F3", "#FF7043", "#66BB6A"]  # High, Low, Medium
            sorted_classes = le_label.classes_
            sorted_proba   = probas
            bars = ax.barh(sorted_classes, sorted_proba * 100,
                           color=["#2196F3" if c=="High" else "#FF7043" if c=="Low" else "#66BB6A"
                                  for c in sorted_classes])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontsize=8)
            ax.set_title("Class Probabilities", fontsize=9, fontweight="bold")
            for bar, p in zip(bars, sorted_proba):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f"{p*100:.1f}%", va="center", fontsize=8)
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Input summary
        st.markdown("**📋 Input Summary**")
        chip_html = "".join([
            f'<span class="metric-chip">🌡️ {temperature}°C</span>',
            f'<span class="metric-chip">💨 {wind_speed} km/h</span>',
            f'<span class="metric-chip">💧 {monthly_rainfall_mm} mm</span>',
            f'<span class="metric-chip">☁️ {humidity:.0f}% humidity</span>',
            f'<span class="metric-chip">☀️ {radiation} MJ/m²</span>',
            f'<span class="metric-chip">📅 Month {month}</span>',
        ])
        st.markdown(chip_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy",   f"{acc*100:.2f}%")
    m2.metric("Algorithm",       "Random Forest")
    m3.metric("Trees",           "200")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontweight="bold")

    # Feature importances
    fi_df = pd.Series(rf.feature_importances_, index=features).sort_values()
    feat_labels_map = {
        "district_enc":"District","month":"Month","temperature":"Temp (mean)",
        "temp_max":"Temp (max)","temp_min":"Temp (min)","wind_speed":"Wind Speed",
        "wind_direction":"Wind Dir","monthly_rainfall_mm":"Monthly Rainfall",
        "rain_days":"Rain Hours","radiation":"Radiation",
        "evapotranspiration":"Evapotranspiration","latitude":"Latitude",
        "longitude":"Longitude","elevation":"Elevation","humidity":"Humidity"
    }
    fi_df.index = [feat_labels_map.get(i, i) for i in fi_df.index]
    colors_fi   = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi_df)))
    fi_df.plot(kind="barh", ax=axes[1], color=colors_fi)
    axes[1].set_title("Feature Importances", fontweight="bold")
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    # Classification report
    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df.style.background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                 use_container_width=True)


# ═════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ═════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Data Explorer")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_city = st.selectbox("Filter by City", ["All"] + districts)
    with col_f2:
        selected_class = st.selectbox("Filter by Rainfall Class", ["All", "High", "Medium", "Low"])

    df_view = df.copy()
    if selected_city  != "All": df_view = df_view[df_view["district"]      == selected_city]
    if selected_class != "All": df_view = df_view[df_view["rainfall_class"] == selected_class]

    st.caption(f"Showing {len(df_view):,} monthly records")

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        monthly_mean = df_view.groupby("month")["monthly_rainfall_mm"].mean()
        month_names  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        ax3.bar(month_names, [monthly_mean.get(i, 0) for i in range(1,13)],
                color="#2196F3", edgecolor="white")
        ax3.set_title("Avg Monthly Rainfall by Month", fontweight="bold", fontsize=10)
        ax3.set_ylabel("mm"); ax3.tick_params(axis="x", labelsize=8)
        plt.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close()

    with ch2:
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        colors_cls = {"High":"#2196F3","Medium":"#66BB6A","Low":"#FF7043"}
        counts = df_view["rainfall_class"].value_counts()
        ax4.pie(counts.values, labels=counts.index,
                colors=[colors_cls.get(c,"gray") for c in counts.index],
                autopct="%1.1f%%", startangle=90)
        ax4.set_title("Rainfall Class Distribution", fontweight="bold", fontsize=10)
        plt.tight_layout(); st.pyplot(fig4, use_container_width=True); plt.close()

    st.dataframe(
        df_view[["district","month","year","temperature","monthly_rainfall_mm",
                 "wind_speed","humidity","rainfall_class"]]
        .round(2)
        .rename(columns={"monthly_rainfall_mm":"rainfall_mm", "rainfall_class":"class"}),
        use_container_width=True, height=300
    )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Sri Lanka Rainfall Predictor · Random Forest Classifier · 
    Data: <a href="https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset" target="_blank">Kaggle</a>
</div>
""", unsafe_allow_html=True)