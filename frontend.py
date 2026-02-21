import os
import warnings
import shap

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import date

from model import train_model as _train_model

warnings.filterwarnings("ignore")

DATA_PATH = "data/sri-lanka-weather-dataset.csv"


def _get_css() -> str:
    return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }
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
    .input-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
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
    .css-1d391kg { background-color: #0f2027 !important; }
    hr { border-color: #e2e8f0; }
    .footer { text-align:center; color:#94a3b8; font-size:0.8rem; padding:2rem 0 1rem; }
</style>
"""


@st.cache_resource(show_spinner="Training model…")
def _cached_train_model(path: str):
    """Cached wrapper around model.train_model for Streamlit."""
    return _train_model(path)


def run_app() -> None:
    """Entry point: page config, load model, render UI."""
    st.set_page_config(
        page_title="Sri Lanka Rainfall Predictor",
        page_icon="🌧️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(_get_css(), unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.error(f"📂 Dataset not found at `{DATA_PATH}`.")
        st.info("Make sure `sri-lanka-weather-dataset.csv` is in a `data/` folder next to `app.py`.")
        return

    with st.spinner("Setting up — this takes ~30 sec on first run…"):
        rf, le_district, le_label, features, df, acc, X_test, y_test, y_pred = _cached_train_model(DATA_PATH)

    districts   = sorted(le_district.classes_.tolist())
    class_names = le_label.classes_

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🌧️ Sri Lanka Rainfall Predictor</h1>
        <p>Enter district & date details — get an instant AI rainfall forecast</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
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

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict", "📊 Model Performance", "📈 Data Explorer", "🔍 Explainability"])

    # Tab 1 — Predict
    with tab1:
        _render_tab_predict(df, districts, features, rf, le_district, le_label)

    # Tab 2 — Model Performance
    with tab2:
        _render_tab_performance(rf, features, y_test, y_pred, class_names, acc)

    # Tab 3 — Data Explorer
    with tab3:
        _render_tab_explorer(df, districts)

    # Tab 4 — SHAP Explainability
    with tab4:
        _render_tab_shap(df, features, rf, le_label)

    st.markdown("""
    <div class="footer">
        Sri Lanka Rainfall Predictor · Random Forest Classifier ·
        Data: <a href="https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset" target="_blank">Kaggle</a>
    </div>
    """, unsafe_allow_html=True)


def _render_tab_predict(df, districts, features, rf, le_district, le_label) -> None:
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
        p_max    = df["monthly_rainfall_mm"].quantile(0.95)
        humidity = float(np.clip(55 + 40 * (monthly_rainfall_mm / (p_max + 1e-9)), 40, 100))
        submitted = st.form_submit_button("🔮 Predict Rainfall", use_container_width=True, type="primary")

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

        prediction = rf.predict(input_data)[0]
        probas     = rf.predict_proba(input_data)[0]
        label      = le_label.inverse_transform([prediction])[0]
        confidence = probas.max() * 100

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
            fig, ax = plt.subplots(figsize=(3, 2.5))
            sorted_classes = le_label.classes_
            sorted_proba   = probas
            bars = ax.barh(sorted_classes, sorted_proba * 100,
                          color=["#2196F3" if c == "High" else "#FF7043" if c == "Low" else "#66BB6A" for c in sorted_classes])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontsize=8)
            ax.set_title("Class Probabilities", fontsize=9, fontweight="bold")
            for bar, p in zip(bars, sorted_proba):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{p*100:.1f}%", va="center", fontsize=8)
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
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


def _render_tab_performance(rf, features, y_test, y_pred, class_names, acc) -> None:
    st.markdown("### 📊 Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy",   f"{acc*100:.2f}%")
    m2.metric("Algorithm",       "Random Forest")
    m3.metric("Trees",           "200")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontweight="bold")
    fi_df = pd.Series(rf.feature_importances_, index=features).sort_values()
    feat_labels_map = {
        "district_enc": "District", "month": "Month", "temperature": "Temp (mean)",
        "temp_max": "Temp (max)", "temp_min": "Temp (min)", "wind_speed": "Wind Speed",
        "wind_direction": "Wind Dir", "monthly_rainfall_mm": "Monthly Rainfall",
        "rain_days": "Rain Hours", "radiation": "Radiation",
        "evapotranspiration": "Evapotranspiration", "latitude": "Latitude",
        "longitude": "Longitude", "elevation": "Elevation", "humidity": "Humidity"
    }
    fi_df.index = [feat_labels_map.get(i, i) for i in fi_df.index]
    colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi_df)))
    fi_df.plot(kind="barh", ax=axes[1], color=colors_fi)
    axes[1].set_title("Feature Importances", fontweight="bold")
    axes[1].set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()
    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]), use_container_width=True)


def _render_tab_explorer(df, districts) -> None:
    st.markdown("### 📈 Data Explorer")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_city = st.selectbox("Filter by City", ["All"] + districts)
    with col_f2:
        selected_class = st.selectbox("Filter by Rainfall Class", ["All", "High", "Medium", "Low"])
    df_view = df.copy()
    if selected_city  != "All":
        df_view = df_view[df_view["district"]      == selected_city]
    if selected_class != "All":
        df_view = df_view[df_view["rainfall_class"] == selected_class]
    st.caption(f"Showing {len(df_view):,} monthly records")
    ch1, ch2 = st.columns(2)
    with ch1:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        monthly_mean = df_view.groupby("month")["monthly_rainfall_mm"].mean()
        month_names  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax3.bar(month_names, [monthly_mean.get(i, 0) for i in range(1, 13)], color="#2196F3", edgecolor="white")
        ax3.set_title("Avg Monthly Rainfall by Month", fontweight="bold", fontsize=10)
        ax3.set_ylabel("mm")
        ax3.tick_params(axis="x", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()
    with ch2:
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        colors_cls = {"High": "#2196F3", "Medium": "#66BB6A", "Low": "#FF7043"}
        counts = df_view["rainfall_class"].value_counts()
        ax4.pie(counts.values, labels=counts.index,
                colors=[colors_cls.get(c, "gray") for c in counts.index],
                autopct="%1.1f%%", startangle=90)
        ax4.set_title("Rainfall Class Distribution", fontweight="bold", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()
    st.dataframe(
        df_view[["district", "month", "year", "temperature", "monthly_rainfall_mm",
                 "wind_speed", "humidity", "rainfall_class"]]
        .round(2)
        .rename(columns={"monthly_rainfall_mm": "rainfall_mm", "rainfall_class": "class"}),
        use_container_width=True,
        height=300
    )

def _render_tab_shap(df, features, rf, le_label) -> None:
    st.markdown("### 🔍 SHAP Explainability")
    st.caption("SHAP explains **why** the model made each prediction — which features pushed the result up or down.")

    feat_labels_map = {
        "district_enc": "District", "month": "Month",
        "temperature": "Temp (mean)", "temp_max": "Temp (max)",
        "temp_min": "Temp (min)", "wind_speed": "Wind Speed",
        "wind_direction": "Wind Dir", "monthly_rainfall_mm": "Monthly Rainfall",
        "rain_days": "Rain Hours", "radiation": "Radiation",
        "evapotranspiration": "Evapotranspiration", "latitude": "Latitude",
        "longitude": "Longitude", "elevation": "Elevation", "humidity": "Humidity"
    }
    readable_features = [feat_labels_map.get(f, f) for f in features]

    from model import get_clean_data
    df_clean = get_clean_data(df, features)
    if len(df_clean) == 0:
        st.warning("No data available for SHAP (all rows dropped as null).")
        return

    # Use explicit feature order and numpy so SHAP shape matches data matrix
    sample_df = df_clean[features].sample(n=min(300, len(df_clean)), random_state=42).reset_index(drop=True)
    X_sample = sample_df[features].to_numpy()

    with st.spinner("Calculating SHAP values — takes ~20 seconds…"):
        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

    # Multi-class: SHAP can return list of (n_samples, n_features) per class
    # or single array (n_samples, n_features, n_classes)
    class_idx = list(le_label.classes_).index("High") if "High" in le_label.classes_ else 0
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) -> take one class
        shap_for_plot = sv[:, :, class_idx]
    elif isinstance(shap_values, list):
        shap_for_plot = np.asarray(shap_values[class_idx])
    else:
        shap_for_plot = sv

    if shap_for_plot.shape != X_sample.shape:
        st.error(f"SHAP shape mismatch: values {shap_for_plot.shape} vs data {X_sample.shape}. Skipping SHAP plots.")
        return

    fig_s, _ = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_for_plot, X_sample,
                      feature_names=readable_features,
                      plot_type="bar", show=False, color="#2196F3")
    plt.gca().set_title("Mean |SHAP Value| — High Rainfall Class", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_s, use_container_width=True)
    plt.close()

    st.info("📌 **How to read:** Longer bar = more important feature.")

    st.markdown("---")

    # ── Beeswarm Plot ────────────────────────────────────
    st.markdown("#### 🐝 Feature Impact Detail (Beeswarm)")
    st.caption("Each dot is one prediction. **Red** = high feature value, **Blue** = low.")

    fig_b, _ = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_for_plot, X_sample,
                      feature_names=readable_features, show=False)
    plt.title("SHAP Beeswarm — High Rainfall Class", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_b, use_container_width=True)
    plt.close()

    st.markdown("---")

    # ── Single Prediction Explainer ──────────────────────
    st.markdown("#### 🎯 Explain a Single Prediction")
    st.caption("Pick a record and see exactly why the model predicted what it did.")

    sample_idx  = st.slider("Select record index", 0, len(sample_df) - 1, 0)
    single_row  = sample_df.iloc[[sample_idx]]
    single_shap = shap_for_plot[sample_idx]
    pred_class  = le_label.inverse_transform(rf.predict(single_row))[0]
    pred_proba  = rf.predict_proba(single_row)[0].max() * 100

    c1, c2 = st.columns([1, 2])
    class_style = {"High": "pred-high", "Medium": "pred-medium", "Low": "pred-low"}
    class_emoji = {"High": "🌊", "Medium": "🌦️", "Low": "☀️"}

    with c1:
        st.markdown(f"""
        <div class="pred-card {class_style[pred_class]}" style="padding:1.2rem">
            <div style="font-size:2.5rem">{class_emoji[pred_class]}</div>
            <div style="font-size:1.4rem; font-weight:bold">{pred_class} Rainfall</div>
            <div style="opacity:0.85">Confidence: {pred_proba:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        fig_w, ax_w = plt.subplots(figsize=(6, 4))
        top_idx   = np.argsort(np.abs(single_shap))[-8:]
        top_vals  = single_shap[top_idx]
        top_names = [readable_features[i] for i in top_idx]
        ax_w.barh(top_names, top_vals,
                  color=["#2196F3" if v > 0 else "#FF7043" for v in top_vals],
                  edgecolor="white")
        ax_w.axvline(0, color="black", linewidth=0.8)
        ax_w.set_title(f"SHAP — Record #{sample_idx}", fontweight="bold", fontsize=10)
        ax_w.set_xlabel("SHAP Value  (+ pushes toward High, − pushes away)")
        ax_w.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig_w, use_container_width=True)
        plt.close()

    st.info("📌 **Blue bars** pushed toward High rainfall. **Red bars** pushed away.")