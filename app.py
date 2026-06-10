"""
California Housing Value Lab — Streamlit App

Interactive California housing analysis with visual exploration and Linear Regression prediction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "california_housing_midterm.csv"
TARGET = "MedianHouseValue_100k"

NUMERIC_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "BedroomRatio",
    "RoomsPerPerson",
    "Income_x_Coastal",
]
CATEGORICAL_FEATURES = ["IncomeBand", "AgeGroup", "CoastalCategory", "CaliforniaRegion"]
BASE_NUMERIC = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

DISPLAY_NAMES: Dict[str, str] = {
    "MedInc": "Median income",
    "HouseAge": "House age",
    "AveRooms": "Average rooms",
    "AveBedrms": "Average bedrooms",
    "Population": "Population",
    "AveOccup": "Average occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "MedianHouseValue_100k": "Median house value ($100k units)",
    "BedroomRatio": "Bedroom ratio",
    "RoomsPerPerson": "Rooms per person",
    "Income_x_Coastal": "Income × coastal signal",
    "IncomeBand": "Income band",
    "AgeGroup": "Housing age group",
    "CoastalCategory": "Coastal category",
    "CaliforniaRegion": "California region",
}

st.set_page_config(
    page_title="California Housing Value Lab",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700;9..144,900&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
:root {
    --ink: #172033; --muted: #5f6f89; --sand: #f8efe0; --cream: #fffaf1;
    --pacific: #0f6f8f; --deep-pacific: #0b3544; --orange: #e66f2f;
    --gold: #e8b95d; --card: rgba(255, 250, 241, 0.88);
}
html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; color: var(--ink); }
.stApp {
    background:
        radial-gradient(circle at 12% 9%, rgba(230,111,47,.18), transparent 24rem),
        radial-gradient(circle at 91% 15%, rgba(15,111,143,.20), transparent 28rem),
        linear-gradient(135deg, #fff8ed 0%, #f7ead4 45%, #eef7fa 100%);
}
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #102b38 0%, #0f4053 55%, #e66f2f 180%); border-right: 1px solid rgba(255,255,255,.15); }
section[data-testid="stSidebar"] * { color: #fffaf1 !important; }
.main .block-container { padding-top: 2rem; padding-bottom: 5rem; max-width: 1240px; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.035em; color: var(--deep-pacific); }
.hero { position: relative; overflow: hidden; border: 1px solid rgba(11,53,68,.13); border-radius: 34px; padding: 34px 38px; background: linear-gradient(135deg, rgba(255,250,241,.94), rgba(238,247,250,.87)), repeating-linear-gradient(45deg, rgba(15,111,143,.06) 0, rgba(15,111,143,.06) 1px, transparent 1px, transparent 13px); box-shadow: 0 24px 75px rgba(16,43,56,.12); margin-bottom: 22px; }
.hero:after { content:""; position:absolute; width:340px; height:340px; border-radius:50%; right:-130px; top:-110px; background: radial-gradient(circle, rgba(232,185,93,.52), rgba(230,111,47,.07) 58%, transparent 68%); }
.eyebrow { display:inline-block; text-transform:uppercase; letter-spacing:.17em; font-size:.78rem; font-weight:700; color:var(--orange); border-bottom:2px solid var(--gold); margin-bottom:10px; }
.hero-title { font-family:'Fraunces', Georgia, serif; font-size:clamp(2.6rem, 6vw, 5.3rem); line-height:.88; letter-spacing:-.06em; max-width:920px; margin:0 0 14px 0; color:var(--deep-pacific); }
.hero-copy { font-size:1.2rem; line-height:1.55; color:var(--muted); max-width:900px; }
.story-card, .warning-card, .success-card, .small-card { border-radius:24px; padding:22px 24px; background:var(--card); border:1px solid rgba(11,53,68,.12); box-shadow:0 14px 38px rgba(16,43,56,.08); }
.warning-card { border-left:7px solid var(--orange); }
.success-card { border-left:7px solid #2d9b71; }
.small-card { min-height: 145px; }
.metric-note { font-size:.9rem; color:var(--muted); margin-top:-6px; }
[data-testid="stMetric"] { background:rgba(255,250,241,.82); border:1px solid rgba(11,53,68,.12); border-radius:22px; padding:16px 18px; box-shadow:0 10px 28px rgba(16,43,56,.07); }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { font-family:'Fraunces', Georgia, serif; color:var(--deep-pacific); }
.stTabs [data-baseweb="tab-list"] { gap:10px; }
.stTabs [data-baseweb="tab"] { border-radius:999px; background:rgba(255,250,241,.76); border:1px solid rgba(11,53,68,.10); padding:10px 18px; }
.stTabs [aria-selected="true"] { background:var(--deep-pacific) !important; color:#fffaf1 !important; }
hr { border:none; border-top:1px solid rgba(11,53,68,.13); margin:1.4rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


def label(col: str) -> str:
    return DISPLAY_NAMES.get(col, col)


def money_from_100k(value: float) -> str:
    return f"${value * 100000:,.0f}"


def normalize_region(lat: float, lon: float) -> str:
    """Simple deterministic region labels for storytelling; not a hidden external dataset."""
    if lat >= 37.0 and lon <= -121.0:
        return "Bay Area / Northern Coast"
    if lat < 35.5 and lon <= -117.5:
        return "Southern California Coast"
    if lat >= 38.0:
        return "Northern California Inland"
    if lon > -119.5:
        return "Central / Inland California"
    return "Central Coast / Valley"


@st.cache_data(show_spinner=False)
def load_base_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical variables and interpretable engineered predictors."""
    out = df.copy()
    out["BedroomRatio"] = out["AveBedrms"] / out["AveRooms"].replace(0, np.nan)
    out["RoomsPerPerson"] = out["AveRooms"] / out["AveOccup"].replace(0, np.nan)
    out["Income_x_Coastal"] = out["MedInc"] * (out["Longitude"].abs() - 114.0)
    if "IncomeBand" not in out.columns:
        out["IncomeBand"] = pd.cut(
            out["MedInc"],
            bins=[-np.inf, 2.56, 3.53, 4.74, np.inf],
            labels=["Lower income", "Middle income", "Upper-middle income", "High income"],
        ).astype(str)
    if "AgeGroup" not in out.columns:
        out["AgeGroup"] = pd.cut(
            out["HouseAge"],
            bins=[0, 15, 30, 45, 60],
            labels=["Newer", "Established", "Older", "Historic"],
            include_lowest=True,
        ).astype(str)
    if "CoastalCategory" not in out.columns:
        out["CoastalCategory"] = np.where(out["Longitude"] <= -120.0, "Coastal / West", "Inland / East")
    if "CaliforniaRegion" not in out.columns:
        out["CaliforniaRegion"] = [normalize_region(lat, lon) for lat, lon in zip(out["Latitude"], out["Longitude"])]
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median(numeric_only=True))
    return out


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return enrich_data(load_base_data())


def selected_feature_groups(selected: Iterable[str]) -> Tuple[List[str], List[str]]:
    selected = list(selected)
    numeric = [c for c in selected if c in NUMERIC_FEATURES]
    categorical = [c for c in selected if c in CATEGORICAL_FEATURES]
    return numeric, categorical


@st.cache_data(show_spinner=False)
def fit_model(feature_cols: Tuple[str, ...], test_size: float, random_state: int) -> dict:
    df = load_data()
    numeric, categorical = selected_feature_groups(feature_cols)
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))

    X = df[list(feature_cols)]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = Pipeline(
        [
            ("prep", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("linear", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": math.sqrt(mean_squared_error(y_test, pred)),
    }

    feature_names = list(model.named_steps["prep"].get_feature_names_out())
    clean_names = [n.replace("numeric__", "").replace("categorical__", "") for n in feature_names]
    coef = pd.DataFrame(
        {"Feature": clean_names, "Coefficient after preprocessing": model.named_steps["linear"].coef_}
    ).sort_values("Coefficient after preprocessing", key=lambda s: s.abs(), ascending=False)
    results = pd.DataFrame({"Actual": y_test, "Predicted": pred})
    results["Residual"] = results["Actual"] - results["Predicted"]
    return {"model": model, "metrics": metrics, "coef": coef, "results": results, "df": df}


def section_hero(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="hero">
  <div class="eyebrow">{eyebrow}</div>
  <div class="hero-title">{title}</div>
  <div class="hero-copy">{copy}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def sidebar_nav() -> str:
    st.sidebar.markdown("## 🏡 Value Lab")
    st.sidebar.caption("California housing analysis + prediction")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Introduction + Exploration",
            "Interactive Data Visualization",
            "Linear Regression Prediction",
        ],
    )
    return page


def page_intro(df: pd.DataFrame) -> None:
    section_hero(
        "Introduction / exploration",
        "What explains California neighborhood home values?",
        "This app explores how income, rooms, occupancy, geography, and neighborhood segments relate to median house value across 20,640 California communities.",
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Total variables", f"{df.shape[1]}")
    c3.metric("Numeric variables", f"{len(numeric_cols)}")
    c4.metric("Categorical variables", f"{len(categorical_cols)}")
    c5.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

    l, r = st.columns([1.1, 0.9])
    with l:
        st.markdown("### Dataset preview")
        st.dataframe(df.head(12), width="stretch")
        st.markdown("### Project story")
        st.markdown(
            """
<div class="success-card">
<b>Main idea:</b> Home values are not explained by one factor alone. Income is important, but location, household density, rooms, and regional categories all add context. The app is organized so viewers can move from data overview, to visual evidence, to a transparent prediction model.
</div>
""",
            unsafe_allow_html=True,
        )
    with r:
        st.markdown("### Column and data-type guide")
        guide = pd.DataFrame(
            {
                "Column": df.columns,
                "Analysis meaning": [label(c) for c in df.columns],
                "Type": [str(t) for t in df.dtypes],
                "Missing": [int(df[c].isna().sum()) for c in df.columns],
            }
        )
        st.dataframe(guide, width="stretch", hide_index=True)

    tab1, tab2, tab3 = st.tabs(["Summary statistics", "Categorical overview", "Data quality"])
    with tab1:
        st.dataframe(df[numeric_cols].describe().T.round(3), width="stretch")
    with tab2:
        for cat in categorical_cols:
            counts = df[cat].value_counts().reset_index()
            counts.columns = [cat, "Rows"]
            fig = px.bar(counts, x=cat, y="Rows", color=cat, title=f"Rows by {label(cat)}")
            fig.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig, width="stretch")
    with tab3:
        missing = df.isna().sum().reset_index()
        missing.columns = ["Column", "Missing values"]
        st.dataframe(missing, width="stretch", hide_index=True)
        if missing["Missing values"].sum() == 0:
            st.success("No missing values found after deterministic feature engineering. This makes the model easier to explain and deploy.")
        else:
            st.warning("Missing values exist and should be handled before modeling.")


def page_visuals(df: pd.DataFrame) -> None:
    section_hero(
        "Interactive data visualization",
        "A visual story from distribution → drivers → segments → geography.",
        "Use the visual tabs to explore the data: home values are continuous, income is the strongest simple driver, categorical segments reveal group differences, and geography matters.",
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Target", "Scatter explorer", "Bar charts", "Line chart", "Map", "Correlation"]
    )

    with tab1:
        fig = px.histogram(
            df,
            x=TARGET,
            nbins=55,
            marginal="box",
            color_discrete_sequence=["#0f6f8f"],
            labels={TARGET: label(TARGET)},
            title="Distribution of the continuous target variable",
        )
        fig.update_layout(template="plotly_white", yaxis_title="Neighborhood count")
        st.plotly_chart(fig, width="stretch")
        st.info("The target is a dollar-value measurement, so the prediction task is regression rather than classification.")

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            xcol = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index("MedInc"))
        with c2:
            ycol = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(TARGET))
        with c3:
            color = st.selectbox("Color by category", categorical_cols, index=categorical_cols.index("IncomeBand"))
        sample_n = st.slider("Sample size for scatter plots", 1000, len(df), 7000, 1000)
        sample = df.sample(sample_n, random_state=11) if sample_n < len(df) else df
        fig = px.scatter(
            sample,
            x=xcol,
            y=ycol,
            color=color,
            opacity=0.58,
            trendline="ols" if xcol != ycol else None,
            labels={c: label(c) for c in df.columns},
            title=f"{label(xcol)} vs {label(ycol)}",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            group_col = st.selectbox("Group by category", categorical_cols, index=categorical_cols.index("IncomeBand"))
        with c2:
            metric = st.selectbox("Numeric metric", [TARGET, "MedInc", "HouseAge", "AveOccup", "Population"], index=0)
        grouped = df.groupby(group_col, as_index=False)[metric].mean().sort_values(metric, ascending=False)
        fig = px.bar(
            grouped,
            x=group_col,
            y=metric,
            color=group_col,
            text=metric,
            labels={group_col: label(group_col), metric: label(metric)},
            title=f"Average {label(metric)} by {label(group_col)}",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with tab4:
        age_summary = df.groupby("HouseAge", as_index=False)[TARGET].mean()
        fig = px.line(
            age_summary,
            x="HouseAge",
            y=TARGET,
            markers=True,
            labels={"HouseAge": label("HouseAge"), TARGET: label(TARGET)},
            title="Line chart: average median value by house age",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    with tab5:
        region_filter = st.multiselect(
            "Filter regions", sorted(df["CaliforniaRegion"].unique()), default=sorted(df["CaliforniaRegion"].unique())
        )
        filtered_map = df[df["CaliforniaRegion"].isin(region_filter)]
        if filtered_map.empty:
            st.warning("Select at least one region to draw the map.")
            return
        map_df = filtered_map.sample(min(12000, len(filtered_map)), random_state=9)
        fig = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color=TARGET,
            size="Population",
            zoom=4.5,
            height=650,
            color_continuous_scale="Turbo",
            labels={TARGET: "Value ($100k)", "Population": "Population"},
            hover_data={"MedInc": True, "HouseAge": True, "CaliforniaRegion": True, TARGET: True},
            title="California median home values by location",
        )
        fig.update_layout(mapbox_style="carto-positron", margin={"r": 0, "t": 35, "l": 0, "b": 0})
        st.plotly_chart(fig, width="stretch")

    with tab6:
        default = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "AveOccup", "Latitude", "Longitude", TARGET]
        corr_cols = st.multiselect("Choose numeric columns", numeric_cols, default=default)
        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                labels={"color": "Correlation"},
                title="Correlation matrix: evidence for feature selection",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Select at least two columns.")

        with st.expander("Seaborn backup visual: target by income band"):
            fig2, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df, x="IncomeBand", y=TARGET, ax=ax, color="#0f6f8f")
            ax.set_title("Higher income bands tend to have higher median home values")
            ax.tick_params(axis="x", rotation=15)
            ax.set_ylabel("Median value ($100k units)")
            st.pyplot(fig2, clear_figure=True)


def page_model(df: pd.DataFrame) -> None:
    section_hero(
        "Linear Regression prediction",
        "Transparent forecasting with metrics, diagnostics, and a live simulator.",
        "This page uses scikit-learn Linear Regression with numeric scaling and one-hot encoding for categorical variables. It reports model accuracy, diagnostics, coefficients, and an interactive prediction simulator.",
    )

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    default_features = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "AveOccup",
        "Latitude",
        "Longitude",
        "BedroomRatio",
        "RoomsPerPerson",
        "IncomeBand",
        "CoastalCategory",
        "CaliforniaRegion",
    ]
    st.markdown("### Model controls")
    c1, c2, c3 = st.columns([1.6, 0.7, 0.7])
    with c1:
        selected = st.multiselect("Choose model features", all_features, default=default_features)
    with c2:
        test_size = st.slider("Test size", 0.15, 0.40, 0.20, 0.05)
    with c3:
        random_state = st.number_input("Random seed", 0, 999, 42)

    if not selected:
        st.error("Please select at least one feature.")
        return

    fit = fit_model(tuple(selected), float(test_size), int(random_state))
    metrics = fit["metrics"]
    results = fit["results"]
    coef = fit["coef"]
    model = fit["model"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² on test set", f"{metrics['r2']:.3f}")
    m2.metric("MAE", money_from_100k(metrics["mae"]))
    m3.metric("RMSE", money_from_100k(metrics["rmse"]))
    m4.metric("Selected features", len(selected))
    st.markdown(
        f"<div class='metric-note'>R² = {metrics['r2']:.3f}, so the selected features explain about {metrics['r2']*100:.1f}% of test-set variation. MAE = {money_from_100k(metrics['mae'])}, meaning the model is off by roughly that amount on average.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        fig = px.scatter(
            results.sample(min(4500, len(results)), random_state=3),
            x="Actual",
            y="Predicted",
            color="Residual",
            color_continuous_scale="RdBu_r",
            labels={"Actual": "Actual value ($100k)", "Predicted": "Predicted value ($100k)"},
            opacity=0.62,
            title="Actual vs predicted values",
        )
        lim_min = min(results["Actual"].min(), results["Predicted"].min())
        lim_max = max(results["Actual"].max(), results["Predicted"].max())
        fig.add_trace(
            go.Scatter(
                x=[lim_min, lim_max],
                y=[lim_min, lim_max],
                mode="lines",
                line=dict(color="#172033", dash="dash"),
                name="Perfect prediction",
            )
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.histogram(
            results,
            x="Residual",
            nbins=55,
            color_discrete_sequence=["#e66f2f"],
            labels={"Residual": "Actual - predicted ($100k units)"},
            title="Residual error distribution",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Coefficients: what pushes value up or down?")
    top_coef = coef.head(20).copy()
    fig = px.bar(
        top_coef,
        x="Coefficient after preprocessing",
        y="Feature",
        orientation="h",
        color="Coefficient after preprocessing",
        color_continuous_scale="Tealrose",
        labels={"Coefficient after preprocessing": "Coefficient"},
        title="Top 20 coefficient magnitudes after preprocessing",
    )
    fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(coef.round(3), width="stretch", hide_index=True)

    st.markdown("### Live prediction simulator")
    sim_inputs: Dict[str, object] = {}
    cols = st.columns(3)
    for i, f in enumerate(BASE_NUMERIC):
        s = df[f]
        default = float(s.median())
        min_v = float(s.quantile(0.01))
        max_v = float(s.quantile(0.99))
        step = float((max_v - min_v) / 100) or 0.01
        with cols[i % 3]:
            sim_inputs[f] = st.slider(label(f), min_value=min_v, max_value=max_v, value=default, step=step)

    sim_base = pd.DataFrame([sim_inputs])
    sim = enrich_data(sim_base)
    # Let users override generated categorical labels, which makes the simulator feel more interactive.
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sim["IncomeBand"] = st.selectbox("Income band", sorted(df["IncomeBand"].unique()), index=1)
    with c2:
        sim["AgeGroup"] = st.selectbox("Age group", sorted(df["AgeGroup"].unique()), index=1)
    with c3:
        sim["CoastalCategory"] = st.selectbox("Coastal category", sorted(df["CoastalCategory"].unique()), index=0)
    with c4:
        sim["CaliforniaRegion"] = st.selectbox("California region", sorted(df["CaliforniaRegion"].unique()), index=0)

    pred_value = float(model.predict(sim.reindex(columns=selected))[0])
    st.markdown(
        f"""
<div class="story-card">
<h3 style="margin-top:0">Predicted median home value: {money_from_100k(pred_value)}</h3>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Model limitations"):
        st.markdown(
            """
- The dataset is historical, so current market prices may differ.
- Very high values are capped in the data, which creates visible prediction limits.
- Linear Regression is interpretable, but nonlinear models could improve accuracy later.
"""
        )


def main() -> None:
    df = load_data()
    page = sidebar_nav()
    if page.startswith("Introduction"):
        page_intro(df)
    elif page.startswith("Interactive"):
        page_visuals(df)
    elif page.startswith("Linear"):
        page_model(df)


if __name__ == "__main__":
    main()
