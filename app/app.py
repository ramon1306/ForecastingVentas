from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_final.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "inferencia_df_transformado.csv"
LAG_COLUMNS = [f"unidades_vendidas_lag_{lag}" for lag in range(1, 8)]
MOVING_AVERAGE_COLUMN = "unidades_vendidas_ma7"
COMPETITOR_COLUMNS = ["Amazon", "Decathlon", "Deporvillage"]
SCENARIOS = {
    "Actual (0%)": 0.0,
    "Competencia -5%": -0.05,
    "Competencia +5%": 0.05,
}

st.set_page_config(
    page_title="Forecast de ventas · Noviembre 2025",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f7f8ff 0%, #f4f6fb 100%); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #24245a 0%, #43347a 100%);
    }
    [data-testid="stSidebar"] * { color: #fff; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e7e8f2; padding: 18px 20px;
        border-radius: 14px; box-shadow: 0 5px 18px rgba(38, 38, 90, .06);
    }
    .hero {
        padding: 24px 28px; margin: 8px 0 22px; border-radius: 18px;
        color: white; background: linear-gradient(105deg, #667eea, #764ba2);
    }
    .hero h1 { margin: 0 0 6px; color: white; }
    .hero p { margin: 0; opacity: .9; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> Any:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_inference_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def format_eur(value: float) -> str:
    return f"€{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def format_number(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def prepare_scenario(
    product_data: pd.DataFrame,
    discount_percent: int,
    competition_change: float,
) -> pd.DataFrame:
    scenario_data = product_data.copy()
    scenario_data["precio_venta"] = scenario_data["precio_base"] * (
        1 - discount_percent / 100
    )

    for column in COMPETITOR_COLUMNS:
        scenario_data[column] = scenario_data[column] * (1 + competition_change)

    scenario_data["precio_competencia"] = scenario_data[COMPETITOR_COLUMNS].mean(axis=1)
    scenario_data["descuento_porcentaje"] = discount_percent
    scenario_data["ratio_precio"] = (
        scenario_data["precio_venta"] / scenario_data["precio_competencia"]
    )
    return scenario_data


def predict_daily(
    product_data: pd.DataFrame,
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    ordered = product_data.sort_values("fecha").reset_index(drop=True).copy()
    has_recursive_columns = all(
        column in ordered.columns for column in [*LAG_COLUMNS, MOVING_AVERAGE_COLUMN]
    )

    previous_lags: list[float] = []
    recent_sales: list[float] = []
    if has_recursive_columns:
        previous_lags = [
            float(ordered[column].astype("float64").iloc[0]) for column in LAG_COLUMNS
        ]
        recent_sales = previous_lags.copy()

    predictions: list[float] = []
    for day_index in range(len(ordered)):
        if day_index > 0 and has_recursive_columns:
            for lag_index, column in enumerate(LAG_COLUMNS):
                ordered.at[day_index, column] = previous_lags[lag_index]
            ordered.at[day_index, MOVING_AVERAGE_COLUMN] = float(
                np.mean(recent_sales[-7:])
            )

        model_input = ordered.loc[[day_index], feature_names]
        prediction = float(model.predict(model_input)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)

        if has_recursive_columns:
            recent_sales.append(prediction)
            previous_lags = [prediction, *previous_lags[:-1]]

    ordered["unidades_predichas"] = predictions
    ordered["ingresos_proyectados"] = (
        ordered["unidades_predichas"] * ordered["precio_venta"]
    )
    ordered["precio_competencia"] = ordered[COMPETITOR_COLUMNS].mean(axis=1)
    ordered["descuento_aplicado"] = (
        (ordered["precio_base"] - ordered["precio_venta"])
        / ordered["precio_base"]
        * 100
    )
    return ordered


def make_daily_table(results: pd.DataFrame) -> pd.DataFrame:
    table = results[
        [
            "fecha",
            "precio_venta",
            "precio_competencia",
            "descuento_aplicado",
            "unidades_predichas",
            "ingresos_proyectados",
        ]
    ].copy()
    table["Día"] = (
        results["fecha"]
        .dt.day_name()
        .map(
            {
                "Monday": "Lunes",
                "Tuesday": "Martes",
                "Wednesday": "Miércoles",
                "Thursday": "Jueves",
                "Friday": "Viernes",
                "Saturday": "Sábado",
                "Sunday": "Domingo",
            }
        )
    )
    table["Fecha"] = results["fecha"].dt.strftime("%d/%m/%Y")
    table.loc[results["fecha"].dt.day.eq(28), "Día"] = "🛍️ Black Friday"
    table = table.rename(
        columns={
            "precio_venta": "Precio venta",
            "precio_competencia": "Precio competencia",
            "descuento_aplicado": "Descuento aplicado",
            "unidades_predichas": "Unidades predichas",
            "ingresos_proyectados": "Ingresos proyectados",
        }
    )
    return table[
        [
            "Fecha",
            "Día",
            "Precio venta",
            "Precio competencia",
            "Descuento aplicado",
            "Unidades predichas",
            "Ingresos proyectados",
        ]
    ]


def style_black_friday(row: pd.Series) -> list[str]:
    if row["Día"] == "🛍️ Black Friday":
        return ["background-color: #fff0f0; color: #9b1c31; font-weight: bold"] * len(
            row
        )
    return [""] * len(row)


def render_daily_chart(results: pd.DataFrame) -> None:
    chart_data = results.assign(dia=results["fecha"].dt.day)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.lineplot(
        data=chart_data,
        x="dia",
        y="unidades_predichas",
        marker="o",
        linewidth=2.8,
        markersize=6,
        color="#667eea",
        ax=ax,
    )
    black_friday = chart_data.loc[chart_data["dia"].eq(28)]
    if not black_friday.empty:
        bf_day = black_friday.iloc[0]
        ax.axvline(28, color="#e63946", linestyle="--", linewidth=1.8, alpha=0.8)
        ax.scatter(
            [28],
            [bf_day["unidades_predichas"]],
            color="#e63946",
            edgecolor="white",
            linewidth=1.5,
            s=120,
            zorder=5,
        )
        ax.annotate(
            "Black Friday · 28 nov",
            xy=(28, bf_day["unidades_predichas"]),
            xytext=(-82, 24),
            textcoords="offset points",
            color="#b42332",
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#e63946"},
        )
    ax.set(
        title="Unidades predichas por día",
        xlabel="Día de noviembre",
        ylabel="Unidades",
        xlim=(1, 30),
    )
    ax.set_xticks(range(1, 31))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22)
    sns.despine(ax=ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def run_app() -> None:
    st.sidebar.title("🎛️ Controles de Simulación")
    try:
        model = load_model()
        inference_data = load_inference_data()
    except (FileNotFoundError, OSError, ValueError, EOFError, ImportError) as error:
        st.error(f"No se pudieron cargar los archivos necesarios: {error}")
        st.info(
            "Verifica que existan `models/modelo_final.joblib` y "
            "`data/processed/inferencia_df_transformado.csv`."
        )
        st.stop()

    if not hasattr(model, "feature_names_in_"):
        st.error(
            "El modelo no expone `feature_names_in_`; no puedo validar sus entradas."
        )
        st.stop()
    feature_names = list(model.feature_names_in_)

    derivable_columns = {
        "descuento_porcentaje",
        "precio_competencia",
        "ratio_precio",
    }
    required_data_columns = {
        "fecha",
        "nombre",
        "precio_base",
        *COMPETITOR_COLUMNS,
        *(set(feature_names) - derivable_columns),
    }
    missing_columns = sorted(required_data_columns - set(inference_data.columns))
    if missing_columns:
        st.error(
            "Faltan columnas requeridas por los datos o el modelo: "
            + ", ".join(missing_columns)
        )
        st.stop()

    inference_data["fecha"] = pd.to_datetime(inference_data["fecha"], errors="coerce")
    inference_data = inference_data.loc[
        inference_data["fecha"].between("2025-11-01", "2025-11-30")
    ].copy()
    if inference_data.empty:
        st.error("El CSV no contiene filas de noviembre de 2025.")
        st.stop()

    product_names = sorted(inference_data["nombre"].dropna().unique().tolist())
    if not product_names:
        st.error("No se encontraron nombres de producto en el CSV.")
        st.stop()

    with st.sidebar.form("simulation_controls"):
        selected_product = st.selectbox("Producto", product_names)
        discount_percent = st.slider(
            "Ajuste de descuento",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            format="%d%%",
            help="Porcentaje aplicado sobre el precio base; un valor positivo reduce el precio.",
        )
        selected_scenario = st.radio("Escenario de competencia", list(SCENARIOS))
        simulate = st.form_submit_button(
            "🚀 Simular Ventas", type="primary", use_container_width=True
        )

    if "simulation_results" not in st.session_state:
        st.session_state["simulation_results"] = None

    if simulate and selected_product is not None:
        product_data = inference_data.loc[
            inference_data["nombre"].eq(selected_product)
        ].copy()
        product_data = product_data.sort_values("fecha")
        expected_dates = pd.date_range("2025-11-01", "2025-11-30", freq="D")
        if (
            product_data["fecha"].nunique() != len(expected_dates)
            or not product_data["fecha"].isin(expected_dates).all()
        ):
            st.error(
                f"{selected_product} no tiene exactamente los 30 días de noviembre de 2025."
            )
            st.stop()

        try:
            with st.spinner(
                "Calculando predicciones diarias y comparando escenarios..."
            ):
                scenarios: dict[str, pd.DataFrame] = {}
                for scenario_name, competition_change in SCENARIOS.items():
                    scenario_data = prepare_scenario(
                        product_data, discount_percent, competition_change
                    )
                    scenarios[scenario_name] = predict_daily(
                        scenario_data, model, feature_names
                    )
                st.session_state["simulation_results"] = {
                    "product": selected_product,
                    "discount": discount_percent,
                    "selected_scenario": selected_scenario,
                    "scenarios": scenarios,
                }
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            st.error(f"No se pudo completar la simulación: {error}")
            st.stop()

    simulation = st.session_state["simulation_results"]
    st.markdown(
        """
        <div class="hero">
          <h1>📈 Simulación de ventas · Noviembre 2025</h1>
          <p>Predicción diaria y análisis de escenarios para tu producto.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lag_features_available = all(
        column in inference_data.columns and column in feature_names
        for column in [*LAG_COLUMNS, MOVING_AVERAGE_COLUMN]
    )
    if not lag_features_available:
        st.warning(
            "Los archivos actuales no incluyen lags ni `unidades_vendidas_ma7` "
            "como entradas del modelo. La app usa exactamente las variables de "
            "`feature_names_in_`; por eso las predicciones de días posteriores no "
            "pueden depender de ventas predichas. Para activar la recursividad, "
            "hay que regenerar el CSV y reentrenar el modelo con esas variables."
        )

    if simulation is None:
        st.info(
            "Selecciona un producto, ajusta los controles y pulsa "
            "**Simular Ventas** para generar el dashboard."
        )
        st.stop()

    results = simulation["scenarios"][simulation["selected_scenario"]]
    st.subheader(f"🛍️ Producto: {simulation['product']}")
    st.caption(
        f"Escenario mostrado: {simulation['selected_scenario']} · "
        f"Ajuste de descuento: {simulation['discount']}%"
    )

    total_units = results["unidades_predichas"].sum()
    total_revenue = results["ingresos_proyectados"].sum()
    average_price = results["precio_venta"].mean()
    average_discount = results["descuento_aplicado"].mean()
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("📦 Unidades proyectadas", format_number(total_units))
    kpi_columns[1].metric("💶 Ingresos proyectados", format_eur(total_revenue))
    kpi_columns[2].metric("🏷️ Precio promedio", format_eur(average_price))
    kpi_columns[3].metric("％ Descuento promedio", f"{average_discount:.2f}%")

    st.divider()
    st.subheader("📊 Predicción diaria")
    render_daily_chart(results)

    st.divider()
    st.subheader("🗓️ Detalle diario")
    daily_table = make_daily_table(results)
    for column in ["Precio venta", "Precio competencia", "Ingresos proyectados"]:
        daily_table[column] = daily_table[column].map(format_eur)
    daily_table["Descuento aplicado"] = daily_table["Descuento aplicado"].map(
        lambda value: f"{value:.2f}%"
    )
    daily_table["Unidades predichas"] = daily_table["Unidades predichas"].map(
        format_number
    )
    st.dataframe(
        daily_table.style.apply(style_black_friday, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("⚖️ Comparativa de escenarios de competencia")
    comparison_columns = st.columns(3)
    for column, scenario_name in zip(comparison_columns, SCENARIOS):
        scenario_results = simulation["scenarios"][scenario_name]
        scenario_units = scenario_results["unidades_predichas"].sum()
        scenario_revenue = scenario_results["ingresos_proyectados"].sum()
        with column:
            st.markdown(f"**{scenario_name}**")
            st.metric("Unidades totales", format_number(scenario_units))
            st.metric("Ingresos totales", format_eur(scenario_revenue))


run_app()
