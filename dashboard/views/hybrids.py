"""Hybrids View: Comparative analysis of RF+ARIMA, RF+ANN, and RF+Boosting variants."""
import plotly.express as px
import streamlit as st
from dashboard.charts import render_horizontal_bar
from dashboard.config import FAMILY_COLORS, PLOT_LAYOUT


def render(M):
    rf_row = M[M.model == "Random Forest"]
    rf_rmse = rf_row.iloc[0]["RMSE"] if not rf_row.empty else 6.25

    h_df = M[M.family.str.startswith("Hybrid") | M.family.eq("Time-series")].copy()
    h_df["vs Random Forest (%)"] = (h_df["RMSE"] / rf_rmse - 1) * 100
    h_sorted = h_df.sort_values("RMSE")

    st.markdown(
        "Evaluating combinations of **Random Forest** with complementary paradigms: "
        "**ARIMA** (temporal autocorrelation), "
        "**ANN** (smooth non-linear mapping), and "
        "**gradient boosting** (residual correction). "
        "Negative delta values indicate error reduction relative to the standalone Random Forest."
    )

    # Relative performance bar chart
    fig_bar = render_horizontal_bar(
        h_sorted,
        x_col="vs Random Forest (%)",
        y_col="model",
        color_col="family",
        height=max(380, len(h_sorted) * 40),
        fmt="{:+.1f}%",
        title="Relative RMSE Change vs. Standalone Random Forest",
    )
    fig_bar.update_layout(xaxis_title="RMSE Change (%)")
    st.plotly_chart(fig_bar, width="stretch")

    # Comparison table
    st.markdown("#### Reference Benchmark Comparison")
    ref_models = ["Persistence (today = tomorrow)", "Random Forest", "LightGBM", "Stacking (RF+XGB+LGBM+Cat)"]
    ref_df = M[M.model.isin(ref_models) | M.family.str.startswith("Hybrid") | M.family.eq("Time-series")].sort_values("RMSE")
    st.dataframe(
        ref_df[["model", "family", "RMSE", "MAE", "R2"]].round(3),
        hide_index=True, width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model", width="large"),
            "family": st.column_config.TextColumn("Family", width="small"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"),
            "MAE": st.column_config.NumberColumn("MAE", format="%.3f"),
            "R2": st.column_config.NumberColumn("R-Squared", format="%.3f"),
        },
    )

    # Summary callout
    if not h_sorted.empty:
        best_hybrid = h_sorted.iloc[0]["model"]
        best_hybrid_rmse = h_sorted.iloc[0]["RMSE"]
        delta_pct = (best_hybrid_rmse / rf_rmse - 1) * 100
        st.info(
            f"Top hybrid: **{best_hybrid}** (RMSE: {best_hybrid_rmse:.3f}, "
            f"{delta_pct:+.1f}% vs Random Forest at {rf_rmse:.3f})."
        )
