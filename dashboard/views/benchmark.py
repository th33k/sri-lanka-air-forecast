"""Benchmark View: Comparative evaluation of Bagging, Boosting, and Stacking models."""
import plotly.express as px
import streamlit as st
from dashboard.charts import render_horizontal_bar
from dashboard.config import ENSEMBLES, FAMILY_COLORS, PLOT_LAYOUT


def render(M, meta):
    st.caption(
        f"Chronological hold-out: {meta.get('n_train', 0):,} zone-days for training "
        f"(before {meta.get('cutoff', 'n/a')}), "
        f"{meta.get('n_test', 0):,} zone-days for testing. "
        f"{len(M)} models compared on the same test period; every tuned model gets the same "
        f"search budget (10 candidates x 4 TimeSeriesSplit folds on training data only)."
    )

    # Controls
    c1, c2 = st.columns([2, 1])
    fam = c1.multiselect(
        "Model Families",
        ["Baseline", "Bagging", "Boosting", "Stacking"],
        default=["Baseline", "Bagging", "Boosting", "Stacking"],
    )
    metric = c2.radio("Rank By", ["RMSE", "MAE", "R2"], horizontal=True)

    m_sub = M[M.family.isin(fam)]
    m_sorted = m_sub.sort_values(metric, ascending=metric != "R2")

    col_chart, col_scatter = st.columns([3, 2])
    with col_chart:
        fig = render_horizontal_bar(
            m_sorted, metric, "model", height=max(420, len(m_sorted) * 32),
            title=f"Hold-Out {metric} by Model",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, width="stretch")

    with col_scatter:
        timed = m_sorted[m_sorted.train_s.notna()]
        if not timed.empty:
            sc = px.scatter(
                timed,
                x="train_s", y="RMSE", color="family", text="model", log_x=True,
                color_discrete_map=FAMILY_COLORS, height=max(420, len(m_sorted) * 32),
                labels={"train_s": "Tuning + Fit Time (seconds, log scale)"},
                title="Computational Cost vs. Prediction Error",
            )
            sc.update_traces(textposition="top center", marker_size=11, textfont_size=9)
            sc.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(sc, width="stretch")

    # Summary tables
    st.markdown("#### Top Performers by Family")
    fam_best = M[M.family.isin(["Baseline", *ENSEMBLES])].sort_values("RMSE").groupby("family").head(1)
    st.dataframe(
        fam_best[["family", "model", "RMSE", "MAE", "R2"]].round(3),
        hide_index=True, width="stretch",
        column_config={
            "family": st.column_config.TextColumn("Family", width="small"),
            "model": st.column_config.TextColumn("Model", width="medium"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"),
            "MAE": st.column_config.NumberColumn("MAE", format="%.3f"),
            "R2": st.column_config.NumberColumn("R-Squared", format="%.3f"),
        },
    )

    with st.expander("Full Benchmark Table with Tuned Hyperparameters", expanded=False):
        display_cols = [c for c in ["model", "family", "RMSE", "MAE", "R2", "train_s", "best_params"] if c in m_sorted.columns]
        st.dataframe(m_sorted[display_cols].round(3), hide_index=True, width="stretch")
