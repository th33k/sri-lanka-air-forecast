"""Exploration View: Longitudinal trends, seasonal distributions, and pollutant correlations."""
import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data import SEASONS, season_of
from dashboard.charts import render_correlation_matrix, render_time_series
from dashboard.config import PLOT_LAYOUT
from dashboard.data_loader import get_district_daily


def render(D, Z):
    D = get_district_daily()  # charts here show district averages; this table is much smaller
    days = sorted(d.date() for d in D.date.unique())
    first_day, last_day = days[0], days[-1]
    all_districts = sorted(D.district.unique())

    # Empty selection = all districts
    if "exp_dist_widget" in st.session_state:
        st.session_state["exp_dist_widget"] = [d for d in st.session_state["exp_dist_widget"] if d in all_districts]

    # Date range lives in the slider's own state; preset buttons set it via callbacks
    if "exp_date_slider" not in st.session_state:
        st.session_state["exp_date_slider"] = (first_day, last_day)

    def _set_range(start, end):
        st.session_state["exp_date_slider"] = (max(first_day, start), min(last_day, end))

    st.markdown("##### Regional and Indicator Selection")

    c1, c2 = st.columns([3, 2])
    with c1:
        chosen_districts = st.multiselect(
            "Districts",
            all_districts,
            key="exp_dist_widget",
            placeholder="All districts",
            help="Pick districts to compare. Leave empty to show all districts.",
        )

    var_map = {
        "aqi": "US AQI (daily mean)",
        "pm25": "PM2.5 (ug/m3)",
        "pm10": "PM10 (ug/m3)",
        "no2": "NO2 (ug/m3)",
        "o3": "Ozone (ug/m3)",
        "so2": "SO2 (ug/m3)",
        "co": "CO (ug/m3)",
        "aod": "Aerosol Optical Depth",
        "dust": "Dust (ug/m3)",
    }
    with c2:
        pol = st.selectbox("Atmospheric Indicator", list(var_map.keys()), format_func=lambda k: var_map[k], key="exp_pol_widget")

    # Date range presets and slider
    dt_p1, dt_p2, dt_p3, dt_p4 = st.columns(4)
    dt_p1.button("Full Timeline", width="stretch", on_click=_set_range, args=(first_day, last_day))
    dt_p2.button("Last 90 Days", width="stretch", on_click=_set_range,
                 args=(last_day - datetime.timedelta(days=90), last_day))
    dt_p3.button("NE Monsoon (Dec-Feb)", width="stretch", on_click=_set_range,
                 args=(datetime.date(2023, 12, 1), datetime.date(2024, 2, 29)))
    dt_p4.button("SW Monsoon (May-Sep)", width="stretch", on_click=_set_range,
                 args=(datetime.date(2024, 5, 1), datetime.date(2024, 9, 30)))

    start_d, end_d = st.slider(
        "Date Range Filter",
        min_value=first_day,
        max_value=last_day,
        format="YYYY-MM-DD",
        key="exp_date_slider",
    )

    # Filter dataset slice
    sel = D[
        (D.date >= pd.Timestamp(start_d)) &
        (D.date <= pd.Timestamp(end_d))
    ]
    active_districts = chosen_districts if chosen_districts else all_districts
    sel = sel[sel.district.isin(active_districts)]

    # Slice Summary KPI Strip
    if not sel.empty:
        mean_val = sel[pol].mean()
        max_idx = sel[pol].idxmax()
        max_item = sel.loc[max_idx]
        min_val = sel[pol].min()
        n_dist_active = sel["district"].nunique()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Mean {pol.upper()}", f"{mean_val:.1f}")
        m2.metric(f"Peak {pol.upper()}", f"{max_item[pol]:.1f}", f"{max_item['district']} ({max_item['date'].strftime('%Y-%m-%d')})")
        m3.metric(f"Minimum {pol.upper()}", f"{min_val:.1f}")
        m4.metric("Districts Active", f"{n_dist_active} of {len(all_districts)}")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Trend Analysis", "Seasonal Variation", "Monthly Heatmap", "Correlation Matrix"])

    with t1:
        if not sel.empty:
            s = sel.groupby(["district", "date"])[pol].mean().reset_index()
            s[pol] = s.groupby("district")[pol].transform(lambda x: x.rolling(7, min_periods=1).mean())
            fig = render_time_series(
                s, x_col="date", y_col=pol, color_col="district",
                title=f"7-Day Rolling Average: {var_map[pol]} by District",
                height=440,
            )
            fig.update_layout(yaxis_title=var_map[pol])
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Rolling 7-day averages smooth short-term variance. "
                "Highlights sustained transboundary or local pollution episodes across districts."
            )
        else:
            st.info("No data available for the chosen filters.")

    with t2:
        if not sel.empty:
            month_to_season = {m: SEASONS[season_of(m)] for m in range(1, 13)}
            sel_season = sel.assign(season=sel.date.dt.month.map(month_to_season))
            fig_box = px.box(
                sel_season,
                x="season", y=pol, color="district",
                category_orders={"season": list(SEASONS.values())},
                height=440,
                title="Distribution Across Monsoon Seasons",
                labels={pol: var_map[pol], "season": "Monsoon Season"},
            )
            fig_box.update_layout(
                yaxis_title=var_map[pol],
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
                **PLOT_LAYOUT,
            )
            st.plotly_chart(fig_box, width="stretch")
            st.caption(
                "Sri Lanka has four monsoon seasons. Air quality is generally worst from December to April "
                "(North-East monsoon and 1st inter-monsoon), when winds carry particulates from the Indian "
                "subcontinent, and cleanest in the South-West monsoon (May-Sep), when winds come off the ocean."
            )
        else:
            st.info("No data available for the chosen filters.")

    with t3:
        if not sel.empty:
            h = sel.assign(year=sel.date.dt.year, month=sel.date.dt.month_name().str[:3]).groupby(["year", "month"])[pol].mean().reset_index()
            order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            h_pivot = h.pivot(index="year", columns="month", values=pol).reindex(columns=order)
            fig_heat = px.imshow(
                h_pivot, text_auto=".1f", aspect="auto",
                color_continuous_scale="YlOrRd", height=320,
                title=f"Monthly Mean Concentration Matrix ({var_map[pol]})",
            )
            fig_heat.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig_heat, width="stretch")
            st.caption(
                "Monthly aggregated matrix reveals annual cycles and identifies anomalous months."
            )
        else:
            st.info("No data available for the chosen filters.")

    with t4:
        cols = ["aqi", "pm25", "pm10", "no2", "so2", "o3", "co", "aod", "dust", "uv_max"]
        available = [c for c in cols if c in Z.columns]
        fig_corr = render_correlation_matrix(Z[available].corr().round(2), height=520)
        st.plotly_chart(fig_corr, width="stretch")
        st.caption(
            f"Pearson correlation matrix across all {len(Z):,} zone-day observations. "
            "AQI follows PM2.5 and PM10 most closely, and PM2.5 and PM10 move almost together. "
            "Ozone rises and falls with particulates here (positive correlation), consistent with "
            "shared regional air masses."
        )
