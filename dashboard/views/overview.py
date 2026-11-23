"""Overview View: Interactive map exploration, island-wide summary, benchmark metrics, and division-level inspection."""
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data import SEASONS, aqi_category, season_of
from dashboard.charts import render_scatter_map
from dashboard.config import AQI_ADVISORIES, AQI_SCALE, AQI_THRESHOLDS, ENSEMBLES, PLOT_LAYOUT


def _aqi_color(val):
    for threshold, _, color in AQI_THRESHOLDS:
        if val <= threshold:
            return color
    return AQI_THRESHOLDS[-1][2]


def render(D, dsds, zone_label, M=None, meta=None, best_name="Ensemble"):
    days = sorted(d.date() for d in D.date.unique())
    first_day, last_day = days[0], days[-1]
    meta = meta or {}

    # Initialize observation date in session state
    if "overview_date" not in st.session_state or st.session_state["overview_date"] not in days:
        st.session_state["overview_date"] = last_day

    # Initialize selected DSD in session state
    all_dsds = sorted(dsds["dsd"].unique())
    if "selected_dsd" not in st.session_state or st.session_state["selected_dsd"] not in all_dsds:
        st.session_state["selected_dsd"] = "Colombo" if "Colombo" in all_dsds else all_dsds[0]

    dsd_to_dist = dict(zip(dsds["dsd"], dsds["district"]))
    current_selected_dsd = st.session_state["selected_dsd"]

    # Model summary: one quiet line, shown under the map
    model_line = ""
    if M is not None and not M.empty:
        ensembles_df = M[M.family.isin(ENSEMBLES)]
        ens = ensembles_df.iloc[0] if not ensembles_df.empty else M.iloc[0]
        persist_df = M[M.model.str.startswith("Persistence")]
        persist = persist_df.iloc[0] if not persist_df.empty else M.iloc[-1]
        overall = M.iloc[0]
        sep = "<span style='color:#cbd5e1;'>&nbsp;|&nbsp;</span>"
        items = [
            f"Top ensemble <b>{ens['model'].split(' (')[0]}</b> (RMSE {ens['RMSE']:.2f}, R² {ens['R2']:.3f})",
            f"<b>{1 - ens['RMSE'] / persist['RMSE']:.0%}</b> lower error than persistence "
            f"({persist['RMSE']:.2f} → {ens['RMSE']:.2f})",
            f"Lowest error <b>{overall['model'].split(' (')[0]}</b> (RMSE {overall['RMSE']:.2f})",
            f"{meta.get('n_zones', 'n/a')} zones · {len(dsds)} DSDs · {len(days):,} days · "
            f"{meta.get('n_test', 0):,} test rows",
        ]
        model_line = (
            f"<div style='font-size:0.74rem; color:#94a3b8; margin-top:2px; line-height:1.6;'>"
            f"{sep.join(items)}</div>"
        )

    # Controls: date, district, division (one clear row, labelled)
    c_date, c_dist, c_dsd = st.columns([1.6, 1.4, 2.2])

    def _step(days_delta):  # move the shared date one day back or forward
        new_day = st.session_state["overview_date"] + datetime.timedelta(days=days_delta)
        st.session_state["overview_date"] = min(max(new_day, first_day), last_day)

    with c_date:
        b_prev, d_in, b_next = st.columns([0.55, 3, 0.55], vertical_alignment="bottom", gap="small")
        b_prev.button("◀", key="ov_prev_day", on_click=_step, args=(-1,), width="stretch",
                      disabled=st.session_state["overview_date"] <= first_day, help="Previous day")
        d_in.date_input("Date", min_value=first_day, max_value=last_day, format="YYYY-MM-DD", key="overview_date")
        b_next.button("▶", key="ov_next_day", on_click=_step, args=(1,), width="stretch",
                      disabled=st.session_state["overview_date"] >= last_day, help="Next day")

    day = st.session_state["overview_date"]
    snap = D[D.date == pd.Timestamp(day)].reset_index(drop=True)

    ALL = "All districts"
    def _on_district():  # user picked a district here: jump to its first division
        dist = st.session_state["ov_dist_select"]
        if dist != ALL:
            st.session_state["selected_dsd"] = sorted(dsds[dsds.district == dist]["dsd"])[0]

    with c_dist:
        dist_options = [ALL] + sorted(dsds["district"].unique())
        current = st.session_state.get("ov_dist_select", ALL)
        sel_dist = dsd_to_dist.get(st.session_state["selected_dsd"], ALL)
        if current not in dist_options:
            current = ALL
        elif current != ALL and current != sel_dist:  # division changed elsewhere (map, other tab)
            current = sel_dist
        st.session_state["ov_dist_select"] = current
        selected_dist = st.selectbox("District", dist_options, key="ov_dist_select", on_change=_on_district)

    with c_dsd:
        eligible = all_dsds if selected_dist == ALL else sorted(dsds[dsds.district == selected_dist]["dsd"].unique())
        st.session_state["ov_dsd_select"] = st.session_state["selected_dsd"]
        st.selectbox(
            "Division (DSD)",
            eligible,
            format_func=lambda d: d if selected_dist != ALL else f"{d} ({dsd_to_dist.get(d, '')})",
            key="ov_dsd_select",
            on_change=lambda: st.session_state.update(selected_dsd=st.session_state["ov_dsd_select"]),
            help="Tip: you can also click a marker on the map.",
        )

    # Island Status Strip
    mean_aqi = snap["aqi"].mean()
    max_row = snap.loc[snap["aqi"].idxmax()]
    min_row = snap.loc[snap["aqi"].idxmin()]
    good_pct = (snap["aqi"] <= 50).mean() * 100
    day_season_name = SEASONS[season_of(day.month)]
    mean_cat = aqi_category(mean_aqi)
    mean_col = _aqi_color(mean_aqi)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 4px 10px; margin-top: 2px; margin-bottom: 6px; font-size: 0.77rem; color: #475569;">
            <div>
                📅 <b>{day.strftime('%A, %d %B %Y')}</b> &nbsp;&bull;&nbsp;
                Season: <b>{day_season_name}</b> &nbsp;&bull;&nbsp;
                Timeline: <b>Day {days.index(day) + 1} of {len(days)}</b>
            </div>
            <div style="display: flex; gap: 14px; align-items: center;">
                <span>Island Mean: <b style="color: {mean_col}; font-size: 0.85rem;">{mean_aqi:.0f}</b> ({mean_cat})</span>
                <span>Peak: <b>{max_row['aqi']:.0f}</b> ({max_row['dsd']})</span>
                <span>Cleanest: <b>{min_row['aqi']:.0f}</b> ({min_row['dsd']})</span>
                <span>Good Air: <b>{good_pct:.0f}%</b> ({int(good_pct * len(snap) / 100)}/{len(snap)} DSDs)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Visual Layout: Left-Aligned Large Map (Height: 820px) & Right-Aligned Detail & District Comparison Pane
    fig = render_scatter_map(
        snap,
        value_col="aqi",
        color_title="US AQI",
        basemap="open-street-map",
        selected_dsd=st.session_state["selected_dsd"],
        height=820,
    )

    left_map_col, right_detail_col = st.columns([7, 5])

    with left_map_col:
        st.caption("Interactive Sri Lanka Air Quality Map (Click any division marker to inspect)")
        map_event = st.plotly_chart(
            fig,
            on_select="rerun",
            selection_mode="points",
            key="overview_map",
            width="stretch",
        )
        if model_line:
            st.markdown(model_line, unsafe_allow_html=True)
        if map_event and "selection" in map_event and map_event["selection"].get("point_indices"):
            selected_idx = map_event["selection"]["point_indices"][0]
            if selected_idx < len(snap):
                clicked_name = snap.iloc[selected_idx]["dsd"]
                if clicked_name != st.session_state["selected_dsd"]:
                    st.session_state["selected_dsd"] = clicked_name  # district box follows on rerun
                    st.rerun()

    with right_detail_col:
        active_dsd = st.session_state["selected_dsd"]
        sel_point = snap[snap["dsd"] == active_dsd]
        if not sel_point.empty:
            item = sel_point.iloc[0]
            cat_name = aqi_category(item["aqi"])
            cat_color = _aqi_color(item["aqi"])
            zone_id = item['zone']
            zone_display = zone_label.get(zone_id, f'Zone {zone_id}')
            advisory_text = AQI_ADVISORIES.get(cat_name, "")

            # Compact Division Header
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-left: 4px solid {cat_color}; padding: 6px 10px; border-radius: 4px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{item['dsd']}</span>
                        <span style="font-size: 0.78rem; color: #64748b; margin-left: 8px;">District: <b>{item['district']}</b> &bull; {zone_display}</span>
                    </div>
                    <span style="background: {cat_color}20; color: {cat_color}; padding: 2px 8px; border-radius: 10px; font-weight: 700; font-size: 0.74rem;">{cat_name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Health Guidance Notice
            st.markdown(
                f"""
                <div style="background: {cat_color}0c; border: 1px solid {cat_color}22; padding: 5px 8px; border-radius: 4px; font-size: 0.74rem; color: #1e293b; margin-bottom: 6px; line-height: 1.35;">
                    <b>Health Guidance</b>: {advisory_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 9 Pollutant Metrics (Compact 3x3 Grid, Zero Truncation)
            m1, m2, m3 = st.columns(3)
            m1.metric("US AQI", f"{item['aqi']:.0f}")
            m2.metric("PM2.5 (µg/m³)", f"{item['pm25']:.1f}")
            m3.metric("PM10 (µg/m³)", f"{item['pm10']:.1f}")

            m4, m5, m6 = st.columns(3)
            m4.metric("NO₂ (µg/m³)", f"{item['no2']:.1f}")
            m5.metric("Ozone (µg/m³)", f"{item['o3']:.1f}")
            m6.metric("SO₂ (µg/m³)", f"{item['so2']:.1f}")

            if "co" in item.index and "aod" in item.index:
                m7, m8, m9 = st.columns(3)
                m7.metric("CO (µg/m³)", f"{item['co']:.1f}")
                m8.metric("AOD (550nm)", f"{item['aod']:.3f}")
                uv_val = f"{item['uv_max']:.1f}" if "uv_max" in item.index else "N/A"
                m9.metric("UV Index", uv_val)

            # Right-Aligned Inspector Tabs (Including District Mean AQI Comparison)
            insp_tab1, insp_tab2, insp_tab3, insp_tab4 = st.tabs([
                "District AQI Comparison",
                "30-Day Trend",
                "CAMS Grid Context",
                "Island Leaders",
            ])

            with insp_tab1:
                # District Mean AQI Comparison Table & Chart (Moved to Right Pane)
                dist_summary = snap.groupby("district")["aqi"].mean().sort_values().reset_index()
                fig_dist_bar = px.bar(
                    dist_summary,
                    x="aqi",
                    y="district",
                    orientation="h",
                    color="aqi",
                    range_color=(0, 150),
                    color_continuous_scale=AQI_SCALE,
                    height=440,
                    labels={"aqi": "Mean US AQI", "district": "District"},
                )
                fig_dist_bar.update_layout(
                    coloraxis_showscale=False,
                    yaxis=dict(tickfont=dict(size=10)),
                    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)", tickfont=dict(size=10)),
                    margin=dict(l=0, r=5, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                fig_dist_bar.update_traces(
                    marker_line_width=0,
                    hovertemplate="District: <b>%{y}</b><br>Mean AQI: <b>%{x:.1f}</b><extra></extra>",
                )
                st.plotly_chart(fig_dist_bar, width="stretch")

            with insp_tab2:
                # 30-Day Historical Trend
                z_hist = D[(D.zone == item["zone"]) & (D.date <= pd.Timestamp(day))].sort_values("date").tail(30)
                if not z_hist.empty:
                    f_hist = go.Figure()
                    f_hist.add_trace(go.Scatter(
                        x=z_hist["date"], y=z_hist["aqi"],
                        mode="lines", name="US AQI",
                        line=dict(color="#1b9e77", width=2),
                        fill="tozeroy", fillcolor="rgba(27,158,119,0.08)",
                        hovertemplate="%{x|%Y-%m-%d}<br>US AQI: <b>%{y:.0f}</b><extra></extra>",
                    ))
                    mean_30d = z_hist["aqi"].mean()
                    f_hist.add_hline(
                        y=mean_30d, line_dash="dash", line_color="#888",
                        annotation_text=f"30d Mean: {mean_30d:.0f}",
                        annotation_position="bottom right",
                        annotation_font_size=9,
                    )
                    f_hist.update_layout(
                        height=420,
                        yaxis_title="AQI", xaxis_title="",
                        margin=dict(l=0, r=10, t=15, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(gridcolor="rgba(0,0,0,0.04)", tickfont=dict(size=10)),
                        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                        showlegend=False,
                    )
                    st.plotly_chart(f_hist, width="stretch")

            with insp_tab3:
                # Neighbor DSDs in same CAMS grid cell
                co_dsds = dsds[dsds.zone == zone_id][["dsd", "district"]].sort_values("dsd")
                st.caption(
                    f"**{zone_display}** represents an atmospheric grid column (~45 km resolution). "
                    f"All **{len(co_dsds)} divisions** inside this grid cell share identical CAMS series:"
                )
                st.dataframe(
                    co_dsds,
                    hide_index=True,
                    height=420,
                    width="stretch",
                    column_config={
                        "dsd": st.column_config.TextColumn("Co-Located Division"),
                        "district": st.column_config.TextColumn("District"),
                    },
                )

            with insp_tab4:
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.caption("Top 5 Cleanest (Lowest AQI)")
                    clean5 = snap.sort_values("aqi", ascending=True).head(5)[["dsd", "district", "aqi"]]
                    st.dataframe(clean5, hide_index=True, height=420, width="stretch")
                with r_col2:
                    st.caption("Top 5 Highest Exposure (Highest AQI)")
                    smog5 = snap.sort_values("aqi", ascending=False).head(5)[["dsd", "district", "aqi"]]
                    st.dataframe(smog5, hide_index=True, height=420, width="stretch")

        else:
            st.info("Select a division from the dropdown or click a marker on the map.")
