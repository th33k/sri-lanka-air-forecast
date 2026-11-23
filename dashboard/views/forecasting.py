"""Forecasting View: today/tomorrow forecasts and hold-out evaluation against observed values."""
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data import aqi_category
from dashboard.charts import render_scatter_map
from dashboard.config import AQI_ADVISORIES, AQI_SCALE, AQI_THRESHOLDS, PLOT_LAYOUT
from dashboard.data_loader import get_forecast_for_date, get_horizon_scores, has_day2_model, key_dates


def _aqi_color(val):
    for threshold, _, color in AQI_THRESHOLDS:
        if val <= threshold:
            return color
    return AQI_THRESHOLDS[-1][2]


def render(T, best_name, P, dsds, zone_label, all_days=None):
    sub1, sub2 = st.tabs(["Forecast", "Hold-Out Reality Check"])
    dsd_to_dist = dict(zip(dsds["dsd"], dsds["district"]))

    if all_days is None:
        all_days = sorted(d.date() for d in P.date.unique())
    first_day, last_day = all_days[0], all_days[-1]

    # Initialize forecasting date
    if "forecasting_date" not in st.session_state or st.session_state["forecasting_date"] not in all_days:
        st.session_state["forecasting_date"] = st.session_state.get("overview_date", last_day)

    with sub1:
        # One control row: data date | forecast for | jump to a notable day | use overview date
        c_date, c_hz, c_jump, c_sync = st.columns([1.3, 1.6, 2.2, 1.5], vertical_alignment="bottom")
        with c_date:
            st.date_input(
                "Data date",
                min_value=first_day,
                max_value=last_day,
                format="YYYY-MM-DD",
                key="forecasting_date",
                help="The complete day of data the forecast starts from. The latest is yesterday.",
            )

        # The data ends at the last complete day (yesterday), so tomorrow is 2 days after it.
        curr_fc_date = st.session_state["forecasting_date"]
        horizons = [1, 2] if has_day2_model() else [1]
        if st.session_state.get("fc_horizon") not in horizons:
            st.session_state["fc_horizon"] = horizons[-1] if curr_fc_date == last_day else 1
        today = pd.Timestamp.now(tz="Asia/Colombo").date()  # Sri Lanka date, not the server's

        def _label(h):
            d = curr_fc_date + datetime.timedelta(days=h)
            rel = {0: "Today", 1: "Tomorrow"}.get((d - today).days, f"+{h} day{'s' if h > 1 else ''}")
            return f"{rel} · {d:%d %b}"

        with c_hz:
            st.segmented_control("Forecast for", horizons, format_func=_label, required=True,
                                 key="fc_horizon", width="stretch")
        h = st.session_state["fc_horizon"]
        scores = get_horizon_scores()
        err_note = (f" Typical error at this range: ±{scores[h]['RMSE']:.0f} AQI."
                    if scores.get(h) else "")

        with c_jump:
            fc_episodes = key_dates(str(P.date.min().date()))
            valid_fc_episodes = {k: d for k, d in fc_episodes.items() if d in all_days}
            if not valid_fc_episodes:
                valid_fc_episodes = {"Latest Record": last_day}

            def _jump():
                choice = st.session_state["fc_ep_picker"]
                if choice:
                    st.session_state["forecasting_date"] = valid_fc_episodes[choice]

            st.selectbox(
                "Jump to a notable day",
                list(valid_fc_episodes.keys()),
                index=None,
                placeholder="Latest, worst, cleanest, ...",
                key="fc_ep_picker",
                on_change=_jump,
            )

        with c_sync:
            ov_date = st.session_state.get("overview_date", last_day)
            st.button(f"Use Overview date ({ov_date:%d %b})", width="stretch",
                      on_click=lambda: st.session_state.update(forecasting_date=ov_date))

        T = get_forecast_for_date(curr_fc_date, h)
        obs_date = T["obs_date"].iloc[0]
        fc_target_date = T["forecast_date"].iloc[0]
        has_ground_truth = not T["actual_next_day"].isna().all()

        T["category"] = T["forecast"].map(aqi_category)
        T["delta"] = T["forecast"] - T["aqi"]
        if has_ground_truth:
            T["error"] = T["forecast"] - T["actual_next_day"]
            T["abs_error"] = T["error"].abs()

        if has_ground_truth:
            st.markdown(
                f"Forecast for **{fc_target_date:%A, %d %B %Y}** ({h} day{'s' if h > 1 else ''} ahead), "
                f"predicted from conditions observed on **{obs_date:%A, %d %B %Y}** "
                f"using **{best_name}**.{err_note}"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                f"Forecast Mean ({fc_target_date:%d %b})",
                f"{T['forecast'].mean():.0f}",
                aqi_category(T['forecast'].mean()),
                delta_color="off",
            )
            c2.metric(
                f"Observed ({fc_target_date:%d %b})",
                f"{T['actual_next_day'].mean():.0f}",
                f"Delta: {T['forecast'].mean() - T['actual_next_day'].mean():+.1f} AQI",
                delta_color="off",
            )
            c3.metric(
                "Island Mean Absolute Error",
                f"{T['abs_error'].mean():.1f} AQI",
                f"RMSE: {np.sqrt((T['error']**2).mean()):.1f} AQI",
                delta_color="off",
            )
            c4.metric(
                "Exceeding 100 Risk",
                f"{(T['forecast'] > 100).sum()}",
                f"Actual Exceeded: {(T['actual_next_day'] > 100).sum()} DSDs",
                delta_color="off",
            )
        else:
            st.markdown(
                f"Forecast for **{fc_target_date:%A, %d %B %Y}** ({h} day{'s' if h > 1 else ''} ahead) by "
                f"**{best_name}**, from the latest complete day of data, **{obs_date:%A, %d %B %Y}**.{err_note}"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Island Forecast Mean", f"{T['forecast'].mean():.0f}", aqi_category(T['forecast'].mean()), delta_color="off")
            c2.metric("Island Highest Risk", f"{T['forecast'].max():.0f}", f"{T.loc[T['forecast'].idxmax()]['dsd']}")
            c3.metric("Divisions Exceeding 100", f"{(T['forecast'] > 100).sum()}", f"of {len(T)} total DSDs", delta_color="inverse")
            c4.metric("Island Cleanest Area", f"{T['forecast'].min():.0f}", f"{T.loc[T['forecast'].idxmin()]['dsd']}")

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        fig_fc = render_scatter_map(
            T,
            value_col="forecast",
            color_title="Forecast AQI",
            basemap="open-street-map",
            selected_dsd=st.session_state.get("selected_dsd"),
            height=820,
        )

        left, right = st.columns([7, 5])
        with left:
            st.caption("Interactive forecast map: Click any division marker to inspect its forecast.")
            fc_event = st.plotly_chart(fig_fc, on_select="rerun", selection_mode="points", key="fc_map", width="stretch")
            if fc_event and "selection" in fc_event and fc_event["selection"].get("point_indices"):
                idx = fc_event["selection"]["point_indices"][0]
                if idx < len(T):
                    clicked_dsd = T.iloc[idx]["dsd"]
                    if clicked_dsd != st.session_state.get("selected_dsd"):
                        st.session_state["selected_dsd"] = clicked_dsd
                        st.rerun()

        with right:
            # Active Division Forecast Detail Card
            active_dsd = st.session_state.get("selected_dsd", T.iloc[0]["dsd"])
            sel_row = T[T["dsd"] == active_dsd]
            if not sel_row.empty:
                item = sel_row.iloc[0]
                cat_name = item["category"]
                cat_color = _aqi_color(item["forecast"])
                zone_id = item['zone']
                zone_display = zone_label.get(zone_id, f'Zone {zone_id}')
                delta_val = item['delta']
                delta_str = f"{delta_val:+.0f} AQI points"
                delta_desc = "Worsening" if delta_val > 0 else "Improving" if delta_val < 0 else "Unchanged"
                delta_color = "normal" if delta_val <= 0 else "inverse"

                st.markdown(
                    f"""
                    <div style="background: rgba(0,0,0,0.02); border-left: 4px solid {cat_color}; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">
                        <div style="font-size: 1.25rem; font-weight: 700;">{item['dsd']}</div>
                        <div style="font-size: 0.83rem; color: #555; margin-top: 2px;">
                            District: <b>{item['district']}</b> &nbsp;|&nbsp;
                            CAMS Zone: <b>{zone_display}</b> &nbsp;|&nbsp;
                            Forecast: <span style="background: {cat_color}22; color: {cat_color}; padding: 2px 8px; border-radius: 12px; font-weight: 700; font-size: 0.8rem;">{cat_name}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Forecast KPI row
                if has_ground_truth:
                    fc_col1, fc_col2, fc_col3 = st.columns(3)
                    fc_col1.metric(f"Forecast ({fc_target_date:%d %b})", f"{item['forecast']:.0f}")
                    fc_col2.metric("Actual Ground Truth", f"{item['actual_next_day']:.0f}")
                    err_val = item['error']
                    fc_col3.metric("Prediction Error", f"{err_val:+.0f}", "Accurate" if abs(err_val) <= 5 else "Over-predicted" if err_val > 0 else "Under-predicted", delta_color="off")
                else:
                    fc_col1, fc_col2, fc_col3 = st.columns(3)
                    fc_col1.metric(f"Forecast ({fc_target_date:%d %b})", f"{item['forecast']:.0f}")
                    fc_col2.metric(f"Observed ({obs_date:%d %b})", f"{item['aqi']:.0f}")
                    fc_col3.metric("Projected Delta", delta_str, delta_desc, delta_color=delta_color)

                # Health guidance
                advisory = AQI_ADVISORIES.get(cat_name, "")
                st.markdown(
                    f"""
                    <div style="background: {cat_color}12; border: 1px solid {cat_color}33; padding: 10px 14px; border-radius: 6px; font-size: 0.82rem; line-height: 1.45; color: #222; margin-bottom: 12px;">
                        <b>Health guidance for {fc_target_date:%d %b}</b>: {advisory}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Division Forecast Directories
            tab_tables = st.tabs(["High-Risk Top 10", "Cleanest Top 10", "District Filter"])
            cols_to_show = ["dsd", "district", "aqi", "forecast", "delta", "category"]
            if has_ground_truth:
                cols_to_show = ["dsd", "district", "aqi", "forecast", "actual_next_day", "error", "category"]

            cfg_dict = {
                "dsd": st.column_config.TextColumn("Division", width="medium"),
                "district": st.column_config.TextColumn("District", width="small"),
                "aqi": st.column_config.NumberColumn(f"Observed {obs_date:%d %b}", format="%d"),
                "forecast": st.column_config.NumberColumn(f"Forecast {fc_target_date:%d %b}", format="%d"),
                "delta": st.column_config.NumberColumn("Delta", format="%+d"),
                "actual_next_day": st.column_config.NumberColumn(f"Actual {fc_target_date:%d %b}", format="%d"),
                "error": st.column_config.NumberColumn("Error", format="%+d"),
                "category": st.column_config.TextColumn("Category", width="large"),
            }

            with tab_tables[0]:
                top10_high = T.sort_values("forecast", ascending=False).head(10)[cols_to_show]
                st.dataframe(
                    top10_high.round(0),
                    hide_index=True,
                    height=440,
                    width="stretch",
                    column_config=cfg_dict,
                )

            with tab_tables[1]:
                top10_clean = T.sort_values("forecast", ascending=True).head(10)[cols_to_show]
                st.dataframe(
                    top10_clean.round(0),
                    hide_index=True,
                    height=440,
                    width="stretch",
                    column_config=cfg_dict,
                )

            with tab_tables[2]:
                dist_options_fc = ["All Districts"] + sorted(dsds.district.unique())
                if "fc_dist_filter" in st.session_state and st.session_state["fc_dist_filter"] not in dist_options_fc:
                    del st.session_state["fc_dist_filter"]
                dist_filter_fc = st.selectbox("Select District", dist_options_fc, key="fc_dist_filter")
                filtered_T = T if dist_filter_fc == "All Districts" else T[T.district == dist_filter_fc]
                st.dataframe(
                    filtered_T[cols_to_show].sort_values("forecast", ascending=False).round(0),
                    hide_index=True,
                    height=440,
                    width="stretch",
                    column_config=cfg_dict,
                )

    with sub2:
        model_cols = [c for c in P.columns if c not in ("date", "zone", "aqi", "y")]

        col_ctl1, col_ctl2 = st.columns([1, 2])

        # Synchronize default district with active division
        active_dsd = st.session_state.get("selected_dsd", "Colombo")
        active_dist = dsd_to_dist.get(active_dsd, "Colombo")
        all_districts = sorted(dsds.district.unique())
        # Show the shared selection; only write it back when the user changes a box here.
        st.session_state["fc_check_dist"] = active_dist if active_dist in all_districts else all_districts[0]

        def _on_district():
            first = sorted(dsds[dsds.district == st.session_state["fc_check_dist"]].dsd)[0]
            st.session_state["selected_dsd"] = first

        d_sel = col_ctl1.selectbox("District", all_districts, key="fc_check_dist", on_change=_on_district)
        dsd_choices = sorted(dsds[dsds.district == d_sel].dsd)
        st.session_state["fc_check_dsd"] = active_dsd if active_dsd in dsd_choices else dsd_choices[0]
        dsd_sel = col_ctl1.selectbox(
            "Division", dsd_choices, key="fc_check_dsd",
            on_change=lambda: st.session_state.update(selected_dsd=st.session_state["fc_check_dsd"]),
        )

        zone = int(dsds[(dsds.district == d_sel) & (dsds.dsd == dsd_sel)].zone.iloc[0])
        col_ctl1.caption(
            f"{dsd_sel} maps to CAMS grid zone **{zone_label.get(zone, f'Zone {zone}')}**. "
            f"All DSDs in the same zone share identical forecast series."
        )

        default_picks = [m for m in ("Persistence (today = tomorrow)", "Random Forest", best_name) if m in model_cols]
        models_picked = col_ctl2.multiselect("Models to Compare", model_cols, default=default_picks)

        q_zone = P[P.zone == zone].sort_values("date")

        # Local zone error performance banner
        if best_name in q_zone.columns and "Persistence (today = tomorrow)" in q_zone.columns:
            local_best_rmse = float(np.sqrt(((q_zone[best_name] - q_zone["y"]) ** 2).mean()))
            local_pers_rmse = float(np.sqrt(((q_zone["Persistence (today = tomorrow)"] - q_zone["y"]) ** 2).mean()))
            local_reduc = (1 - local_best_rmse / local_pers_rmse) * 100

            z1, z2, z3 = st.columns(3)
            z1.metric(f"Hold-Out RMSE ({best_name})", f"{local_best_rmse:.2f}", f"Local Grid Zone {zone}")
            z2.metric("Persistence Baseline RMSE", f"{local_pers_rmse:.2f}", "Local Grid Zone")
            z3.metric("Local Error Reduction", f"{local_reduc:.1f}%", "vs Persistence Baseline")

        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=q_zone.date + pd.Timedelta(days=1), y=q_zone.y,
            name="Actual Observed", line=dict(color="black", width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Actual AQI: <b>%{y:.0f}</b><extra></extra>",
        ))
        for m in models_picked:
            fig_val.add_trace(go.Scatter(
                x=q_zone.date + pd.Timedelta(days=1), y=q_zone[m],
                name=m, line=dict(width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>" + m + ": <b>%{y:.0f}</b><extra></extra>",
            ))
        fig_val.update_layout(
            yaxis_title="US AQI", height=460,
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_val, width="stretch")

        # Error stratification
        st.markdown("#### Error Stratification Analysis")
        strat_mode = st.radio("Stratify RMSE By", ["district", "month", "AQI level"], horizontal=True)
        PP = P.merge(dsds.groupby("zone").agg(district=("district", "first")).reset_index(), on="zone")
        PP["month"] = PP.date.dt.strftime("%Y-%m")
        PP["AQI level"] = pd.cut(PP.y, [0, 50, 75, 100, 500], labels=["<=50 Good", "51-75 Moderate", "76-100 Sensitive", ">100 Unhealthy"])
        rows = [
            {strat_mode: key, "model": m, "RMSE": float(np.sqrt(((grp[m] - grp["y"]) ** 2).mean()))}
            for key, grp in PP.groupby(strat_mode, observed=True) for m in models_picked
        ]
        if rows:
            fig_strat = px.bar(
                pd.DataFrame(rows), x=strat_mode, y="RMSE", color="model",
                barmode="group", height=420,
                title=f"RMSE Stratified by {strat_mode.title()}",
                labels={"RMSE": "Hold-Out RMSE (lower is better)"},
            )
            fig_strat.update_layout(
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
                **PLOT_LAYOUT,
            )
            st.plotly_chart(fig_strat, width="stretch")
