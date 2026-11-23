"""Sri Lanka Air Quality forecasting dashboard. Run: streamlit run app.py"""
from pathlib import Path

import streamlit as st

from dashboard.config import ENSEMBLES
from dashboard.data_loader import get_dsd_daily, get_dsds, get_forecast_for_date, get_results, get_zones
from dashboard.views import benchmark, exploration, forecasting, hybrids, interpretation, methodology, overview
from src.data import DATA_DIR

PERSISTENT_KEYS = [
    "overview_date", "ov_dist_select",
    "forecasting_date", "fc_check_dist", "fc_check_dsd",
    "exp_dist_widget", "exp_pol_widget", "exp_date_slider",
]

st.set_page_config(
    page_title="Sri Lanka Air Quality: Ensemble Forecasting",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Flag_of_Sri_Lanka.svg/32px-Flag_of_Sri_Lanka.svg.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(f"<style>{(Path(__file__).parent / 'dashboard' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main():
    # Verify dataset presence
    if not (DATA_DIR / "zone_daily.csv.gz").exists():
        st.error("Dataset not found. Run `python -m src.prepare_data` first.")
        st.stop()

    # Load cached datasets and model artifacts
    Z = get_zones()
    D = get_dsd_daily()
    days = sorted(d.date() for d in D.date.unique())
    M, P, IMP, meta, best_name = get_results()

    if M is None or P is None:
        st.error("Model benchmark results not found. Run `python -m src.train` first.")
        st.stop()

    dsds = get_dsds()
    zone_label = dsds.groupby("zone")["zone_name"].first()
    all_dsds = sorted(dsds["dsd"].unique())
    dsd_to_dist = dict(zip(dsds["dsd"], dsds["district"]))

    # Initialize active division in session state
    if "selected_dsd" not in st.session_state or st.session_state["selected_dsd"] not in all_dsds:
        st.session_state["selected_dsd"] = "Colombo" if "Colombo" in all_dsds else all_dsds[0]

    # Header
    ensembles_df = M[M.family.isin(ENSEMBLES)]
    ens = ensembles_df.iloc[0] if not ensembles_df.empty else M.iloc[0]
    persist_df = M[M.model.str.startswith("Persistence")]
    persist = persist_df.iloc[0] if not persist_df.empty else M.iloc[-1]

    # Header (Streamlit's top bar is transparent, so its buttons sit to the right of this)
    st.markdown(
        f"""
        <div class="app-topbar">
            <div class="app-title">Sri Lanka Air Quality</div>
            <div class="app-line2"><span class="app-sub">US AQI forecasts for today and tomorrow</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Keep control values when their tab is hidden (Streamlit forgets widgets it does not draw)
    for key in PERSISTENT_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]

    # Only the open tab is built on each run, which keeps the app fast
    views = {
        "Overview": lambda: overview.render(D, dsds, zone_label, M, meta, best_name),
        "Data Exploration": lambda: exploration.render(D, Z),
        "Model Benchmark": lambda: benchmark.render(M, meta),
        "Hybrid Architectures": lambda: hybrids.render(M),
        "Forecasting": lambda: forecasting.render(get_forecast_for_date(), best_name, P, dsds, zone_label, all_days=days),
        "Model Interpretation": lambda: interpretation.render(IMP, P, best_name, M, persist, ens),
        "Methodology": lambda: methodology.render(meta, dsds),
    }
    for tab, render in zip(st.tabs(list(views), key="main_tab", on_change="rerun"), views.values()):
        if tab.open:
            with tab:
                render()


if __name__ == "__main__":
    main()
