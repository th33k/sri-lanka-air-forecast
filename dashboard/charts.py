"""Reusable Plotly chart and map builders for the dashboard."""
import plotly.express as px
import plotly.graph_objects as go
from dashboard.config import AQI_SCALE, DEFAULT_ZOOM, FAMILY_COLORS, PLOT_LAYOUT, SRI_LANKA_CENTER


def render_horizontal_bar(df, x_col, y_col, color_col="family", height=480, fmt="{:.3f}", title="", **kwargs):
    """Generates a styled horizontal bar chart with outside text annotations."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        orientation="h",
        height=height,
        color_discrete_map=FAMILY_COLORS,
        text=df[x_col].map(lambda v: fmt.format(v) if isinstance(v, (int, float)) else str(v)),
        title=title,
        **kwargs
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
        **PLOT_LAYOUT,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def render_time_series(df, x_col="date", y_col="aqi", color_col="district", title="", height=430):
    """Generates a line chart for temporal air quality progression."""
    fig = px.line(df, x=x_col, y=y_col, color=color_col, height=height, title=title)
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
        hovermode="x unified",
        **PLOT_LAYOUT,
    )
    fig.update_traces(line_width=1.8)
    return fig


def render_correlation_matrix(corr_df, height=520, title="Inter-Pollutant Correlation Structure"):
    """Generates an annotated diverging heatmap for inter-pollutant correlation analysis."""
    fig = px.imshow(
        corr_df,
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        height=height,
        title=title,
    )
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def render_error_histogram(errors, title="", nbins=60, height=520):
    """Generates a histogram of out-of-sample forecast residuals."""
    fig = px.histogram(errors, nbins=nbins, height=height, title=title)
    fig.update_layout(
        showlegend=False,
        xaxis_title="Forecast - Actual (AQI Points)",
        yaxis_title="Count",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
        **PLOT_LAYOUT,
    )
    fig.update_traces(marker_color="#1b9e77", opacity=0.82)
    return fig


def render_scatter_map(
    df,
    value_col="aqi",
    color_title="US AQI",
    basemap="open-street-map",
    selected_dsd=None,
    height=820,
    range_color=(0, 150),
    marker_size=12,
):
    """Builds a geographic scatter map centered on Sri Lanka with optional marker highlighting."""
    custom_cols = ["district"]
    if "pm25" in df.columns:
        custom_cols.append("pm25")

    fig = px.scatter_map(
        df,
        lat="lat",
        lon="lon",
        color=value_col,
        range_color=range_color,
        color_continuous_scale=AQI_SCALE,
        hover_name="dsd",
        custom_data=custom_cols,
        zoom=DEFAULT_ZOOM,
        center=SRI_LANKA_CENTER,
        height=height,
        map_style=basemap,
    )

    if "pm25" in df.columns:
        hover_tmpl = (
            "<b>%{hovertext}</b><br>"
            "District: %{customdata[0]}<br>"
            + color_title + ": <b>%{marker.color:.0f}</b><br>"
            "PM2.5: %{customdata[1]:.1f} ug/m3<extra></extra>"
        )
    else:
        hover_tmpl = (
            "<b>%{hovertext}</b><br>"
            "District: %{customdata[0]}<br>"
            + color_title + ": <b>%{marker.color:.0f}</b><extra></extra>"
        )

    fig.update_traces(
        marker_size=marker_size,
        opacity=0.88,
        hovertemplate=hover_tmpl,
    )

    if selected_dsd:
        sel_row = df[df["dsd"] == selected_dsd]
        if not sel_row.empty:
            fig.add_trace(go.Scattermap(
                lat=sel_row["lat"],
                lon=sel_row["lon"],
                mode="markers+text",
                marker=dict(
                    size=26,
                    color="rgba(217, 4, 41, 0.2)",
                    symbol="circle",
                ),
                text=[selected_dsd],  # map labels are plain text (no HTML)
                textposition="top right",
                textfont=dict(size=13, color="#111", weight="bold"),
                name="Selected Division",
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title=dict(text=color_title, font=dict(size=11)),
            thickness=14,
            len=0.6,
            yanchor="middle",
            y=0.5,
        ),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    return fig
