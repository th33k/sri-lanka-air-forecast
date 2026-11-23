"""Configuration constants, color palettes, and layout parameters for the dashboard."""

FAMILY_COLORS = {
    "Baseline": "#8a8f98",
    "Bagging": "#1b9e77",
    "Boosting": "#e6761b",
    "Stacking": "#7b5ea7",
    "Component": "#b9c0cb",
    "Hybrid RF+ARIMA": "#2b6cb0",
    "Hybrid RF+ANN": "#c2185b",
    "Hybrid RF+Boosting": "#b7791f",
    "Time-series": "#5c6770",
}

AQI_SCALE = [
    [0.0, "#2e9e5b"],    # Good (0-50)
    [0.33, "#e0c200"],   # Moderate (51-100)
    [0.66, "#f08a24"],   # Unhealthy for Sensitive Groups (101-150)
    [1.0, "#c9302c"],    # Unhealthy (151+)
]

AQI_THRESHOLDS = [
    (50, "Good", "#2e9e5b"),
    (100, "Moderate", "#e0c200"),
    (150, "Unhealthy for Sensitive Groups", "#f08a24"),
    (200, "Unhealthy", "#c9302c"),
    (300, "Very Unhealthy", "#7b2d8e"),
    (500, "Hazardous", "#660033"),
]

AQI_ADVISORIES = {
    "Good": "Air quality is satisfactory; air pollution poses little or no risk to the population.",
    "Moderate": "Air quality is acceptable. Unusually sensitive individuals should consider limiting prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Members of sensitive groups (children, elderly, respiratory conditions) may experience health effects; general public is less affected.",
    "Unhealthy": "Some members of the general public may experience health effects; sensitive groups may experience more serious effects. Avoid prolonged outdoor exertion.",
    "Very Unhealthy": "Health alert: Serious risk of health effects for all individuals. Active children and adults should avoid outdoor activities.",
    "Hazardous": "Health emergency warning: Entire population is likely to be affected. Everyone should remain indoors and avoid outdoor physical activity.",
}

PROVINCES = {
    "Western": ["Colombo", "Gampaha", "Kalutara"],
    "Central": ["Kandy", "Matale", "Nuwara Eliya"],
    "Southern": ["Galle", "Matara", "Hambantota"],
    "Northern": ["Jaffna", "Kilinochchi", "Mannar", "Mullaitivu", "Vavuniya"],
    "Eastern": ["Batticaloa", "Ampara", "Trincomalee"],
    "North Western": ["Kurunegala", "Puttalam"],
    "North Central": ["Anuradhapura", "Polonnaruwa"],
    "Uva": ["Badulla", "Moneragala"],
    "Sabaragamuwa": ["Ratnapura", "Kegalle"],
}

ENSEMBLES = ["Bagging", "Boosting", "Stacking"]

PLOT_LAYOUT = dict(
    margin=dict(l=0, r=10, t=30, b=0),
    legend_title_text="",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    hoverlabel=dict(font_size=12),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

SRI_LANKA_CENTER = dict(lat=7.8731, lon=80.7718)
DEFAULT_ZOOM = 6.8
