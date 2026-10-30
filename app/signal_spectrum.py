
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# PHARMASIGNAL AI — SIGNAL SPECTRUM DESIGN SYSTEM
# ============================================================

COLORS = {
    "background": "#080F20",
    "sidebar": "#0C162B",
    "surface": "#101D36",
    "surface_alt": "#162541",
    "border": "#2A3A59",
    "text": "#EEF4FF",
    "muted": "#A2B2D0",
    "blue": "#38BDF8",
    "violet": "#A78BFA",
    "magenta": "#F472B6",
    "amber": "#FBBF24",
    "green": "#34D399",
}

MODEL_COLORS = {
    "Logistic Regression": COLORS["blue"],
    "Random Forest": COLORS["violet"],
    "Gradient Boosting": COLORS["magenta"],
}

SIGNAL_COLORS = {
    "ROR": COLORS["blue"],
    "PRR": COLORS["violet"],
    "IC": COLORS["magenta"],
}


def apply_signal_spectrum():
    """Apply the shared dark theme to a Streamlit page."""

    st.markdown(
        """
        <style>
        @import url(
          'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
        );

        html, body, [class*="css"], .stApp {
            font-family: 'DM Sans', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 85% 0%,
                    rgba(116, 67, 211, 0.13),
                    transparent 32%
                ),
                radial-gradient(
                    circle at 8% 20%,
                    rgba(24, 135, 216, 0.09),
                    transparent 35%
                ),
                #080F20;
            color: #EEF4FF;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4 {
            color: #EEF4FF !important;
            letter-spacing: -0.025em;
        }

        p, label {
            color: #C1CEE4;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0B1428,
                #101B35
            );
            border-right: 1px solid #283956;
        }

        section[data-testid="stSidebar"] * {
            color: #D9E5F8;
        }

        section[data-testid="stSidebar"] a:hover {
            background: rgba(167, 139, 250, 0.12);
            border-radius: 9px;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                #142440,
                #101C33
            );
            border: 1px solid #2B3C5C;
            border-radius: 17px;
            padding: 19px 21px;
            box-shadow: 0 8px 26px rgba(0,0,0,0.14);
            transition: transform 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: #7968C8;
        }

        div[data-testid="stMetricLabel"] {
            color: #AABAD5;
        }

        div[data-testid="stMetricValue"] {
            color: #EEF4FF;
            font-weight: 800;
        }

        div[data-testid="stPlotlyChart"] {
            background: linear-gradient(
                150deg,
                #101D36,
                #121C32
            );
            border: 1px solid #293B5B;
            border-radius: 17px;
            padding: 12px;
        }

        div[data-testid="stDataFrame"] {
            background: #101D36;
            border: 1px solid #293B5B;
            border-radius: 15px;
            overflow: hidden;
        }

        div[data-testid="stExpander"] {
            background: #101D36;
            border: 1px solid #293B5B;
            border-radius: 13px;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        div[data-baseweb="select"] > div {
            background-color: #172640 !important;
            border-color: #344768 !important;
            border-radius: 11px !important;
            color: #EEF4FF !important;
        }

        div[data-baseweb="select"] input {
            color: #EEF4FF !important;
        }

        button[data-baseweb="tab"] {
            color: #AABAD5;
            font-weight: 650;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #C4B5FD;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(
                90deg,
                #2563EB,
                #7C3AED
            );
            color: white;
            border: 1px solid #6155B8;
            border-radius: 11px;
            font-weight: 650;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: linear-gradient(
                90deg,
                #3B82F6,
                #8B5CF6
            );
            color: white;
            border-color: #A78BFA;
        }

        hr {
            border-color: #2A3A59;
        }

        .spectrum-hero {
            position: relative;
            overflow: hidden;
            padding: 30px 36px;
            margin-bottom: 26px;
            border-radius: 20px;
            border: 1px solid #33466D;
            background:
                radial-gradient(
                    circle at 90% 10%,
                    rgba(244,114,182,0.16),
                    transparent 35%
                ),
                linear-gradient(
                    115deg,
                    #10233F,
                    #191D46
                );
        }

        .spectrum-kicker {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.17em;
            color: #A78BFA;
            margin-bottom: 12px;
        }

        .spectrum-title {
            font-size: clamp(30px, 3vw, 43px);
            font-weight: 800;
            line-height: 1.15;
            background: linear-gradient(
                90deg,
                #FFFFFF,
                #70CFFF,
                #D1ADFF
            );
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        .spectrum-subtitle {
            max-width: 900px;
            color: #B9C8E5;
            font-size: 14px;
            line-height: 1.7;
        }

        @media(max-width: 760px) {
            .spectrum-hero {
                padding: 24px 20px;
            }
        }

        @media(prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                transition: none !important;
                animation: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def spectrum_header(title, subtitle, kicker="PHARMASIGNAL AI"):
    """Display a Signal Spectrum page header."""

    from html import escape

    st.markdown(
        f"""
        <div class="spectrum-hero">
            <div class="spectrum-kicker">
                {escape(kicker)}
            </div>
            <div class="spectrum-title">
                {escape(title)}
            </div>
            <div class="spectrum-subtitle">
                {escape(subtitle)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_spectrum_figure(fig, height=460):
    """Style a Plotly figure for the dark dashboard."""

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans",
            color=COLORS["text"],
            size=12,
        ),
        margin=dict(l=25, r=25, t=55, b=35),
        legend=dict(
            title_text="",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        ),
        hoverlabel=dict(
            bgcolor=COLORS["surface_alt"],
            font_color="#FFFFFF",
        ),
    )

    fig.update_xaxes(
        gridcolor="#283A56",
        zeroline=False,
        linecolor="#3A4A66",
        tickfont=dict(color=COLORS["muted"]),
    )

    fig.update_yaxes(
        gridcolor="#283A56",
        zeroline=False,
        linecolor="#3A4A66",
        tickfont=dict(color=COLORS["muted"]),
    )

    return fig


# Optional aliases for easier integration with existing pages.
apply_theme = apply_signal_spectrum
polish = style_spectrum_figure


# ============================================================
# COMPATIBILITY HELPERS FOR STATISTICAL SIGNALS
# AND PRIORITY MATRIX
# ============================================================

BLUE = COLORS["blue"]
VIOLET = COLORS["violet"]
PINK = COLORS["magenta"]
AMBER = COLORS["amber"]


def hero(title, subtitle, kicker="PHARMASIGNAL AI"):
    """Compatibility wrapper for existing page headers."""
    spectrum_header(title, subtitle, kicker)


def chart_style(fig, height=460):
    """Compatibility wrapper for existing Plotly charts."""
    return style_spectrum_figure(fig, height=height)


def note(message, kind="info"):
    """Display a themed information or interpretation note."""
    if kind in ("warning", "warn"):
        st.warning(message)
    elif kind == "error":
        st.error(message)
    elif kind == "success":
        st.success(message)
    else:
        st.info(message)
