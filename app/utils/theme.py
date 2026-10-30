
# app/utils/theme.py

from html import escape

import streamlit as st


COLORS = {
    "background": "#080F20",
    "sidebar": "#0B1530",
    "panel": "#101D36",
    "panel_alt": "#152440",
    "border": "#273858",
    "text": "#EEF4FF",
    "muted": "#A2B2D0",
    "blue": "#38BDF8",
    "violet": "#A78BFA",
    "magenta": "#F472B6",
    "amber": "#FBBF24",
}


def apply_theme():
    st.markdown(
        """
        <style>
        @import url(
          'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
        );

        :root {
            color-scheme: dark;
            --ps-bg: #080F20;
            --ps-panel: #101D36;
            --ps-panel-alt: #152440;
            --ps-border: #273858;
            --ps-text: #EEF4FF;
            --ps-muted: #A2B2D0;
            --ps-blue: #38BDF8;
            --ps-violet: #A78BFA;
            --ps-pink: #F472B6;
        }

        html, body, .stApp {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
              radial-gradient(
                ellipse at 82% 0%,
                rgba(94, 57, 190, .15),
                transparent 36%
              ),
              radial-gradient(
                ellipse at 8% 45%,
                rgba(0, 119, 200, .09),
                transparent 42%
              ),
              var(--ps-bg);
            color: var(--ps-text);
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: transparent;
        }

        .block-container {
            max-width: 1600px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4 {
            color: var(--ps-text) !important;
            letter-spacing: -.035em;
        }

        h1 { font-weight: 800 !important; }
        h2 { font-weight: 700 !important; }

        p, label, li {
            color: var(--ps-text);
        }

        [data-testid="stCaptionContainer"] {
            color: var(--ps-muted) !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg, #0C1732, #091127
            );
            border-right: 1px solid var(--ps-border);
        }

        section[data-testid="stSidebar"] a {
            border-radius: 10px;
            transition:
                background .2s ease,
                transform .2s ease;
        }

        section[data-testid="stSidebar"] a:hover {
            background: rgba(100, 91, 255, .14);
            transform: translateX(3px);
        }

        section[data-testid="stSidebar"]
        [aria-current="page"] {
            background: linear-gradient(
                90deg,
                rgba(102, 75, 255, .35),
                rgba(102, 75, 255, .08)
            );
        }

        /* Hero */
        .dash-hero {
            position: relative;
            overflow: hidden;
            padding: 36px 40px;
            border-radius: 20px;
            border: 1px solid #35436B;
            background:
              radial-gradient(
                circle at 95% 0%,
                rgba(244, 114, 182, .16),
                transparent 36%
              ),
              linear-gradient(
                115deg, #101C3D, #191641
              );
            margin-bottom: 26px;
            animation: ps-rise .55s ease both;
        }

        .dash-hero::after {
            content: "";
            position: absolute;
            width: 350px;
            height: 160px;
            right: -70px;
            bottom: -100px;
            border-radius: 50%;
            border: 1px solid rgba(167,139,250,.35);
            box-shadow:
                0 0 0 20px rgba(167,139,250,.025),
                0 0 0 50px rgba(56,189,248,.025);
            pointer-events: none;
        }

        .dash-hero h1 {
            font-size: clamp(2rem, 3vw, 3rem);
            font-weight: 800;
            margin: 0 0 10px;
            background: linear-gradient(
                90deg, #FFFFFF, #80D9FF, #C4B5FD
            );
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .dash-hero p {
            max-width: 850px;
            color: #B8C8E6;
            line-height: 1.75;
            margin: 0;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background: linear-gradient(
                145deg, #142443, #101C34
            );
            border: 1px solid var(--ps-border);
            border-radius: 15px;
            padding: 20px;
            transition:
                transform .25s ease,
                border-color .25s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: #6D5DE5;
        }

        [data-testid="stMetricValue"] {
            color: #F5F7FF !important;
            font-weight: 800;
        }

        [data-testid="stMetricLabel"] {
            color: #A2B2D0 !important;
        }

        /* Charts and tables */
        div[data-testid="stPlotlyChart"],
        div[data-testid="stDataFrame"] {
            background: #101D36;
            border: 1px solid var(--ps-border);
            border-radius: 16px;
            padding: 12px;
        }

        /* Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: #142440 !important;
            border-color: #354769 !important;
            border-radius: 10px !important;
        }

        /* Buttons */
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: linear-gradient(
                100deg, #5345D8, #7C3AED
            );
            border: 1px solid #8678F2;
            color: white;
            border-radius: 10px;
            font-weight: 600;
            transition: transform .2s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-2px);
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: #A2B2D0;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #C4B5FD;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            border: 1px solid var(--ps-border);
            border-radius: 12px;
            background: var(--ps-panel);
        }

        /* Motion */
        @keyframes ps-rise {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 1.2rem .9rem 3rem;
            }
            .dash-hero {
                padding: 25px 22px;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation: none !important;
                transition: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="dash-hero">
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def polish(fig, height=470):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color=COLORS["text"],
            size=12,
        ),
        title_font=dict(
            color=COLORS["text"],
            size=17,
        ),
        margin=dict(l=20, r=25, t=55, b=35),
        legend_title_text="",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["muted"]),
        ),
        hoverlabel=dict(
            bgcolor="#192746",
            font_color="#FFFFFF",
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(115,135,175,.15)",
        linecolor="#354769",
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["muted"]),
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor="rgba(115,135,175,.15)",
        linecolor="#354769",
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["muted"]),
        zeroline=False,
    )

    return fig
