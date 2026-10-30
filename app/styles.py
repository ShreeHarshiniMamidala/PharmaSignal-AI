
import streamlit as st


def apply_styles():
    st.markdown("""
    <style>
    :root {
        --primary: #0f766e;
        --accent: #06b6d4;
        --indigo: #4f46e5;
        --ink: #0f172a;
        --muted: #64748b;
        --border: #e2e8f0;
        }

    .stApp {
        background:
            radial-gradient(circle at 8% 4%,
                rgba(20,184,166,.10), transparent 28%),
            radial-gradient(circle at 94% 8%,
                rgba(99,102,241,.09), transparent 26%),
            #f8fafc;
        color: var(--ink);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -.035em;
    }

    .pharma-hero, .dashboard-hero {
        position: relative;
        overflow: hidden;
        padding: 44px 46px;
        border-radius: 26px;
        background: linear-gradient(
            120deg, #115e59, #0891b2 52%, #4338ca);
        box-shadow: 0 20px 45px rgba(15,118,110,.18);
        animation: fadeUp .65s ease-out both;
        margin-bottom: 28px;
    }

    .pharma-hero::after, .dashboard-hero::after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -90px;
        top: -130px;
        border-radius: 50%;
        background: rgba(255,255,255,.08);
        pointer-events: none;
    }

    .pharma-hero-title, .dashboard-hero h1 {
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 800;
        line-height: 1.15;
        color: white !important;
        margin: 0 0 12px;
    }

    .pharma-hero-subtitle {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ccfbf1;
    }

    .pharma-hero-text, .dashboard-hero p {
        color: #ecfeff;
        font-size: 1rem;
        line-height: 1.75;
        max-width: 900px;
    }

    .pharma-status {
        display: inline-block;
        padding: 8px 15px;
        margin-top: 20px;
        border-radius: 30px;
        background: rgba(255,255,255,.15);
        border: 1px solid rgba(255,255,255,.25);
        color: white;
        font-size: .85rem;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(15,23,42,.05);
        transition: transform .25s ease,
                    box-shadow .25s ease;
        animation: fadeUp .5s ease-out both;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 30px rgba(15,23,42,.11);
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary);
        font-weight: 800;
    }

    div[data-testid="stPlotlyChart"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 10px;
        box-shadow: 0 6px 20px rgba(15,23,42,.05);
        animation: fadeUp .6s ease-out both;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stExpander"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #ecfeff, #f0fdfa 55%, #eef2ff);
        border-right: 1px solid #dbeafe;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg, #0f766e, #0891b2);
        border: none;
        color: white;
        border-radius: 12px;
        font-weight: 650;
        transition: transform .2s ease,
                    box-shadow .2s ease;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(8,145,178,.2);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    button[data-baseweb="tab"] {
        font-weight: 650;
        border-radius: 10px;
    }

    button[data-baseweb="tab"]:hover {
        background: #ecfeff;
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (max-width: 768px) {
        .block-container { padding-top: 1rem; }
        .pharma-hero, .dashboard-hero {
            padding: 26px 22px;
            border-radius: 18px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title, subtitle, eyebrow="PHARMASIGNAL AI"):
    import html

    st.markdown(
        '<div class="dashboard-hero">'
        f'<p style="font-size:.8rem;letter-spacing:.16em;'
        f'font-weight:700;margin-bottom:10px">'
        f'{html.escape(eyebrow)}</p>'
        f'<h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
