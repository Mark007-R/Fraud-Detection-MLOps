"""Shared visual theme for Streamlit apps — mark.dev paper (or mark.dev after dark) with one project accent.

Copy this file into a repo as ``ui_theme.py`` (next to the app, or inside the
package), set MODE and the three ACCENT values from the palette, and call
``apply_theme()`` right after ``st.set_page_config(...)`` on EVERY page script
(multipage apps run each page on its own, so CSS does not carry over).

Pair it with ``.streamlit/config.toml`` at the repo root so native widgets
(sliders, checkboxes, focus rings) pick up the same colours.

Light (MODE = "light"):
    [theme]
    base = "light"
    primaryColor = "<ACCENT>"
    backgroundColor = "#fefaf5"
    secondaryBackgroundColor = "#faf2e9"
    textColor = "#1c1714"
    font = "sans serif"

Dark (MODE = "dark"):
    [theme]
    base = "dark"
    primaryColor = "<ACCENT>"
    backgroundColor = "#14110f"
    secondaryBackgroundColor = "#1f1a17"
    textColor = "#f4ede4"
    font = "sans serif"
"""
from __future__ import annotations

import streamlit as st

# ---- the only values that change between projects -------------------------
MODE = "dark"              # "light" = mark.dev paper, "dark" = mark.dev after dark
ACCENT = "#F0655A"         # main accent (buttons, links, active tab, metric values)
ACCENT_STRONG = "#F4877E"  # hover / pressed
ACCENT_SOFT = "#D0564C"    # partner for gradients and secondary marks
# ---------------------------------------------------------------------------

_PALETTES = {
    "light": dict(
        paper="#fefaf5", paper_alt="#faf2e9", card="#fffdfa",
        ink="#1c1714", ink_2="#57504a", ink_3="#756c65",
        line="#e9dbcd", line_strong="#d8c3b2", on_accent="#fefaf5",
        header="rgba(254, 250, 245, .82)", placeholder="#a89a8f",
        shadow_sm="0 1px 2px rgba(60, 40, 20, .04), 0 4px 16px rgba(60, 40, 20, .05)",
        shadow_md="0 2px 4px rgba(60, 40, 20, .05), 0 14px 38px rgba(60, 40, 20, .09)",
        neutrals=("#8c7b6b", "#c9a27e", "#3b3230", "#b9ad9f"), dots=1.0,
    ),
    "dark": dict(
        paper="#14110f", paper_alt="#1a1613", card="#1f1a17",
        ink="#f4ede4", ink_2="#c9bcae", ink_3="#9a8c7f",
        line="#2f2823", line_strong="#463b33", on_accent="#14110f",
        header="rgba(20, 17, 15, .82)", placeholder="#7d6f63",
        shadow_sm="0 1px 2px rgba(0, 0, 0, .35), 0 6px 18px rgba(0, 0, 0, .28)",
        shadow_md="0 2px 4px rgba(0, 0, 0, .4), 0 16px 40px rgba(0, 0, 0, .45)",
        neutrals=("#b3a595", "#d9b48f", "#f4ede4", "#7d6f63"), dots=0.75,
    ),
}


def _palette() -> dict:
    return _PALETTES["dark" if MODE == "dark" else "light"]


# module-level names for app code that builds inline HTML (resolved from MODE above)
PAPER = _palette()["paper"]
PAPER_ALT = _palette()["paper_alt"]
CARD = _palette()["card"]
INK = _palette()["ink"]
INK_2 = _palette()["ink_2"]
INK_3 = _palette()["ink_3"]
LINE = _palette()["line"]
LINE_STRONG = _palette()["line_strong"]
ON_ACCENT = _palette()["on_accent"]
FONT_DISPLAY = "'Fraunces', Georgia, 'Times New Roman', serif"
FONT_BODY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, Consolas, monospace"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def colorway() -> list[str]:
    """Categorical chart colours: accent first, then warm neutrals that sit on the ground."""
    n = _palette()["neutrals"]
    return [ACCENT, n[0], ACCENT_SOFT, n[1], n[2], n[3]]


def sequential(n: int = 6) -> list[str]:
    """Ground-to-accent ramp for heatmaps / ordered categories."""
    h = ACCENT.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    g0 = _palette()["paper_alt"].lstrip("#")
    base = tuple(int(g0[i:i + 2], 16) for i in (0, 2, 4))
    out = []
    for i in range(n):
        t = 0.15 + 0.85 * i / max(n - 1, 1)
        out.append("#%02x%02x%02x" % tuple(round(base[k] + (c - base[k]) * t)
                                            for k, c in enumerate((r, g, b))))
    return out


def plotly_layout(**overrides) -> dict:
    """Transparent plotly layout that blends into the ground.

    Render with ``st.plotly_chart(fig, theme=None)`` — Streamlit's default chart
    theme otherwise repaints the traces. Do not add a ``title_font`` without a
    title: Streamlit then prints "undefined". Prefer ``style_fig(fig)``.
    """
    p = _palette()
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=p["ink_2"], size=13),
        colorway=colorway(),
        xaxis=dict(gridcolor=p["line"], linecolor=p["line_strong"], zerolinecolor=p["line"]),
        yaxis=dict(gridcolor=p["line"], linecolor=p["line_strong"], zerolinecolor=p["line"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=p["ink_2"])),
        hoverlabel=dict(bgcolor=p["card"], bordercolor=p["line_strong"], font=dict(color=p["ink"])),
    )
    layout.update(overrides)
    return layout


# plotly's stock palette — traces still wearing these get repainted by style_fig()
_PLOTLY_DEFAULTS = {"#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
                    "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52"}


def style_fig(fig, **layout_overrides):
    """Apply the layout AND recolour traces, then render with theme=None.

    plotly express stamps its own colour onto each trace at creation time, so
    setting ``colorway`` afterwards does nothing — hence the second pass. Even
    better: pass ``color_discrete_sequence=ui_theme.colorway()`` to the px call.
    """
    fig.update_layout(**plotly_layout(**layout_overrides))
    palette, i = colorway(), 0
    for trace in fig.data:
        for holder in ("marker", "line"):
            obj = getattr(trace, holder, None)
            colour = getattr(obj, "color", None) if obj is not None else None
            if isinstance(colour, str) and colour.lower() in _PLOTLY_DEFAULTS:
                obj.color = palette[i % len(palette)]
                i += 1
    return fig


def _dots_svg() -> str:
    fill = ACCENT.replace("#", "%23")
    k = _palette()["dots"]
    dots = [(52, 71, 1.4, .28), (178, 38, .9, .22), (312, 95, 1.6, .3), (458, 52, 1, .24),
            (546, 128, 1.3, .26), (96, 188, 1.1, .25), (236, 222, 1.5, .28), (392, 176, .9, .2),
            (508, 246, 1.4, .27), (28, 298, 1.2, .24), (158, 344, 1.6, .3), (286, 312, 1, .22),
            (430, 366, 1.3, .26), (566, 402, 1.1, .23), (72, 452, 1.5, .29), (204, 486, .9, .21),
            (348, 438, 1.4, .27), (482, 528, 1.2, .25), (128, 566, 1.3, .26), (268, 592, 1, .22),
            (412, 556, 1.5, .28), (588, 486, .9, .2)]
    circles = "".join(f"%3Ccircle cx='{x}' cy='{y}' r='{r}' opacity='{round(o * k, 3)}'/%3E"
                      for x, y, r, o in dots)
    return ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='600'%3E"
            f"%3Cg fill='{fill}'%3E{circles}%3C/g%3E%3C/svg%3E")


def theme_css() -> str:
    p = _palette()
    tint = _rgba(ACCENT, .08 if MODE == "dark" else .06)
    glow = _rgba(ACCENT, .22 if MODE == "dark" else .14)
    line_accent = _rgba(ACCENT, .32 if MODE == "dark" else .24)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500;1,9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --paper: {p["paper"]}; --paper-alt: {p["paper_alt"]}; --card: {p["card"]};
  --ink: {p["ink"]}; --ink-2: {p["ink_2"]}; --ink-3: {p["ink_3"]};
  --line: {p["line"]}; --line-strong: {p["line_strong"]}; --on-accent: {p["on_accent"]};
  --accent: {ACCENT}; --accent-strong: {ACCENT_STRONG}; --accent-soft: {ACCENT_SOFT};
  --accent-tint: {tint}; --accent-glow: {glow}; --accent-line: {line_accent};
  --shadow-sm: {p["shadow_sm"]};
  --shadow-md: {p["shadow_md"]};
}}

/* ground with the mark.dev dot texture */
.stApp {{
  background-color: var(--paper);
  background-image: url("{_dots_svg()}");
  background-repeat: repeat;
  color: var(--ink);
}}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp td, .stApp th, [data-testid="stMarkdownContainer"] {{ font-family: {FONT_BODY}; }}
::selection {{ background: var(--accent); color: var(--on-accent); }}

[data-testid="stHeader"] {{ background: {p["header"]}; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }}
[data-testid="stDecoration"] {{ background-image: linear-gradient(90deg, var(--accent), var(--accent-soft)); height: 2px; }}

[data-testid="stSidebar"] {{ background-color: var(--paper-alt); border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] > div:first-child {{ background: transparent; }}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{ border-radius: 999px; }}
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNavLink"]:hover {{ background: var(--accent-tint); }}
[data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] {{ background: var(--accent-tint); }}
[data-testid="stSidebarNav"] a[aria-current="page"] span, [data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: var(--accent); font-weight: 600; }}

/* type */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: {FONT_DISPLAY}; color: var(--ink); letter-spacing: -.015em; text-wrap: balance; }}
.stApp h1 {{ font-weight: 700; line-height: 1.08; }}
.stApp h2, .stApp h3 {{ font-weight: 600; }}
.stApp h4 {{ font-weight: 600; font-size: 1.1rem; }}
.stApp a {{ color: var(--accent); }}
.stApp a:hover {{ color: var(--accent-strong); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--ink-3); }}
.stApp hr {{ border-color: var(--line); }}
.stApp code {{ font-family: {FONT_MONO}; font-size: .85em; color: var(--accent); background: var(--accent-tint); border-radius: 6px; padding: .1em .35em; }}
.stApp pre code, [data-testid="stCode"] code {{ color: inherit; background: transparent; padding: 0; }}
[data-testid="stCode"] pre, .stCode pre {{ background: var(--paper-alt) !important; border: 1px solid var(--line); border-radius: 10px; }}

/* widget labels read like mark.dev form labels */
[data-testid="stWidgetLabel"] p {{ font-family: {FONT_MONO}; font-size: .7rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); }}
.stCheckbox [data-testid="stWidgetLabel"] p, [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
.stToggle [data-testid="stWidgetLabel"] p, [data-testid="stToggle"] [data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label[data-baseweb="radio"] p {{
  font-family: {FONT_BODY}; font-size: .92rem; font-weight: 500; letter-spacing: 0; text-transform: none; color: var(--ink-2);
}}

/* buttons: pills */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="baseButton-secondary"], [data-testid="stBaseButton-secondary"] {{
  border-radius: 999px; border: 1px solid var(--line); background: var(--card); color: var(--ink-2);
  font-weight: 600; padding: .5rem 1.25rem; box-shadow: var(--shadow-sm);
  transition: background .25s ease, border-color .25s ease, color .25s ease, transform .25s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover,
[data-testid="baseButton-secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {{
  border-color: var(--accent); color: var(--accent); background: var(--accent-tint); transform: translateY(-1px);
}}
.stApp button[kind="primary"], [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] {{
  background: var(--accent); border-color: var(--accent); color: var(--on-accent);
  box-shadow: 0 2px 12px {_rgba(ACCENT, .2)};
}}
.stApp button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  background: var(--accent-strong); border-color: var(--accent-strong); color: var(--on-accent);
}}
.stApp button[kind="primary"] p, [data-testid="baseButton-primary"] p, [data-testid="stBaseButton-primary"] p {{ color: var(--on-accent); }}
.stApp button:focus-visible {{ box-shadow: 0 0 0 3px var(--accent-glow); outline: none; }}

/* inputs */
.stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div {{
  background-color: var(--card); border: 1px solid var(--line); border-radius: 10px;
}}
.stApp [data-baseweb="base-input"], .stApp [data-baseweb="input"] input, .stApp [data-baseweb="textarea"] textarea {{ background-color: var(--card); color: var(--ink); }}
.stApp input::placeholder, .stApp textarea::placeholder {{ color: {p["placeholder"]}; }}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within, .stApp [data-baseweb="select"] > div:focus-within {{
  border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow);
}}
/* number inputs: Streamlit frames the container and zeroes the inner input, so the
   generic input rule above must not re-border the inner one */
[data-testid="stNumberInputContainer"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; }}
[data-testid="stNumberInputContainer"]:focus-within {{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }}
.stApp [data-testid="stNumberInputContainer"] [data-baseweb="input"],
.stApp [data-testid="stNumberInputContainer"] [data-baseweb="input"]:focus-within {{ border: 0; border-radius: 0; box-shadow: none; background: transparent; }}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{ background: transparent; color: var(--ink-3); }}
[data-testid="stNumberInputStepDown"]:hover:enabled, [data-testid="stNumberInputStepUp"]:hover:enabled {{ background: var(--accent-tint); color: var(--accent); }}
.stApp [data-baseweb="tag"] {{ background-color: var(--accent-tint); border: 1px solid var(--accent-line); border-radius: 999px; }}
.stApp [data-baseweb="tag"] span {{ color: var(--accent); }}
[data-testid="stSlider"] [role="slider"] {{ background-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-glow); }}
[data-testid="stThumbValue"], [data-testid="stSliderThumbValue"] {{ color: var(--accent); font-family: {FONT_MONO}; }}

/* metrics as mark.dev stat cards */
[data-testid="stMetric"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: .9rem 1rem; box-shadow: var(--shadow-sm); }}
[data-testid="stMetricLabel"] p {{ font-family: {FONT_MONO}; font-size: .68rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ font-family: {FONT_DISPLAY}; color: var(--accent); font-weight: 600; }}
/* serif digits run wider than Streamlit's default face: scale with the viewport so 5-6 metrics in a row don't ellipsize */
[data-testid="stMetricValue"] {{ font-size: clamp(1.35rem, 1rem + 1.1vw, 2.1rem); }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; }}
.stTabs [data-baseweb="tab"] {{ color: var(--ink-2); font-weight: 500; padding-left: .15rem; padding-right: .15rem; }}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--accent); }}
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color: var(--accent); font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--accent); }}
.stTabs [data-baseweb="tab-border"] {{ background-color: var(--line); }}

/* surfaces */
[data-testid="stExpander"] details {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--shadow-sm); }}
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover p {{ color: var(--accent); }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--card); border: 1.5px dashed var(--line-strong); border-radius: 16px; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--accent); background: var(--accent-tint); }}
[data-testid="stFileUploaderDropzone"] svg {{ color: var(--accent); fill: var(--accent); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--card); }}
[data-testid="stAlert"], [data-testid="stAlert"] > div {{ border-radius: 12px; }}
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {{ border-radius: 12px; }}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--line-strong); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}
</style>
"""


# ===========================================================================
# SENTINEL extensions (Fraud-Detection-MLOps) — status colours, chart scales
# and the app's own components. Everything below resolves from MODE/ACCENT
# above, exactly like the shared half, so nothing here hardcodes light or dark.
# ===========================================================================

_STATUS = {
    "light": dict(ok="#3f7a3a", ok_tint="rgba(63, 122, 58, .10)",
                  warn="#a86a12", warn_tint="rgba(168, 106, 18, .10)",
                  bad="#b3261e", bad_tint="rgba(179, 38, 30, .08)",
                  info="#2f5f8a", info_tint="rgba(47, 95, 138, .09)"),
    "dark": dict(ok="#6fbf73", ok_tint="rgba(111, 191, 115, .12)",
                 warn="#e0a445", warn_tint="rgba(224, 164, 69, .12)",
                 bad="#f07167", bad_tint="rgba(240, 113, 103, .12)",
                 info="#7fb0e0", info_tint="rgba(127, 176, 224, .12)"),
}


def _status() -> dict:
    return _STATUS["dark" if MODE == "dark" else "light"]


# status colours keep their meaning in every project; always shown on their tint
OK = _status()["ok"]
WARN = _status()["warn"]
BAD = _status()["bad"]
INFO = _status()["info"]
NEUTRAL = _palette()["neutrals"][0]   # first warm neutral of the colorway


def colorscale(n: int = 6) -> list[list]:
    """``sequential()`` as a plotly colorscale — low values recede into the ground."""
    steps = sequential(n)
    return [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]


def bar_scale() -> list[list]:
    """Warm neutral -> accent, for bars coloured by value (both ends stay visible)."""
    return [[0.0, NEUTRAL], [1.0, ACCENT]]


def ramp(n: int = 5) -> list[str]:
    """``bar_scale()`` as n discrete colours, for ordered bar categories."""
    a = [int(NEUTRAL.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(ACCENT.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return ["#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * i / max(n - 1, 1)) for k in range(3))
            for i in range(n)]


def diverging() -> list[list]:
    """info -> ground -> accent, for correlation matrices centred on zero."""
    return [[0.0, INFO], [0.5, PAPER_ALT], [1.0, ACCENT]]


def components_css() -> str:
    """SENTINEL's own components, re-expressed in the shared tokens.

    The class names are the ones app.py and pages/*.py already emit in their
    markdown blocks — style hooks only, so they keep their names.
    """
    s = _status()
    return f"""
<style>
:root {{
  --ok: {s["ok"]}; --ok-tint: {s["ok_tint"]};
  --warn: {s["warn"]}; --warn-tint: {s["warn_tint"]};
  --bad: {s["bad"]}; --bad-tint: {s["bad_tint"]};
  --info: {s["info"]}; --info-tint: {s["info_tint"]};
}}

/* ---- headers -------------------------------------------------------------- */
.main-header {{ text-align: center; margin-bottom: 2.25rem; padding: 1.5rem 0 .5rem; }}
.main-header h1 {{
  font-family: {FONT_DISPLAY} !important; font-weight: 700; font-size: clamp(2.4rem, 5vw, 3.2rem) !important;
  line-height: 1.05; letter-spacing: -.025em; color: var(--ink) !important;
  padding: 0 !important; margin-bottom: .5rem;
}}
.main-header p {{ color: var(--ink-2); font-size: 1.02rem; }}

.page-header {{ padding: .5rem 0 1.75rem; }}
.page-header h1 {{
  font-family: {FONT_DISPLAY} !important; font-weight: 700; font-size: clamp(2rem, 4vw, 2.6rem) !important;
  line-height: 1.07; letter-spacing: -.025em; color: var(--ink) !important;
  padding: 0 !important; margin-bottom: .35rem;
}}
.page-header p {{ color: var(--ink-2); }}

.ops-title {{
  text-align: center; font-family: {FONT_DISPLAY}; font-weight: 700;
  font-size: clamp(2rem, 4.4vw, 2.8rem); line-height: 1.06; letter-spacing: -.025em;
  color: var(--ink); margin-bottom: .3rem;
}}
.ops-sub {{ text-align: center; color: var(--ink-2); margin-bottom: 1.75rem; }}

/* ---- KPI tiles (home) ------------------------------------------------------ */
.kpi-card {{
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.35rem 1rem; text-align: center; box-shadow: var(--shadow-sm);
  transition: border-color .3s ease, transform .3s ease, box-shadow .3s ease;
}}
.kpi-card:hover {{ border-color: var(--line-strong); transform: translateY(-3px); box-shadow: var(--shadow-md); }}
.kpi-card .kpi-value {{ font-family: {FONT_DISPLAY}; font-size: 2rem; font-weight: 700; line-height: 1.12; color: var(--accent); }}
.kpi-card .kpi-label {{
  font-family: {FONT_MONO}; font-size: .66rem; font-weight: 500; color: var(--ink-3);
  text-transform: uppercase; letter-spacing: .12em; margin-top: .55rem;
}}
.kpi-card .kpi-delta {{ font-family: {FONT_MONO}; font-size: .7rem; color: var(--ok); margin-top: .3rem; }}

/* ---- metric tiles (ops) ---------------------------------------------------- */
.metric-card {{
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.1rem 1.25rem; box-shadow: var(--shadow-sm);
  transition: border-color .3s ease, transform .3s ease, box-shadow .3s ease;
}}
.metric-card:hover {{ border-color: var(--line-strong); transform: translateY(-3px); box-shadow: var(--shadow-md); }}
.metric-card h4 {{
  font-family: {FONT_MONO} !important; font-size: .66rem !important; font-weight: 600 !important;
  color: var(--ink-3) !important; text-transform: uppercase; letter-spacing: .12em;
  padding: 0 !important; margin: 0 0 .45rem !important;
}}
.metric-card .value {{ font-family: {FONT_DISPLAY}; color: var(--accent); font-size: 1.65rem; font-weight: 700; line-height: 1.15; }}
.metric-card .delta {{ color: var(--ok); font-size: .82rem; margin-top: .3rem; line-height: 1.45; }}
.metric-card .delta.bad {{ color: var(--bad); }}

/* ---- capability cards ------------------------------------------------------ */
.feature-card {{
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.5rem 1.4rem; margin-bottom: 1rem; box-shadow: var(--shadow-sm);
  position: relative; overflow: hidden;
  transition: border-color .3s ease, transform .3s ease, box-shadow .3s ease;
}}
.feature-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--accent); opacity: 0; transition: opacity .3s ease;
}}
.feature-card:hover {{ border-color: var(--line-strong); transform: translateY(-3px); box-shadow: var(--shadow-md); }}
.feature-card:hover::before {{ opacity: 1; }}
.feature-card h4 {{
  font-family: {FONT_DISPLAY} !important; font-size: 1.05rem !important; font-weight: 600 !important;
  color: var(--ink) !important; padding: 0 !important; margin-bottom: .5rem;
}}
.feature-card p {{ color: var(--ink-2) !important; font-size: .92rem; line-height: 1.65; margin: 0; }}

/* ---- prediction verdict callouts -------------------------------------------- */
.result-fraud, .result-safe {{
  border: 1px solid var(--line); border-radius: 16px;
  padding: 1.35rem 1.5rem; margin: 1rem 0; box-shadow: var(--shadow-sm);
}}
.result-fraud {{ background: var(--bad-tint); border-color: {_rgba(s["bad"], .3)}; border-left: 4px solid var(--bad); }}
.result-safe {{ background: var(--ok-tint); border-color: {_rgba(s["ok"], .3)}; border-left: 4px solid var(--ok); }}
.result-fraud h3, .result-safe h3 {{
  font-family: {FONT_MONO} !important; font-size: .72rem !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: .14em; padding: 0 !important; margin: 0 0 .55rem !important;
}}
.result-fraud h3 {{ color: var(--bad) !important; }}
.result-safe h3 {{ color: var(--ok) !important; }}
.result-fraud p, .result-safe p {{ color: var(--ink) !important; margin: 0; font-weight: 500; line-height: 1.6; }}

/* ---- status pill ("System Online") ------------------------------------------ */
.status-badge {{
  display: inline-flex; align-items: center; gap: 8px;
  font-family: {FONT_MONO}; font-size: .66rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .13em;
  padding: 6px 14px; border-radius: 999px;
}}
.status-active {{ background: var(--ok-tint); color: var(--ok); border: 1px solid {_rgba(s["ok"], .3)}; }}
.status-active::before {{
  content: ''; width: 7px; height: 7px; border-radius: 50%;
  background: var(--ok); box-shadow: 0 0 0 3px {_rgba(s["ok"], .2)};
}}

/* ---- sidebar brand + meta blocks -------------------------------------------- */
.brand-block {{ text-align: center; padding: 1.25rem .5rem .25rem; }}
.brand-mark {{
  font-family: {FONT_DISPLAY}; font-style: italic; font-weight: 700;
  font-size: 2.2rem; line-height: 1; color: var(--accent); margin-bottom: .45rem;
}}
.brand-block h2 {{
  font-family: {FONT_DISPLAY} !important; font-weight: 700 !important; font-size: 1.25rem !important;
  letter-spacing: .04em; color: var(--ink) !important; padding: 0 !important; margin: 0 !important;
}}
.brand-block p {{
  font-family: {FONT_MONO}; font-size: .6rem; font-weight: 500; color: var(--ink-3) !important;
  text-transform: uppercase; letter-spacing: .16em; margin: .4rem 0 0 !important;
}}
.side-block {{ padding: 0 .5rem; }}
.side-block .side-label {{
  font-family: {FONT_MONO}; font-size: .62rem; font-weight: 600; color: var(--accent);
  text-transform: uppercase; letter-spacing: .14em; margin-bottom: .6rem;
}}
.side-block .side-list {{ color: var(--ink-2); font-size: .84rem; line-height: 1.95; margin: 0; }}

/* ---- markdown tables: mono column heads on the line colour ------------------- */
[data-testid="stMarkdownContainer"] table {{ border-collapse: collapse; }}
[data-testid="stMarkdownContainer"] th {{
  font-family: {FONT_MONO}; font-size: .66rem; font-weight: 600; color: var(--ink-3);
  text-transform: uppercase; letter-spacing: .12em; background: var(--paper-alt);
}}
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {{ border-color: var(--line); }}
[data-testid="stMarkdownContainer"] td {{ color: var(--ink-2); }}
[data-testid="stMarkdownContainer"] td strong {{ color: var(--ink); }}

/* ---- rules and footers ------------------------------------------------------ */
.section-divider {{
  height: 1px; border: 0; margin: 2rem 0;
  background: linear-gradient(90deg, transparent, var(--line-strong), transparent);
}}
.footer-text {{
  text-align: center; border-top: 1px solid var(--line); padding: 1.5rem 0 .5rem;
  font-family: {FONT_MONO}; font-size: .64rem; font-weight: 500; color: var(--ink-3);
  text-transform: uppercase; letter-spacing: .12em;
}}
</style>
"""


def apply_theme() -> None:
    st.markdown(theme_css(), unsafe_allow_html=True)
    st.markdown(components_css(), unsafe_allow_html=True)  # SENTINEL components


def eyebrow(text: str) -> None:
    """Small mono uppercase label above a heading, like mark.dev's status badge."""
    st.markdown(
        f"<div style=\"font-family:{FONT_MONO};font-size:.68rem;font-weight:600;letter-spacing:.14em;"
        f"text-transform:uppercase;color:{ACCENT};margin-bottom:.35rem\">{text}</div>",
        unsafe_allow_html=True,
    )
