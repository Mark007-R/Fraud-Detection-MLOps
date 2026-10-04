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
        xaxis=dict(gridcolor=p["line"], linecolor=p["line_strong"], zerolinecolor=p["line"], automargin=True),
        yaxis=dict(gridcolor=p["line"], linecolor=p["line_strong"], zerolinecolor=p["line"], automargin=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=p["ink_2"])),
        hoverlabel=dict(bgcolor=p["card"], bordercolor=p["line_strong"], font=dict(color=p["ink"])),
    )
    for key, value in overrides.items():
        # axis overrides add to the themed axis instead of replacing it
        if key in ("xaxis", "yaxis") and isinstance(value, dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
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


def bar_scale() -> list[list]:
    """Warm neutral -> accent, for bars coloured by value (both ends stay visible)."""
    return [[0.0, NEUTRAL], [1.0, ACCENT]]


def components_css() -> str:
    """SENTINEL's own components, built on the shared tokens.

    Class names are the ones ui_kit.py and views/*.py emit. Layout rules for
    Streamlit's top navigation and bordered containers live here too, since
    the shared half above targets the sidebar layout.
    """
    s = _status()
    tint = _rgba(ACCENT, .12)
    return f"""
<style>
:root {{
  --ok: {s["ok"]}; --ok-tint: {s["ok_tint"]};
  --warn: {s["warn"]}; --warn-tint: {s["warn_tint"]};
  --bad: {s["bad"]}; --bad-tint: {s["bad_tint"]};
  --info: {s["info"]}; --info-tint: {s["info_tint"]};
  --neutral: {NEUTRAL};
  --display: {FONT_DISPLAY}; --mono: {FONT_MONO};
}}

/* ---- shell: top navigation + page width ---------------------------------- */
[data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-top: 5.25rem; padding-bottom: 3rem; }}
[data-testid="stHeader"] {{ border-bottom: 1px solid var(--line); }}
[data-testid="stHeaderLogo"], [data-testid="stLogo"] {{ height: 2rem; max-width: 11rem; }}
[data-testid="stTopNavLink"] {{ border-radius: 999px; padding-left: .8rem; padding-right: .8rem; transition: background .2s ease; }}
[data-testid="stTopNavLink"]:hover {{ background: var(--accent-tint); }}
[data-testid="stTopNavLink"]:hover span {{ color: var(--accent) !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background: var(--accent-tint); box-shadow: inset 0 0 0 1px var(--accent-line); }}
[data-testid="stTopNavLink"][aria-current="page"] span {{ color: var(--accent) !important; font-weight: 600; }}

/* hero calls to action: page links as pills, the first one filled */
.st-key-hero_cta {{ gap: .7rem; }}
.st-key-hero_cta [data-testid="stPageLink-NavLink"] {{
  border: 1px solid var(--line-strong); border-radius: 999px; padding: .55rem 1.15rem; background: var(--card);
  transition: border-color .25s ease, background .25s ease, transform .25s ease;
}}
.st-key-hero_cta [data-testid="stPageLink-NavLink"]:hover {{ border-color: var(--accent); background: var(--accent-tint); transform: translateY(-1px); }}
.st-key-hero_cta [data-testid="stPageLink-NavLink"] p {{ font-weight: 600; }}
.st-key-hero_cta [data-testid="stElementContainer"]:first-child [data-testid="stPageLink-NavLink"] {{
  background: var(--accent); border-color: var(--accent); box-shadow: 0 4px 18px {_rgba(ACCENT, .25)};
}}
.st-key-hero_cta [data-testid="stElementContainer"]:first-child [data-testid="stPageLink-NavLink"] span,
.st-key-hero_cta [data-testid="stElementContainer"]:first-child [data-testid="stPageLink-NavLink"] p {{ color: var(--on-accent) !important; }}
.st-key-hero_cta [data-testid="stElementContainer"]:first-child [data-testid="stPageLink-NavLink"]:hover {{ background: var(--accent-strong); }}

/* bordered containers read as cards */
.stApp [class*="st-key-card_"] {{
  background: var(--card); border-color: var(--line); border-radius: 18px; box-shadow: var(--shadow-sm);
  padding: 1.1rem 1.25rem 1rem;
}}

/* ---- shared text pieces ---------------------------------------------------- */
.eyebrow {{
  display: inline-flex; align-items: center; gap: .55rem;
  font-family: var(--mono); font-size: .68rem; font-weight: 600; letter-spacing: .14em;
  text-transform: uppercase; color: var(--accent);
}}
.pulse {{ width: 8px; height: 8px; border-radius: 50%; background: var(--accent); position: relative; }}
.pulse::after {{
  content: ''; position: absolute; inset: -4px; border-radius: 50%; border: 1px solid var(--accent);
  animation: sentinel-pulse 2.4s ease-out infinite;
}}
@keyframes sentinel-pulse {{ 0% {{ transform: scale(.6); opacity: .9; }} 100% {{ transform: scale(2.2); opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .pulse::after {{ animation: none; }} }}
.stApp .fine {{ color: var(--ink-3); font-size: .84rem; line-height: 1.6; margin: .6rem 0 0; }}
.stApp .fine b {{ color: var(--ink-2); font-weight: 600; }}
.stApp .prose p {{ color: var(--ink-2); font-size: 1rem; line-height: 1.75; margin: 0 0 1rem; }}
.prose b {{ color: var(--ink); font-weight: 600; }}
.stApp em {{ font-style: italic; color: var(--accent); }}

/* ---- hero (overview) -------------------------------------------------------- */
.hero {{ padding: .5rem 0 .25rem; }}
.hero-title {{
  font-family: var(--display) !important; font-weight: 600 !important; font-size: clamp(2.6rem, 5.4vw, 4.2rem) !important;
  line-height: 1.02 !important; letter-spacing: -.035em; color: var(--ink) !important; margin: 1rem 0 1.1rem !important; padding: 0 !important;
}}
.hero-title em {{ font-weight: 500; }}
.stApp .hero-lede {{ color: var(--ink-2); font-size: 1.08rem; line-height: 1.7; max-width: 38rem; margin: 0 0 1.4rem; }}

.live-card {{
  background: linear-gradient(165deg, {tint}, var(--card) 55%); border: 1px solid var(--line);
  border-radius: 20px; padding: 1.2rem 1.3rem .6rem; box-shadow: var(--shadow-md);
}}
.live-head {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: .3rem 1rem; margin-bottom: .6rem; }}
.live-head .eyebrow {{ white-space: nowrap; }}
.live-tag {{ font-family: var(--mono); font-size: .62rem; letter-spacing: .08em; color: var(--ink-3); text-transform: uppercase; }}
.live-row {{
  display: grid; grid-template-columns: 1fr auto; gap: .2rem 1rem; align-items: center;
  padding: .85rem 0 .9rem; border-top: 1px solid var(--line);
}}
.live-name {{ font-family: var(--display); font-weight: 600; font-size: 1.05rem; color: var(--ink); }}
.live-desc {{ color: var(--ink-3); font-size: .8rem; margin-top: .1rem; }}
.live-score {{ font-family: var(--display); font-weight: 600; font-size: 1.5rem; text-align: right; line-height: 1; }}
.live-score span {{ display: block; font-family: var(--mono); font-size: .58rem; letter-spacing: .14em; text-transform: uppercase; margin-top: .3rem; }}
.live-row.is-flag .live-score {{ color: var(--accent); }}
.live-row.is-clear .live-score {{ color: var(--ok); }}
.live-bar {{ grid-column: 1 / -1; height: 4px; border-radius: 99px; background: var(--line); overflow: hidden; margin-top: .45rem; }}
.live-bar i {{ display: block; height: 100%; border-radius: 99px; background: var(--neutral); }}
.live-row.is-flag .live-bar i {{ background: linear-gradient(90deg, var(--accent-soft), var(--accent)); }}
.live-row.is-clear .live-bar i {{ background: var(--ok); }}

/* ---- page + section headers ---------------------------------------------------- */
.ph {{ margin: .25rem 0 1.6rem; }}
.ph-title {{
  font-family: var(--display) !important; font-weight: 600 !important; font-size: clamp(2.2rem, 4.2vw, 3.1rem) !important;
  line-height: 1.05 !important; letter-spacing: -.03em; color: var(--ink) !important; margin: .7rem 0 .55rem !important; padding: 0 !important;
}}
.ph-title em {{ font-weight: 500; }}
.stApp .ph-lede {{ color: var(--ink-2); font-size: 1.02rem; line-height: 1.65; max-width: 46rem; margin: 0; }}

.sec {{ margin: 3.2rem 0 1.1rem; }}
.sec-index {{
  display: flex; align-items: center; gap: .8rem; font-family: var(--mono); font-size: .68rem; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase; color: var(--accent);
}}
.sec-index::after {{ content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--line-strong), transparent); }}
.sec-title {{
  font-family: var(--display) !important; font-weight: 600 !important; font-size: clamp(1.5rem, 2.6vw, 1.95rem) !important;
  letter-spacing: -.02em; color: var(--ink) !important; margin: .55rem 0 .35rem !important; padding: 0 !important;
}}
.stApp .sec-sub {{ color: var(--ink-2); font-size: .95rem; line-height: 1.6; max-width: 52rem; margin: 0; }}

.card-head {{ margin: .1rem 0 .5rem; }}
.card-title {{ font-family: var(--display); font-weight: 600; font-size: 1.12rem; color: var(--ink); margin-top: .3rem; letter-spacing: -.01em; }}

/* ---- stat tiles ---------------------------------------------------------------------- */
.stat-grid {{ display: grid; grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr)); gap: 14px; margin: 1.6rem 0 1.1rem; }}
.stat {{
  position: relative; overflow: hidden; background: var(--card); border: 1px solid var(--line);
  border-radius: 16px; padding: 1.05rem 1.15rem 1.1rem; box-shadow: var(--shadow-sm);
  transition: border-color .3s ease, transform .3s ease;
}}
.stat::before {{
  content: ''; position: absolute; top: 0; left: 0; width: 46%; height: 2px;
  background: linear-gradient(90deg, var(--accent), transparent);
}}
.stat-ink::before {{ background: linear-gradient(90deg, var(--line-strong), transparent); }}
.stat:hover {{ border-color: var(--line-strong); transform: translateY(-2px); }}
.stat-label {{ font-family: var(--mono); font-size: .64rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); }}
.stat-value {{
  font-family: var(--display); font-weight: 600; font-size: clamp(1.7rem, 1.2rem + 1.2vw, 2.3rem);
  line-height: 1.1; color: var(--accent); margin-top: .45rem; letter-spacing: -.02em; font-variant-numeric: tabular-nums;
}}
.stat-ink .stat-value {{ color: var(--ink); }}
.stat-ok .stat-value {{ color: var(--ok); }}
.stat-unit {{ font-family: var(--mono); font-size: .78rem; font-weight: 500; color: var(--ink-3); margin-left: .35rem; letter-spacing: .02em; }}
.stat-foot {{ color: var(--ink-2); font-size: .82rem; line-height: 1.5; margin-top: .4rem; }}

/* ---- callouts ------------------------------------------------------------------------- */
.callout {{
  background: var(--paper-alt); border: 1px solid var(--line); border-left: 3px solid var(--line-strong);
  border-radius: 12px; padding: .9rem 1.1rem; margin: .6rem 0 1rem;
}}
.callout-title {{ font-family: var(--mono); font-size: .64rem; font-weight: 600; letter-spacing: .13em; text-transform: uppercase; color: var(--accent); margin-bottom: .35rem; }}
.callout-body {{ color: var(--ink-2); font-size: .92rem; line-height: 1.65; }}
.callout-body b {{ color: var(--ink); font-weight: 600; }}
.callout-info {{ border-left-color: var(--accent); }}
.callout-warn {{ border-left-color: var(--warn); background: var(--warn-tint); }}
.callout-bad {{ border-left-color: var(--bad); background: var(--bad-tint); }}

/* ---- the loop (overview) ------------------------------------------------------------------ */
.flow {{ list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
.stApp .flow > li.flow-step {{
  position: relative; background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 1rem 1.1rem 1.1rem; margin: 0; box-shadow: var(--shadow-sm);
}}
.flow-step::after {{
  content: '→'; position: absolute; right: -13px; top: 50%; transform: translateY(-50%);
  font-family: var(--mono); font-size: .8rem; color: var(--line-strong); z-index: 1;
}}
.flow-step:nth-child(4n)::after, .flow-step:last-child::after {{ content: none; }}
.flow-step.is-loop {{ border-color: var(--accent-line); background: linear-gradient(160deg, {tint}, var(--card) 70%); }}
.flow-n {{ font-family: var(--mono); font-size: .66rem; font-weight: 600; color: var(--accent); letter-spacing: .1em; }}
.flow-t {{ display: block; font-family: var(--display); font-weight: 600; font-size: 1.08rem; color: var(--ink); margin: .35rem 0 .3rem; }}
.flow-d {{ display: block; color: var(--ink-2); font-size: .85rem; line-height: 1.55; }}
.flow-note {{ font-family: var(--mono); font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); margin-top: .8rem; }}

/* ---- specialist vs LLM (overview) ------------------------------------------------------------ */
.vs {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm); }}
.vs-row {{
  display: grid; grid-template-columns: 1.25fr 2fr .8fr .9fr; gap: 1.2rem; align-items: center;
  padding: 1rem 1.3rem; border-top: 1px solid var(--line);
}}
.vs-row:first-child {{ border-top: 0; }}
.vs-row.is-champ {{ background: {_rgba(ACCENT, .07)}; }}
.vs-name {{ font-family: var(--display); font-weight: 600; font-size: 1.02rem; color: var(--ink); }}
.vs-row.is-champ .vs-name {{ color: var(--accent); }}
.vs-metric {{ display: flex; align-items: center; gap: .8rem; }}
.vs-metric span {{ font-family: var(--mono); font-size: .72rem; color: var(--ink-2); white-space: nowrap; }}
.vs-bar {{ flex: 1; height: 8px; border-radius: 99px; background: var(--line); overflow: hidden; }}
.vs-bar i {{ display: block; height: 100%; border-radius: 99px; background: var(--neutral); }}
.vs-row.is-champ .vs-bar i {{ background: linear-gradient(90deg, var(--accent-soft), var(--accent)); }}
.vs-num b {{ display: block; font-family: var(--display); font-weight: 600; font-size: 1.1rem; color: var(--ink); font-variant-numeric: tabular-nums; }}
.vs-num span {{ font-family: var(--mono); font-size: .58rem; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); }}

/* ---- chips + footer -------------------------------------------------------------------------- */
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{
  font-family: var(--mono); font-size: .7rem; font-weight: 500; color: var(--ink-2); background: var(--card);
  border: 1px solid var(--line); border-radius: 999px; padding: 6px 13px;
}}
.pg-foot {{
  display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;
  border-top: 1px solid var(--line); margin-top: 3.5rem; padding: 1.2rem 0 .3rem;
  font-family: var(--mono); font-size: .62rem; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3);
}}
.pg-foot code {{ font-size: .95em; }}
.pg-foot a {{ color: var(--accent) !important; text-decoration: none; }}
.pg-foot a:hover {{ color: var(--accent-strong) !important; }}

/* ---- verdict (score) ------------------------------------------------------------------------ */
.verdict {{ border: 1px solid var(--line); border-radius: 20px; padding: 1.35rem 1.5rem 1.25rem; box-shadow: var(--shadow-md); }}
.verdict.is-flag {{ background: linear-gradient(160deg, {_rgba(ACCENT, .18)}, var(--card) 62%); border-color: var(--accent-line); }}
.verdict.is-clear {{ background: linear-gradient(160deg, {_rgba(s["ok"], .14)}, var(--card) 62%); border-color: {_rgba(s["ok"], .3)}; }}
.v-top {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; }}
.v-pill {{
  font-family: var(--mono); font-size: .62rem; font-weight: 600; letter-spacing: .13em; text-transform: uppercase;
  padding: 5px 12px; border-radius: 999px;
}}
.is-flag .v-pill {{ background: var(--accent); color: var(--on-accent); }}
.is-clear .v-pill {{ background: var(--ok-tint); color: var(--ok); border: 1px solid {_rgba(s["ok"], .35)}; }}
.v-prob {{
  font-family: var(--display); font-weight: 600; font-size: clamp(3.6rem, 7vw, 5.4rem); line-height: 1;
  letter-spacing: -.04em; margin: .9rem 0 1.1rem; font-variant-numeric: tabular-nums;
}}
.is-flag .v-prob {{ color: var(--accent); }}
.is-clear .v-prob {{ color: var(--ok); }}
.v-prob span {{ font-size: .42em; color: var(--ink-3); margin-left: .15rem; letter-spacing: 0; }}
.meter {{ position: relative; height: 10px; border-radius: 99px; background: var(--line); }}
.meter i {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; transition: width .4s ease; }}
.is-flag .meter i {{ background: linear-gradient(90deg, var(--neutral), var(--accent)); }}
.is-clear .meter i {{ background: var(--ok); }}
.meter b {{ position: absolute; top: -6px; width: 2px; height: 22px; margin-left: -1px; background: var(--ink); border-radius: 2px; }}
.v-scale {{ display: flex; justify-content: space-between; font-family: var(--mono); font-size: .6rem; color: var(--ink-3); margin-top: .55rem; letter-spacing: .04em; }}
.stApp .v-note {{ color: var(--ink-2); font-size: .9rem; line-height: 1.6; margin: 1rem 0 0; }}

.signals {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.3rem 1rem; margin-top: 1rem; box-shadow: var(--shadow-sm); }}
.signals dl {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem 1.2rem; margin: .8rem 0 0; }}
.signals dt {{ font-family: var(--mono); font-size: .6rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); }}
.signals dd {{ font-family: var(--display); font-weight: 600; font-size: 1.22rem; color: var(--ink); margin: .2rem 0 .1rem; font-variant-numeric: tabular-nums; }}
.signals small {{ color: var(--ink-3); font-size: .74rem; }}

/* ---- confusion matrix (model) ------------------------------------------------------------------ */
.cm {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 8px; align-items: stretch; margin-top: .4rem; }}
.cm-axis {{ grid-column: 1 / -1; text-align: right; font-family: var(--mono); font-size: .58rem; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); }}
.cm-head {{ font-family: var(--mono); font-size: .6rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); text-align: center; }}
.cm-side {{ display: flex; align-items: center; justify-content: flex-end; padding-right: .3rem; font-family: var(--mono); font-size: .6rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); }}
.cm-cell {{ border: 1px solid var(--line); border-radius: 14px; padding: .85rem .95rem; background: var(--paper-alt); }}
.cm-cell b {{ display: block; font-family: var(--display); font-weight: 600; font-size: 1.45rem; color: var(--ink); font-variant-numeric: tabular-nums; }}
.cm-cell span {{ display: block; color: var(--ink); font-size: .82rem; font-weight: 500; margin-top: .15rem; }}
.cm-cell small {{ color: var(--ink-3); font-size: .74rem; }}
.cm-tp {{ background: var(--ok-tint); border-color: {_rgba(s["ok"], .3)}; }}
.cm-tp b {{ color: var(--ok); }}
.cm-fn {{ background: var(--bad-tint); border-color: {_rgba(s["bad"], .3)}; }}
.cm-fn b {{ color: var(--bad); }}
.cm-fp {{ background: var(--warn-tint); border-color: {_rgba(s["warn"], .28)}; }}
.cm-fp b {{ color: var(--warn); }}

/* ---- drift days, retrain events, scoreboard (ops) ------------------------------------------------ */
.days {{ display: grid; grid-template-columns: repeat(30, minmax(0, 1fr)); gap: 4px; margin: .4rem 0 0; }}
.day {{ height: 40px; border-radius: 8px; display: flex; align-items: flex-end; justify-content: center; padding-bottom: 5px; border: 1px solid var(--line); background: var(--paper-alt); }}
.day span {{ font-family: var(--mono); font-size: .58rem; color: var(--ink-3); }}
.day-hit {{ background: var(--accent); border-color: var(--accent); }}
.day-hit span {{ color: var(--on-accent); font-weight: 600; }}
.day-miss {{ background: var(--bad-tint); border: 1px dashed var(--bad); }}
.day-false {{ background: var(--warn-tint); border-color: var(--warn); }}
.days-legend {{ display: flex; flex-wrap: wrap; gap: .5rem 1.3rem; margin: .7rem 0 1.1rem; font-family: var(--mono); font-size: .6rem; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }}
.days-legend span {{ display: inline-flex; align-items: center; gap: .45rem; }}
.days-legend i {{ width: 11px; height: 11px; border-radius: 3px; border: 1px solid var(--line); background: var(--paper-alt); }}
.days-legend i.day-hit {{ background: var(--accent); border-color: var(--accent); }}
.days-legend i.day-miss {{ background: var(--bad-tint); border: 1px dashed var(--bad); }}
.days-legend i.day-false {{ background: var(--warn-tint); border-color: var(--warn); }}

.events {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 1rem; }}
.event {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.15rem .9rem; box-shadow: var(--shadow-sm); }}
.event-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: .8rem; }}
.event-pill {{ font-family: var(--mono); font-size: .6rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; padding: 4px 10px; border-radius: 999px; }}
.event-pill.ok {{ background: var(--ok-tint); color: var(--ok); border: 1px solid {_rgba(s["ok"], .3)}; }}
.event-pill.bad {{ background: var(--bad-tint); color: var(--bad); border: 1px solid {_rgba(s["bad"], .3)}; }}
.event-row {{ display: grid; grid-template-columns: 3.4rem 1fr 3.2rem; gap: .7rem; align-items: center; margin: .35rem 0; }}
.event-row span {{ font-family: var(--mono); font-size: .6rem; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); }}
.event-row b {{ font-family: var(--mono); font-size: .8rem; font-weight: 600; color: var(--ink); text-align: right; }}
.event-bar {{ height: 8px; border-radius: 99px; background: var(--line); overflow: hidden; }}
.event-bar i {{ display: block; height: 100%; border-radius: 99px; background: var(--neutral); }}
.event-bar.is-new i {{ background: linear-gradient(90deg, var(--accent-soft), var(--accent)); }}
.event-foot {{ display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; border-top: 1px solid var(--line); margin-top: .8rem; padding-top: .7rem; }}
.event-foot span {{ color: var(--ink-3); font-size: .76rem; }}
.event-foot b {{ font-family: var(--display); font-weight: 600; font-size: 1.2rem; color: var(--accent); white-space: nowrap; }}

.board {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--card); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; box-shadow: var(--shadow-sm); }}
.board th {{ font-family: var(--mono); font-size: .6rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); background: var(--paper-alt); padding: .75rem .95rem; text-align: left; border: 0; }}
.board td {{ padding: .8rem .95rem; border: 0; border-top: 1px solid var(--line); color: var(--ink-2); font-size: .88rem; line-height: 1.45; vertical-align: top; }}
.board .num {{ text-align: right; font-family: var(--mono); font-size: .8rem; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.board td.delta {{ color: var(--ok); }}
.board .layer {{ display: inline-block; font-family: var(--mono); font-size: .6rem; font-weight: 600; color: var(--accent); background: var(--accent-tint); border: 1px solid var(--accent-line); border-radius: 6px; padding: 1px 6px; margin-right: .55rem; }}
.board tr.is-champ td {{ color: var(--ink); background: {_rgba(ACCENT, .07)}; }}
.board tr.is-champ td:first-child {{ color: var(--accent); font-weight: 600; }}

/* ---- responsive --------------------------------------------------------------------------------- */
@media (max-width: 900px) {{
  .stat-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .flow {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .flow-step::after {{ content: none; }}
  .events {{ grid-template-columns: 1fr; }}
  .days {{ grid-template-columns: repeat(15, minmax(0, 1fr)); }}
  .vs-row {{ grid-template-columns: 1fr 1fr; }}
  .vs-metric {{ grid-column: 1 / -1; order: 3; }}
}}
@media (max-width: 520px) {{
  .stat-grid, .flow {{ grid-template-columns: 1fr; }}
  .days {{ grid-template-columns: repeat(10, minmax(0, 1fr)); }}
  .signals dl {{ grid-template-columns: 1fr; }}
  [data-testid="stMainBlockContainer"] {{ padding-left: 1rem; padding-right: 1rem; }}
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
