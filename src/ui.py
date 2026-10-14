"""DataPeek 디자인 시스템: CSS + Plotly 공통 테마 + HTML 카드 컴포넌트.

데이터 분석 로직과 다크 테마 화면 스타일을 분리합니다.
"""
from html import escape

import streamlit as st


DARK = {
    "background": "#080F20", "surface": "#111C32", "surface2": "#0D182C",
    "border": "#223453", "text": "#F3F6FF", "muted": "#9AABCC",
    "grid": "rgba(150,171,206,.12)",
}

CHART_COLORS = ["#6579FF", "#BA65FF", "#2DD6C7", "#FF6A9D", "#F7B955", "#50A8F9", "#A2D87A"]


def set_design():
    """Streamlit 위젯, 사이드바, 그래프 컨테이너에 사용할 스타일."""
    mode = DARK
    bg, surface, border, text, muted = [mode[k] for k in ("background", "surface", "border", "text", "muted")]
    st.markdown(f"""<style>
    :root {{ --dp-bg:{bg}; --dp-card:{surface}; --dp-border:{border}; --dp-text:{text}; --dp-muted:{muted}; --text-color:{text}; --background-color:{bg}; --secondary-background-color:{surface}; }}
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{background:{bg}!important;color:{text}!important;}}
    /* 전체 앱은 다크 테마만 사용합니다. */
    [data-testid="stAppViewContainer"] :is(h1,h2,h3,h4,h5,h6,p,li,label),
    [data-testid="stSidebar"] :is(h1,h2,h3,h4,h5,h6,p,li,label) {{color:{text};}}
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{color:{muted}!important;}}
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {{color:{text}!important;font-size:1.01rem!important;}}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p {{font-size:1.0rem;line-height:1.75;}}
    [data-testid="stAppViewContainer"] .stTextInput input,
    [data-testid="stAppViewContainer"] .stNumberInput input,
    [data-testid="stAppViewContainer"] .stTextArea textarea,
    [data-testid="stSidebar"] input,
    [data-testid="stAppViewContainer"] input {{color:{text}!important;caret-color:{text}!important;}}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="popover"] [role="listbox"] {{color:{text}!important;background:{surface}!important;border-color:{border}!important;}}
    [data-baseweb="select"] span, [data-baseweb="select"] div[role="button"],
    [data-baseweb="popover"] li {{color:{text}!important;}}
    [data-testid="stFileUploaderDropzone"] {{background:{surface}!important;color:{text}!important;}}
    [data-testid="stFileUploaderDropzone"] * {{color:{text}!important;}}
    [data-testid="stToggle"] label p, [data-testid="stCheckbox"] label p,
    [data-testid="stRadio"] label p, [data-testid="stSelectbox"] label p {{color:{text}!important;}}
    [data-testid="stAppViewContainer"] [data-testid="stExpander"] {{color:{text}!important;}}
    [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary {{color:{text}!important;}}
    [data-testid="stAppViewContainer"] [data-testid="stAlert"] p {{color:inherit!important;}}
    /* CSS 변수는 Streamlit 테마와 HTML 컴포넌트 양쪽에서 재사용합니다. */
    .stApp {{--dp-text:{text};--dp-muted:{muted};--dp-border:{border};--dp-card:{surface};}}
    [data-testid="stHeader"] {{background:transparent;}}
    [data-testid="stAppViewBlockContainer"] {{padding-top:1.05rem;padding-bottom:4rem;max-width:1560px;padding-left:2.0rem;padding-right:2.0rem;}}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{background:{mode['surface2']}!important;color:{text}!important;border-right:1px solid {border};}}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap:.75rem;}}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{color:{text}!important;font-size:.96rem!important;}}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {{color:{text}!important;}}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] strong {{color:#FFFFFF!important;}}
    [data-testid="stSidebar"] [role="radiogroup"] label > span:first-child {{display:none!important;}}
    [data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {{flex:1;}}
    [data-testid="stSidebar"] [role="radiogroup"] {{gap:.26rem;}}
    [data-testid="stSidebar"] [role="radiogroup"] label {{border-radius:10px;padding:.58rem .65rem!important;border:1px solid transparent;transition:background .15s;}}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{background:rgba(126,110,255,.12);}}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{background:linear-gradient(90deg,#3866E8,#513AC4);border-color:rgba(171,164,255,.32);box-shadow:0 5px 18px rgba(76,79,220,.2);}}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{color:#fff!important;font-weight:700;}}
    [data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] p {{font-size:1.0rem!important;margin:0;line-height:1.45;}}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {{background:transparent;}}
    [data-testid="stFileUploaderDropzone"] {{border:1px dashed {border};background:{surface};border-radius:11px;}}
    html {{font-size:16px;}}
    h1,h2,h3,h4,p,li,label {{font-family:'Pretendard','Noto Sans KR','Segoe UI',sans-serif;}}
    h1,h2,h3 {{letter-spacing:-.025em;}}
    h1 {{font-size:2.05rem!important;}}
    h2 {{font-size:1.7rem!important;}}
    h3 {{font-size:1.32rem!important;}}
    [data-testid="stMetric"] {{background:{surface};border:1px solid {border};border-radius:13px;padding:17px 18px;box-shadow:0 10px 30px rgba(0,0,0,.07);}}
    [data-testid="stMetricLabel"] {{color:{muted};}}
    [data-testid="stMetricValue"] {{color:{text};font-weight:760;}}
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div[data-testid="stVerticalBlock"]) {{border-radius:13px;}}
    [data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {{background:{surface};}}
    [data-testid="stVerticalBlockBorderWrapper"] {{border-color:{border}!important;}}
    [data-testid="stPlotlyChart"] {{border:0!important;}}
    [data-testid="stDataFrame"] {{border:1px solid {border};border-radius:10px;overflow:hidden;}}
    [data-testid="stAlert"] {{border-radius:12px;}}
    [data-testid="stExpander"] {{border-radius:11px;border:1px solid {border};background:{surface};}}
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{background:linear-gradient(100deg,#7545F8,#5368F5);border:0;color:white;border-radius:10px;font-weight:700;box-shadow:0 6px 18px rgba(100,87,244,.25);}}
    .stButton > button:hover, .stDownloadButton > button:hover {{border-color:#8577FC!important;}}
    [data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stNumberInput"] {{border-radius:10px;}}
    .dp-brand {{display:flex;align-items:center;gap:10px;padding:14px 3px 12px;}}
    .dp-logo {{height:32px;width:32px;border-radius:10px;background:linear-gradient(145deg,#287FFA,#9B45F8);display:grid;place-items:center;color:white;font-size:20px;font-weight:900;box-shadow:0 4px 16px rgba(91,93,250,.35);}}
    .dp-brand-name {{font-size:1.35rem;letter-spacing:-.035em;font-weight:820;color:{text};line-height:1.0;}}
    .dp-brand-sub {{color:{muted};font-size:.68rem;margin-top:5px;letter-spacing:.07em;}}
    .dp-side-heading {{font-size:.78rem;letter-spacing:.16em;color:{muted};margin-top:.5rem;font-weight:800;}}
    .dp-hero {{position:relative;overflow:hidden;isolation:isolate;border:1px solid #283C77;border-radius:17px;margin-bottom:15px;background:radial-gradient(ellipse 55% 100% at 85% 95%,rgba(95,55,244,.38),transparent 80%),radial-gradient(ellipse 65% 125% at 65% -15%,rgba(34,72,176,.42),transparent 70%),linear-gradient(110deg,#101D3A,#0A1431 55%,#171A59);padding:24px 30px;min-height:142px;}}
    .dp-hero:before {{content:'';position:absolute;width:55%;height:170px;right:-20px;bottom:-70px;opacity:.48;border-radius:50%;border-top:2px solid #8B68FF;transform:rotate(-16deg);box-shadow:0 -11px 30px #635BFA,0 -25px 0 -23px #95A4FF,0 -48px 0 -46px #6F60E8;z-index:-1;}}
    .dp-eyebrow {{color:#A5B9FF;text-transform:uppercase;font-weight:760;font-size:.67rem;letter-spacing:.16em;margin-bottom:8px;}}
    .dp-hero-title {{color:white;font-size:1.9rem;font-weight:850;letter-spacing:-.04em;line-height:1.25;max-width:760px;}}
    .dp-hero-desc {{color:#ADBFDD;font-size:1.02rem;margin-top:10px;max-width:740px;}}
    .dp-metrics {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:26px;}}
    .dp-metric {{background:{surface};border:1px solid {border};border-radius:12px;padding:16px 17px;display:flex;align-items:center;gap:13px;min-width:0;}}
    .dp-metric-icon {{width:41px;height:41px;border-radius:11px;display:grid;place-items:center;font-size:20px;flex:0 0 auto;}}
    .dp-metric-value {{font-size:1.38rem;font-weight:850;color:{text};letter-spacing:-.04em;line-height:1.1;}}
    .dp-metric-label {{font-size:.92rem;margin-top:5px;color:{muted};}}
    .dp-section {{display:flex;align-items:center;gap:12px;padding:7px 0 11px;margin-top:10px;}}
    .dp-section-number {{height:34px;width:34px;border-radius:50%;display:grid;place-items:center;background:linear-gradient(145deg,#436FFB,#8151F1);color:white;font-weight:850;box-shadow:0 5px 16px rgba(65,99,247,.26);}}
    .dp-section-title {{font-weight:850;color:{text};font-size:1.48rem;letter-spacing:-.03em;}}
    .dp-section-desc {{font-size:.97rem;color:{muted};margin-top:3px;}}
    .dp-chart-label {{font-size:1.12rem;color:{text};font-weight:760;padding-top:2px;}}
    .dp-chart-hint {{font-size:.95rem;color:{muted};padding:4px 0 3px;line-height:1.55;}}
    .dp-insight {{border:1px solid {border};background:{surface};padding:16px 20px;border-radius:12px;color:{text};font-size:1.0rem;line-height:1.7;}}
    .dp-insight-flag {{display:inline-flex;color:#BDA6FF;font-size:.72rem;font-weight:800;letter-spacing:.05em;margin-right:7px;}}
    .dp-insight-grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:8px;}}
    .dp-mini {{border:1px solid {border};border-radius:12px;background:{surface};padding:16px;min-height:114px;}}
    .dp-mini-kicker {{font-size:.71rem;font-weight:800;color:#A98DFF;letter-spacing:.06em;margin-bottom:8px;}}
    .dp-mini-title {{font-size:.9rem;font-weight:760;color:{text};margin-bottom:5px;}}
    .dp-mini-body {{font-size:.94rem;line-height:1.6;color:{muted};}}
    .dp-note {{font-size:.92rem;color:{muted};margin:8px 0 13px;}}
    .dp-feature-grid {{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:20px 0 28px;}}
    .dp-feature {{background:{surface};border:1px solid {border};border-radius:15px;padding:21px;min-height:175px;transition:transform .15s,border-color .15s;}}
    .dp-feature:hover {{transform:translateY(-2px);border-color:#7A70F2;}}
    .dp-feature-icon {{font-size:1.55rem;margin-bottom:13px;}}
    .dp-feature-title {{color:{text};font-size:1.14rem;font-weight:780;margin-bottom:8px;}}
    .dp-feature-detail {{color:{muted};font-size:.93rem;line-height:1.7;}}
    .dp-preview-title {{font-size:1.5rem;font-weight:850;color:{text};margin:15px 0;}}
    .dp-preview-caption {{color:{muted};font-size:.93rem;margin:-5px 0 12px;}}
    @media(max-width:1000px) {{.dp-feature-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
    @media(max-width:660px) {{.dp-feature-grid{{grid-template-columns:1fr}}}}
    @media(max-width:1000px) {{.dp-metrics{{grid-template-columns:repeat(2,1fr)}}.dp-insight-grid{{grid-template-columns:1fr}}}}
    @media(max-width:560px) {{.dp-metrics{{grid-template-columns:1fr}}.dp-hero{{padding:22px}}.dp-hero-title{{font-size:1.35rem}}}}
    </style>""", unsafe_allow_html=True)


def brand():
    st.markdown('<div class="dp-brand"><span class="dp-logo">◈</span><div><div class="dp-brand-name">DataPeek</div><div class="dp-brand-sub">ANALYTICS STUDIO</div></div></div>', unsafe_allow_html=True)


def hero(name, subtitle):
    st.markdown(f'<div class="dp-hero"><div class="dp-eyebrow">◈ &nbsp; DATA ANALYSIS WORKSPACE</div><div class="dp-hero-title">{escape(name)}</div><div class="dp-hero-desc">{escape(subtitle)}</div></div>', unsafe_allow_html=True)


def metric_grid(rows, columns, missing_ratio, memory_mb):
    cards = [
        ("▤", "총 데이터 수", f"{rows:,}", "#5D75FF", "rgba(81,111,252,.15)"),
        ("▥", "컬럼 수", f"{columns:,}", "#45A6FF", "rgba(61,159,255,.14)"),
        ("%", "결측치 비율", f"{missing_ratio:.1f}%", "#FF729D", "rgba(255,100,149,.15)"),
        ("▱", "메모리 사용량", f"{memory_mb:.2f} MB", "#28D5C3", "rgba(40,213,195,.13)"),
    ]
    inner = "".join(f'<div class="dp-metric"><div class="dp-metric-icon" style="background:{bg};color:{color}">{icon}</div><div><div class="dp-metric-value">{number}</div><div class="dp-metric-label">{label}</div></div></div>' for icon, label, number, color, bg in cards)
    st.markdown(f'<div class="dp-metrics">{inner}</div>', unsafe_allow_html=True)


def section(number, title, description=""):
    st.markdown(f'<div class="dp-section"><div class="dp-section-number">{escape(str(number))}</div><div><div class="dp-section-title">{escape(title)}</div><div class="dp-section-desc">{escape(description)}</div></div></div>', unsafe_allow_html=True)


def insight(content, label="자동 분석"):
    return f'<div class="dp-insight"><span class="dp-insight-flag">✦ {escape(label)}</span> {escape(str(content))}</div>'


def mini_cards(items):
    """(소제목, 제목, 내용) 형식 카드 3개."""
    body = "".join(f'<div class="dp-mini"><div class="dp-mini-kicker">{escape(kicker)}</div><div class="dp-mini-title">{escape(title)}</div><div class="dp-mini-body">{escape(content)}</div></div>' for kicker,title,content in items)
    st.markdown(f'<div class="dp-insight-grid">{body}</div>',unsafe_allow_html=True)


def chart_theme(fig, dashboard=False):
    """Plotly 차트에 배경·폰트·축·색상 적용 (원래 데이터 변경 없음)."""
    c = DARK
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Noto Sans KR, sans-serif",size=14,color=c["text"]),
        colorway=CHART_COLORS,
        title=dict(text=""),
        height=480 if dashboard else 530,
        margin=dict(l=70,r=60,t=32,b=90),
        legend=dict(orientation="h",yanchor="top",y=-.22, x=0,font=dict(size=13)),
        hoverlabel=dict(bgcolor=c["surface"],font_color=c["text"]),
    )
    fig.update_xaxes(gridcolor=c["grid"],zerolinecolor=c["grid"],linecolor=c["grid"],tickfont=dict(size=13,color=c["text"]),title_font=dict(size=14,color=c["text"]),automargin=True)
    fig.update_yaxes(tickfont=dict(size=13,color=c["text"]),title_font=dict(size=14,color=c["text"]),automargin=True)
    fig.update_yaxes(gridcolor=c["grid"],zerolinecolor=c["grid"],linecolor=c["grid"])
    # 셀 경계와 각 셀의 상관계수 숫자를 유지하면서 폭·높이를 자동 확장합니다.
    if any(getattr(tr, "type", None) == "heatmap" and getattr(tr, "meta", None) == "datapeek_corr" for tr in fig.data):
        n = max((len(tr.x) for tr in fig.data if getattr(tr, "meta", None) == "datapeek_corr"), default=0)
        fig.update_layout(height=max(490, min(1000, 160 + n * 46)),
                          margin=dict(l=150, r=90, t=34, b=135))
        fig.update_xaxes(tickangle=-38)
    for trace in fig.data:
        if getattr(trace, "type", None) in ("bar", "histogram"):
            marker_color = getattr(trace.marker, "color", None)
            if marker_color is None or (isinstance(marker_color, str) and marker_color.lower() == "#636efa"):
                trace.marker.color = CHART_COLORS[1]
        if getattr(trace, "type", None) == "pie":
            trace.marker.colors=CHART_COLORS
            trace.update(hole=max(trace.hole or 0, .48))
    return fig
