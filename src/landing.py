"""CSV 미업로드 상태의 기능 소개 화면.

이미지는 실제 CSV를 불러오는 샘플 데이터가 아니라, 제공 기능을 설명하는 정적 썸네일입니다.
"""
from html import escape
from pathlib import Path

import streamlit as st

from src.ui import hero, section

PREVIEW_DIR = Path(__file__).resolve().parent.parent / "assets" / "previews"

FEATURES = [
    ("◫", "데이터 개요", "행·열 수, 자료형, 기초 통계 및 변수별 분포를 확인합니다."),
    ("◇", "데이터 품질", "결측률·중복·왜도·데이터 누수 의심 항목을 진단합니다."),
    ("◎", "Target 분석", "클래스 불균형, 타깃 분포, 타깃별 피처 차이를 분석합니다."),
    ("⌘", "Feature 관계", "Pearson/Spearman 상관계수와 피처 간 비선형 패턴을 탐색합니다."),
    ("⇆", "Train/Test 비교", "분포 변화, KS 검정, PSI 및 새로운 범주를 파악합니다."),
    ("◇", "이상치·모델 진단", "IQR/Isolation Forest, 모델 검증, 중요 변수와 오답을 분석합니다."),
]

PREVIEWS = [
    ("target", "타깃 분석 · 클래스 비율", "분류 문제의 불균형을 확인하고 각 클래스의 표본 수를 비교합니다."),
    ("quality", "데이터 품질 · 결측률", "값이 누락된 컬럼과 결측률을 한눈에 파악합니다."),
    ("correlation", "피처 관계 · 상관관계", "각 셀의 상관계수와 경계선을 함께 확인합니다."),
    ("outlier", "이상치 탐지 · 산점도", "평범한 관측값과 이상치 후보를 시각적으로 구별합니다."),
]


def show_landing(catalog):
    """파일 업로드 전 랜딩. 미리보기 그림은 분석용 데이터가 아닙니다."""
    hero("DataPeek | CSV 데이터를 한눈에", "기초 통계부터 Target 분석, 피처 관계, 이상치 탐지, 모델 진단까지. CSV 파일 하나로 데이터 탐색을 시작하세요.")
    st.info("**분석 시작하기**: 왼쪽 사이드바의 **Train CSV**에 파일을 업로드하세요. Test CSV는 선택 사항입니다. 실제 분석은 업로드한 CSV만 사용합니다.")
    section("1", "제공 기능", "다양한 분석 항목, 주요 통계 지표, 풍부한 인터랙티브 그래프를 제공합니다.")
    cards = "".join(
        f'<div class="dp-feature"><div class="dp-feature-icon">{escape(icon)}</div>'
        f'<div class="dp-feature-title">{escape(title)}</div>'
        f'<div class="dp-feature-detail">{escape(detail)}</div></div>'
        for icon, title, detail in FEATURES
    )
    st.markdown(f'<div class="dp-feature-grid">{cards}</div>', unsafe_allow_html=True)
    section("2", "분석 화면 미리보기", "실제 분석에서 제공하는 그래프의 예시입니다. 아래 이미지는 안내용으로 실제 업로드 데이터와 관계없습니다.")
    for key, title, detail in PREVIEWS:
        st.markdown(f'<div class="dp-preview-title">{escape(title)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dp-preview-caption">{escape(detail)}</div>', unsafe_allow_html=True)
        path = PREVIEW_DIR / f"{key}_dark.png"
        st.image(str(path), use_container_width=True)
    with st.expander("지원하는 전체 차트 종류 확인"):
        for area, charts in catalog.items():
            st.markdown(f"**{area} · {len(charts)}종**")
            st.write(" · ".join(charts))
    st.caption("미리보기 이미지는 기능 설명을 위한 가상 예시입니다. 실제 분석에는 사용자가 업로드한 CSV만 사용됩니다.")
