# DataPeek v5 — Dark Analytics Studio

CSV 파일을 올려 **데이터 통계 + 다양한 그래프 + 간단한 해석**을 제공하는 Streamlit 대시보드입니다.

## v5 변경 사항

- **초기 화면**: 분석 기능 소개 6개와 차트 이미지 미리보기 4개 제공. 예시 이미지는 정적 PNG이며, 자동 로드되는 샘플 CSV는 없습니다.
- **다크 모드 전용**: 라이트 모드 토글과 라이트 전용 리소스를 제거, 사이드바·차트·메인 화면을 다크 테마로 고정
- **업로드 전 메뉴 안내**: CSV 없이 대시보드 외 메뉴를 선택하면 “데이터셋을 업로드해 주세요” 안내 표시. CSV 업로드 후 선택한 메뉴의 분석 화면 활성화
- **글자 확대**: 메뉴, 설명, 그래프 축 및 제목의 폰트 크기를 확대
- **분석 차트 1행 1개**: 기존 2~3열 카드를 1열 전체 폭으로 변경. 모든 분석 페이지와 대시보드에 적용
- **상관관계 히트맵**: Pearson/Spearman에 각 셀 상관계수(-1~1) 표기, 셀 경계선, 넓은 여백/자동 높이 적용
- 이전 DataPeek의 EDA / Target / Train-Test / 이상치 / 모델 검증 / HTML 리포트 유지

## 실행

Windows PowerShell (Python 3.11 이상 권장):

```powershell
cd C:\dev\toy-projects\datapeek
uv venv
uv pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

주소: <http://localhost:8501>

선택 기능인 SHAP을 사용하려면:

```powershell
uv pip install -r requirements-optional.txt
```

## 사용법

1. 대시보드에서는 업로드 전 기능 소개/미리보기를 볼 수 있습니다. CSV는 자동으로 로드되지 않습니다.
2. 업로드 전에 다른 분석 메뉴를 클릭하면 **“데이터셋을 업로드해 주세요”** 안내가 표시됩니다.
3. 왼쪽 **Train CSV**에 분석할 데이터를 업로드하고, 선택적으로 **Test CSV**를 추가합니다.
4. 타깃이 있다면 `타깃 컬럼`을 직접 지정합니다. 기본 선택은 `(없음)`입니다.
5. 메뉴에서 데이터 품질, 타깃 분포, 피처 관계 등을 분석합니다.
6. ML/SHAP/리포트 같은 무거운 처리는 사용자가 직접 버튼으로 실행합니다.

## 주요 구조

```text
datapeek/
├── app.py                          # 화면·메뉴·차트 1열 출력
├── src/
│   ├── ui.py                       # 다크 전용 CSS와 Plotly 스타일
│   ├── landing.py                  # CSV 업로드 전 소개 화면
│   ├── visualizations/gallery.py   # Pearson/Spearman 히트맵 등 그래프 로직
│   ├── analyzer.py
│   ├── statistics.py
│   ├── target.py
│   ├── compare.py
│   ├── outliers.py
│   ├── modeling.py
│   └── report.py
├── assets/previews/                # 소개용 다크 예시 PNG 4장
├── tests/
├── .streamlit/config.toml
├── requirements.txt
└── requirements-optional.txt
```

### 디자인 수정 위치

| 변경 사항 | 파일 |
|---|---|
| 사이드바 / 다크 스타일 / 글꼴 크기 | `src/ui.py` → `set_design()` |
| 차트 높이 / 축 폰트 / 컬러 | `src/ui.py` → `chart_theme()` |
| 차트 한 줄에 한 개 | `app.py` → `show_charts()` |
| Pearson/Spearman 숫자 표시 / 셀 구분 | `src/visualizations/gallery.py` → `correlation_heatmap()` |
| 업로드 전 소개 · 미리보기 | `src/landing.py`, `assets/previews/` |

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

개발 환경에서 **자동 테스트 32개가 모두 통과**했습니다. **Streamlit 서버와 브라우저 화면은 이 파일 제작 환경에 Streamlit이 설치되어 있지 않아 직접 검증하지 못했습니다.** 실제 브라우저에서 다크 레이아웃과 다양한 CSV 업로드를 확인해 주세요.

### 주의

- 히트맵의 `0.83`, `-0.51` 등은 **상관계수**이지 퍼센트가 아닙니다. 상관관계는 인과관계를 뜻하지 않습니다.
- 소개 이미지들은 기능 설명용 가상 그래프일 뿐, 업로드한 데이터의 분석 결과가 아닙니다.
- CSV당 업로드 제한은 15MB이며 분석 실행에 사용하는 데이터는 메모리에 보관됩니다. 서버를 외부에 공개할 경우 별도 접근 통제를 고려해야 합니다.
- 차트 유형 76개는 등록 카탈로그 기준이며 데이터 유형에 따라 일부 차트는 생략됩니다.
