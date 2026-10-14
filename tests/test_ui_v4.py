"""Streamlit 서버 없이 구조/히트맵/정적 미리보기를 검사합니다."""
import ast
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

from src.visualizations.gallery import correlation_heatmap, relationship_charts

ROOT = Path(__file__).resolve().parents[1]

class TestUIv4(unittest.TestCase):
    def test_annotated_correlation_has_visible_grid(self):
        frame = pd.DataFrame({"weight": [1, 2, 3, 4, 5], "length": [2, 4, 6, 8, 10],
                              "year": [4, 1, 5, 2, 3]})
        fig = correlation_heatmap(frame.corr())
        heatmap = fig.data[0]
        self.assertEqual(heatmap.xgap, 3)
        self.assertEqual(heatmap.ygap, 3)
        self.assertEqual(len(fig.layout.annotations), 9)
        self.assertEqual(fig.layout.annotations[0].text, "1.00")
        self.assertIn("상관계수", heatmap.hovertemplate)

    def test_relationship_uses_annotated_heatmaps(self):
        frame = pd.DataFrame({"a":[1,3,4,7],"b":[5,4,3,1],"c":[3,2,5,6]})
        charts = relationship_charts(frame,None,"a","회귀")
        heatmaps = [fig for name,fig,tip in charts if 'Heatmap' in name]
        self.assertEqual(len(heatmaps),2)
        self.assertTrue(all(len(fig.layout.annotations)>0 for fig in heatmaps))

    def test_charts_are_single_column(self):
        source=(ROOT/'app.py').read_text(encoding='utf-8')
        root=ast.parse(source)
        func=next(n for n in root.body if isinstance(n,ast.FunctionDef) and n.name=='show_charts')
        calls=[node for node in ast.walk(func) if isinstance(node,ast.Call)]
        self.assertFalse(any(isinstance(c.func,ast.Attribute) and c.func.attr=='columns' for c in calls))
        self.assertTrue(any(isinstance(c.func,ast.Attribute) and c.func.attr=='plotly_chart' for c in calls))

    def test_no_sample_csv_and_previews_exist(self):
        self.assertFalse(list(ROOT.rglob('*.csv')))
        for stem in ('target','quality','correlation','outlier'):
            self.assertGreater((ROOT/'assets'/'previews'/f'{stem}_dark.png').stat().st_size, 5_000)
        self.assertFalse(list((ROOT/'assets'/'previews').glob('*_light.png')))

    def test_dark_design_only(self):
        fake_st=types.ModuleType('streamlit')
        shown=[]
        fake_st.markdown=lambda value, **kwargs: shown.append(value)
        from unittest.mock import patch
        with patch.dict(sys.modules, {"streamlit": fake_st}):
            from src import ui
        original_st=ui.st
        try:
            ui.st=fake_st
            ui.set_design()
            css=shown[-1]
            self.assertIn('background:#080F20',css)
            self.assertIn('color:#F3F6FF',css)
            self.assertIn('stSidebar',css)
            self.assertNotIn('#F4F7FE',css)
        finally:
            ui.st=original_st
        self.assertFalse(hasattr(ui,'COLORS'))
        source=(ROOT/'app.py').read_text(encoding='utf-8')
        self.assertNotIn('st.toggle("라이트 모드"',source)
        self.assertNotIn('theme_name',source)
        self.assertEqual((ROOT/'.streamlit'/'config.toml').read_text().count('base = "dark"'),1)

    def test_no_csv_menu_shows_warning_but_dashboard_shows_landing(self):
        """앱의 CSV 미업로드 분기를 직접 실행해 메뉴 동작을 확인합니다."""
        import sys
        from unittest.mock import patch
        source=(ROOT/'app.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        branch=next(node for node in tree.body if isinstance(node,ast.If)
                    and isinstance(node.test,ast.Compare)
                    and ast.unparse(node.test)=='train_bytes is None')
        code=compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),str(ROOT/'app.py'),'exec')
        class Halt(Exception): pass
        def run(page):
            shown=[]
            class FakeSt:
                def header(self,x): shown.append(('header',x))
                def warning(self,x): shown.append(('warning',x))
                def info(self,x): shown.append(('info',x))
                def stop(self): raise Halt()
            landing=types.ModuleType('src.landing')
            landing.show_landing=lambda catalog: shown.append(('landing',catalog))
            ctx={'st':FakeSt(), 'train_bytes':None, 'page':page,
                 'PAGES':['대시보드','데이터 개요','데이터 품질'], 'CHART_CATALOG':{'overview':['Histogram']}}
            with patch.dict(sys.modules,{'src.landing':landing}):
                try: exec(code,ctx)
                except Halt: pass
            return shown
        self.assertEqual(run('대시보드'),[('landing',{'overview':['Histogram']})])
        result=run('데이터 개요')
        self.assertTrue(any(k=='warning' and '데이터셋을 업로드해 주세요' in v for k,v in result))
        self.assertFalse(any(k=='landing' for k,v in result))
        self.assertTrue(any(k=='info' and 'Train CSV' in v for k,v in result))

    def test_dark_preview_lookup_only(self):
        source=(ROOT/'src'/'landing.py').read_text(encoding='utf-8')
        self.assertIn('f"{key}_dark.png"',source)
        self.assertNotIn('_{theme}',source)

if __name__=='__main__': unittest.main()
