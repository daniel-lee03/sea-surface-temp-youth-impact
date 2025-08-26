# streamlit_app.py
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe
import matplotlib.patches as patches

# --- 폰트 설정 ---
import matplotlib
from matplotlib import font_manager as fm, rcParams
from pathlib import Path

def setup_font():
    font_path = Path(__file__).parent / "fonts" / "Pretendard-Bold.ttf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        rcParams["font.family"] = font_name
    rcParams["axes.unicode_minus"] = False

setup_font()
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.25
PE = [pe.withStroke(linewidth=2.5, foreground="white")]

# --- NOAA OISST 데이터 URL ---
# NOAA OISST v2 High Resolution Dataset: 
# 미국 해양대기청(NOAA)에서 제공하는 고해상도(0.25도 격자) 일일 해수면 온도(SST) 데이터셋입니다. 
# 위성, 선박, 부표 등 다양한 출처의 관측 자료를 종합하여 생성됩니다.
BASE_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"

@st.cache_data(show_spinner=False)
def load_sst(date: datetime.date):
    year = date.year
    url = BASE_URL.format(year=year)
    try:
        try:
            ds = xr.open_dataset(url)
        except Exception:
            ds = xr.open_dataset(url, engine="pydap")
        da = ds["sst"].sel(
            time=date.strftime("%Y-%m-%d"),
            lat=slice(28, 42), lon=slice(120, 135)
        ).squeeze()
        da.load()
        if np.all(np.isnan(da.values)):
            return None
        return da
    except Exception as e:
        st.error(f"데이터 불러오기 실패: {e}")
        return None

def plot_sst(da, date):
    fig, ax = plt.subplots(
        figsize=(9, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent([120, 135, 28, 42], crs=ccrs.PlateCarree())
    norm = TwoSlopeNorm(vmin=20, vcenter=30, vmax=34)
    im = da.plot.pcolormesh(
        ax=ax, x="lon", y="lat",
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd", norm=norm, add_colorbar=False
    )
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05)
    cbar.set_label("해수면 온도 (℃)")
    ax.set_title(f"해수면 온도: {date.strftime('%Y-%m-%d')}")
    return fig

# --- 유틸 함수 (Bullet, Lollipop, Combo, Waffle) ---
def bullet(ax, value, target, label="", color="#F28E2B"):
    lo, hi = min(value, target), max(value, target)
    pad = (hi - lo) * 0.5 + 0.5
    vmin, vmax = lo - pad, hi + pad
    ax.barh([0], [vmax - vmin], left=vmin, color="#EEEEEE", height=0.36)
    ax.barh([0], [value - vmin], left=vmin, color=color, height=0.36)
    ax.axvline(target, color="#333333", lw=2.2)
    ax.set_yticks([]); ax.set_xlim(vmin, vmax); ax.set_xlabel("℃"); ax.set_title(label)
    delta = value - target
    badge = f"+{delta:.1f}℃" if delta >= 0 else f"{delta:.1f}℃"
    ax.text(value, 0.1, f"{value:.1f}℃", ha="left", va="bottom", weight="bold", path_effects=PE)
    ax.text(0.02, 0.9, badge, transform=ax.transAxes,
            fontsize=12, weight="bold", color="white", path_effects=PE,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#C1272D" if delta>=0 else "#2B7A78",
                      edgecolor="none"))

def lollipop_horizontal(ax, labels, values, title, unit="℃", color="#4C78A8", highlight_color="#E45756"):
    idx = np.argsort(values)[::-1]
    labels_sorted = [labels[i] for i in idx]
    values_sorted = [values[i] for i in idx]
    y = np.arange(len(labels_sorted))
    ax.hlines(y, [0]*len(values_sorted), values_sorted, color="#CCCCCC", lw=3)
    vmax_i = int(np.argmax(values_sorted))
    for i, v in enumerate(values_sorted):
        col = highlight_color if i == vmax_i else color
        ax.plot(v, y[i], "o", ms=10, mfc=col, mec=col)
        ax.text(v + max(values_sorted)*0.03, y[i], f"{v:.2f}{unit}" if unit.endswith("년") else f"{v:.1f}{unit}",
                va="center", weight="bold" if i == vmax_i else 500, color=col, path_effects=PE)
    ax.set_yticks(y, labels_sorted); ax.set_xlabel(unit); ax.set_title(title); ax.grid(axis="x", alpha=0.25)

def combo_bar_line(ax, x_labels, bars, line, bar_color="#FDB863", line_color="#C1272D"):
    x = np.arange(len(x_labels))
    ax.bar(x, bars, color=bar_color, width=0.55)
    ax.set_xticks(x, x_labels); ax.set_ylabel("총 환자 수(명)")
    ax2 = ax.twinx()
    ax2.plot(x, line, marker="o", ms=7, lw=2.5, color=line_color)
    ax2.set_ylabel("총 사망자 수(명)", color=line_color)

def waffle(ax, percent, rows=10, cols=10, on="#F03B20", off="#EEEEEE", title=None):
    total = rows*cols
    k = int(round(percent/100*total))
    for i in range(total):
        r = i // cols; c = i % cols
        color = on if i < k else off
        rect = patches.Rectangle((c, rows-1-r), 0.95, 0.95, facecolor=color, edgecolor="white")
        ax.add_patch(rect)
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.axis("off")
    if title: ax.set_title(title)
    ax.text(cols/2, rows/2, f"{percent:.0f}%", ha="center", va="center",
            fontsize=20, weight="bold", color="#333", path_effects=PE)

# --- 보고서 본문 ---
st.title("🌊 해수면 온도 상승은 고등학생에게 어떠한 영향을 미치는가?")

st.header("I. 서론: 뜨거워지는 바다, 위협받는 교실")
st.markdown("""
한반도는 지구 평균보다 2~3배 빠른 해수면 온도 상승을 겪고 있으며, 이는 더 이상 추상적인 환경 문제가 아니라 
미래 세대의 학습권과 건강을 직접적으로 위협하는 현실입니다. 본 보고서는 고등학생을 기후 위기의 가장 취약한 집단이자 
변화의 핵심 동력으로 조명하며, SST 상승의 실태와 파급효과를 다각도로 분석합니다.
""")

st.header("II. 조사 계획")
st.subheader("1) 조사 기간")
st.markdown("2025년 1월 ~ 2025년 8월")
st.subheader("2) 조사 방법과 대상")
st.markdown("""
- **데이터 분석**: NOAA OISST v2 High Resolution Dataset
  - *미국 해양대기청(NOAA)에서 제공하는 고해상도(0.25도 격자) 일일 해수면 온도 데이터로, 위성, 선박, 부표 등 다양한 관측 자료를 종합하여 생성됩니다.*
- **문헌 조사**: 기상청, 연구 논문, 보도자료 등 
- **대상**: 대한민국 고등학생의 건강·학업·사회경제적 영향
""")

st.header("III. 조사 결과")

# NOAA 지도
st.subheader("1) NOAA OISST Dataset 시각화")
date = st.date_input("날짜 선택", value=datetime.date.today()-datetime.timedelta(days=2))
with st.spinner("데이터 불러오는 중..."):
    da = load_sst(date)
if da is not None:
    st.pyplot(plot_sst(da, date), clear_figure=True)

# 1번 데이터 (Bullet)
st.subheader("📈 최근 기록과 평년 대비 편차 — Bullet Charts")
c1, c2, c3 = st.columns(3)
with c1:
    fig, ax = plt.subplots(figsize=(5,2.6))
    bullet(ax, 23.2, 21.2, label="2024-10 vs 최근10년")
    st.pyplot(fig, clear_figure=True)
    st.caption("➡️ 2024년 10월 해수면 온도는 최근 10년 평균보다 2.0℃ 높음.")
with c2:
    fig, ax = plt.subplots(figsize=(5,2.6))
    bullet(ax, 19.8, 19.2, label="2023 연평균 vs 2001–2020", color="#2E86AB")
    st.pyplot(fig, clear_figure=True)
    st.caption("➡️ 연평균 기준으로도 장기 평균 대비 +0.6℃ 상승.")
with c3:
    fig, ax = plt.subplots(figsize=(5,2.6))
    bullet(ax, 22.6, 22.6-2.8, label="서해 2024-10 vs 최근10년", color="#E67E22")
    st.pyplot(fig, clear_figure=True)
    st.caption("➡️ 서해는 평년 대비 +2.8℃ 상승, 단기간 급격히 뜨거워짐.")

# 2번 데이터 (Lollipop)
st.subheader("📊 해역별 장·단기 상승과 편차 — Lollipop Charts")
regions = ["동해", "서해", "남해"]
rise_1968_2008 = [1.39, 1.23, 1.27]
rate_since_2010 = [0.36, 0.54, 0.38]
anom_2024 = [3.4, 2.8, 1.1]
cL1, cL2, cL3 = st.columns(3)
with cL1:
    fig, ax = plt.subplots(figsize=(4.8,3))
    lollipop_horizontal(ax, regions, rise_1968_2008, title="장기 상승폭 (1968–2008)", unit="℃")
    st.pyplot(fig, clear_figure=True)
    st.caption("➡️ 동해가 +1.39℃로 장기 상승폭 최대.")
with cL2:
    fig, ax = plt.subplots(figsize=(4.8,3))
    lollipop_horizontal(ax, regions, rate_since_2010, title="연평균 상승률 (2010~)", unit="℃/년", color="#59A14F")
    st.pyplot(fig, clear_figure=True)
    st.caption("➡️ 최근 10여년간은 서해가 +0.54℃/년으로 가장 빠르게 상승.")
with cL3:
    fig, ax = plt.subplots(figsize=(4.8,3))
    lollipop_horizontal(ax, regions, anom_2024, title="2024 편차", unit="℃", color="#F28E2B")
    st.pyplot(fig, clear_figure=True)
    st.caption("➡️ 2024년 동해는 평년 대비 +3.4℃ 치솟음.")

# 3번 데이터 (온열질환)
st.subheader("🧑‍⚕️ 온열질환 연도별 현황 — Bars + Line")
years = ["2022년", "2023년", "2024년"]
patients = [1564, 2818, 3704]
deaths = [9, 32, 34]
figM, axM = plt.subplots(figsize=(8, 3.6))
combo_bar_line(axM, years, patients, deaths)
axM.set_title("온열질환 환자·사망 추이")
st.pyplot(figM, clear_figure=True)
st.caption("➡️ 환자 수는 3년간 급격히 증가, 사망자도 꾸준히 발생.")

# 4번 데이터 (기후우울)
st.subheader("🧠 청소년 기후불안 — Waffle Charts")
cwa, cwb = st.columns(2)
with cwa:
    figW1, axW1 = plt.subplots(figsize=(4.2, 4.2))
    waffle(axW1, 59, title="매우/극도로 우려(%)")
    st.pyplot(figW1, clear_figure=True)
with cwb:
    figW2, axW2 = plt.subplots(figsize=(4.2, 4.2))
    waffle(axW2, 45, title="일상에 부정적 영향(%)")
    st.pyplot(figW2, clear_figure=True)
st.caption("➡️ 전 세계 16–25세 중 59%는 기후변화를 극도로 우려, 45%는 일상에 부정적 영향 보고.")

# --- 추가된 내용 ---
st.subheader("2) 기후 위기의 전조: 극단적 기상 현상의 일상화")
st.markdown("""
- **더 강력한 태풍**: 따뜻한 바다의 잠열 공급 → 태풍 세력 강화  
- **집중호우 빈발**: 기온 1℃ 상승 → 대기 수증기 7% 증가 → '물폭탄' 국지성 폭우  
- **혹독한 폭염**: 해양 열돔(Heat Dome) 현상 → 학업 집중력 저하, 건강 위협
""")

st.subheader("3) 청소년 건강 영향")
st.markdown("""
- **신체 건강**: 온열질환자 2024년 3,704명 (전년 대비 급증), 미세먼지 증가 → 천식·알레르기 악화  
- **정신 건강**: 전 세계 청소년 59% 기후변화 극도 우려, '기후 우울' 확산 → 집중력·학업 동기 저하
""")

st.subheader("4) 학업과 학교생활 영향")
st.markdown("""
- 폭염 시 시험 점수 **1℃ 상승당 1% 감소** - 태풍·집중호우로 인한 **휴교 증가 → 학습 결손** - 냉방·환기 시설 격차로 인한 **기후 성취 격차(Climate Achievement Gap)**
""")

st.subheader("5) 생활환경과 사회경제적 파장")
st.markdown("""
- 어종 변화: 명태·도루묵 ↓, 오징어·멸치 ↑, 아열대 어종 등장  
- 식품 가격 변동 → 저소득층 가계 부담 증가, 학교 급식 질 악화  
- 식문화 단절 → 세대 간 경험 단절, 청소년 불안 심화
""")

st.subheader("6) 대응과 미래 세대를 위한 제언")
st.markdown("""
- **정책**: 학교 냉방·환기 시스템 현대화, 청소년 건강 통계 세분화  
- **교육**: 기후변화 과목 신설, 프로젝트 기반 학습, '기후테크' 진로 지도  
- **청소년 행동**: 플라스틱 저감 캠페인, 기후행동 소송, 지역사회 활동 확산  
""")

# --- 결론 및 참고 자료 ---
st.header("IV. 결론")
st.markdown("""
대한민국 주변 해수면 온도의 상승은 단순한 해양 문제가 아니라,  
고등학생들의 건강·학업·생활 전반을 위협하는 **복합 위기**입니다.  
그러나 교육과 청소년 주도의 기후 행동을 통해 이 위기를 기회로 전환할 수 있습니다.  
""")

st.header("V. 참고 자료")
st.markdown("""
- 기상청 보도자료 (2024)  
- 한국해양수산개발원 연구보고서  
- 청소년 기후불안 국제 조사 (Lancet, 2021)  
- Planet03 해양열파 연구 (2021)  
- Newstree, YTN Science 외 기사 및 연구논문  
""")

st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: gray; font-size: 0.9em;'>
        미림마이스터고등학교 1학년 4반 5조 · 지속가능한지구사랑해조
    </div>
    """,
    unsafe_allow_html=True
)


