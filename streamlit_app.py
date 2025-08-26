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
import textwrap

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
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

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
st.title("🌊 뜨거워지는 지구: 해수면 온도 상승이 고등학생에게 미치는 영향")

st.header("I. 서론: 뜨거워지는 바다, 위협받는 교실")
st.markdown("""
한반도는 지구 평균보다 2~3배 빠른 해수면 온도 상승을 겪고 있으며, 이는 더 이상 추상적인 환경 문제가 아니라 
미래 세대의 학습권과 건강을 직접적으로 위협하는 현실입니다. 본 보고서는 고등학생을 기후 위기의 가장 취약한 집단이자 
변화의 핵심 동력으로 조명하며, 해수면 온도(Sea Surface Temperature, SST) 상승의 실태와 파급효과를 다각도로 분석합니다.
""")

st.header("II. 조사 계획")
st.subheader("1) 조사 기간")
st.markdown("2025년 7월 ~ 2025년 8월")
st.subheader("2) 조사 방법과 대상")
st.markdown("""
- **데이터 분석**: NOAA OISST v2 High Resolution Dataset
  - *미국 해양대기청(NOAA)에서 제공하는 고해상도(0.25도 격자) 일일 해수면 온도 데이터로, 위성, 선박, 부표 등 다양한 관측 자료를 종합하여 생성됩니다.*
- **문헌 조사**: 기상청, 연구 논문, 보도자료 등 
- **대상**: 대한민국 고등학생의 건강·학업·사회경제적 영향
""")

st.header("III. 조사 결과")

# NOAA 지도
st.subheader("1) 한반도 주변 해수면 온도 상승 실태")
date = st.date_input("날짜 선택", value=datetime.date.today()-datetime.timedelta(days=2))
with st.spinner("데이터 불러오는 중..."):
    da = load_sst(date)
if da is not None:
    st.pyplot(plot_sst(da, date), clear_figure=True)

# Bullet Charts
st.subheader("📈 최근 기록과 평년 대비 편차")
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

# Lollipop Charts
st.subheader("📊 해역별 장·단기 상승과 편차")
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

st.subheader("2) 지구에 미치는 영향: 극단적 기상 현상의 심화")
st.markdown("""
해수면 온도 상승은 단순히 바다만 뜨거워지는 현상에 그치지 않고, 대기와 상호작용하며 지구 전체의 기상 시스템을 교란합니다.
- **더 강력한 태풍**: 따뜻한 바다는 태풍에 더 많은 에너지(잠열)를 공급하여, 과거보다 훨씬 강력한 '슈퍼 태풍'으로 발달할 가능성을 높입니다.
- **집중호우 빈발**: 기온이 1℃ 오르면 대기가 머금을 수 있는 수증기량은 약 7% 증가합니다. 이는 예측 불가능한 '물폭탄' 형태의 국지성 폭우로 이어져 홍수 위험을 키웁니다.
- **혹독한 폭염**: 바다가 데워지면 대기 역시 뜨거워져 '열돔(Heat Dome)' 현상이 발생하기 쉽습니다. 이는 한 지역에 폭염이 장기간 지속되는 결과로 나타납니다.
""")
# 데이터 준비
temps2 = np.arange(0, 6)  # 0~5℃
humidity_increase = 7 * temps2  # 1℃당 7% 증가 (선형近似)

figH2, axH2 = plt.subplots(figsize=(7,4))

# 곡선 라인
axH2.plot(temps2, humidity_increase, color="#2E86AB", lw=3, marker="o")

# 곡선 아래 채우기 (gradient-like 효과)
axH2.fill_between(temps2, humidity_increase, color="#2E86AB", alpha=0.2)

# 축 설정
axH2.set_xlabel("기온 상승 (℃)")
axH2.set_ylabel("대기 수증기량 증가율 (%)")
axH2.set_title("기온 상승에 따른 대기 수증기량 증가")

# 주요 포인트 강조
highlight_points = {1: 7, 2: 14, 3: 21, 4: 28, 5: 35}
for t, v in highlight_points.items():
    axH2.scatter(t, v, color="red", zorder=5)
    axH2.annotate(f"+{v:.0f}%", (t, v),
                  textcoords="offset points", xytext=(0,10),
                  ha="center", color="red", weight="bold")

st.pyplot(figH2, clear_figure=True)

st.caption("➡️ 기온이 1℃ 오를 때마다 약 7%씩 대기 수증기량이 늘어나 집중호우 가능성이 커집니다.")

st.subheader("3) 고등학생에게 미치는 영향")
st.markdown("이러한 지구 환경의 변화는 고등학생들의 학습 환경과 신체적, 정신적 건강에 직접적인 영향을 미칩니다.")

st.markdown("##### 학업 성취도 저하")
st.markdown("""
- **폭염과 학습 효율**: 폭염은 교실의 온도를 높여 학생들의 집중력을 심각하게 저하시킵니다.<br>
  **전미경제연구소(NBER)의 'Heat and Learning' 연구**에 따르면, 교내 냉방 시설이 없는 환경에서  
  **기온이 섭씨 1°C 상승할 때마다 학생들의 학업 성취도는 약 1.8%씩 감소**했습니다.
- **학습 결손 및 교육 불평등**: 강력해진 태풍과 집중호우로 인한 잦은 휴교는 학습 결손으로 이어집니다.<br>
  또한, 냉방 시설이 잘 갖춰진 학교와 그렇지 않은 학교 간의 학습 환경 차이는  
  <strong>기후 성취 격차(Climate Achievement Gap)</strong>를 유발하여 교육 불평등을 심화시킬 수 있습니다.
""", unsafe_allow_html=True)


# --- 📉 기온 상승 → 학업 성취도 감소 (시각화) ---
temps = np.arange(0, 6)  # 0~5℃
impact = 100 - (1.8 * temps)  # 100% 기준 → °C당 감소 반영

figC, axC = plt.subplots(figsize=(7,4))

# 막대 그래프
axC.bar(temps, impact, color="#FDB863", alpha=0.7, label="구간별 학업 성취도")

# 선 그래프 (추세선)
axC.plot(temps, impact, marker="o", color="#C1272D", lw=2.5, label="추세선 (1℃ 당 -1.8%)")

# 축/제목
axC.set_xlabel("기온 상승 (℃)")
axC.set_ylabel("학업 성취도 (%)")
axC.set_title("기온 상승이 학업 성취도에 미치는 영향")
axC.set_ylim(80, 102)

# 데이터 라벨
for t, v in zip(temps, impact):
    axC.text(t, v+0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

axC.legend()
st.pyplot(figC, clear_figure=True)

st.caption("➡️ 기온이 1℃ 오를 때마다 학생들의 학업 성취도는 약 1.8%씩 감소합니다.")


st.markdown("##### 신체 및 정신 건강 위협")
# 온열질환 차트
years = ["2022년", "2023년", "2024년"]
patients = [1564, 2818, 3704]
deaths = [9, 32, 34]
figM, axM = plt.subplots(figsize=(8, 3.6))
combo_bar_line(axM, years, patients, deaths)
axM.set_title("온열질환 환자·사망 추이")
st.pyplot(figM, clear_figure=True)
st.caption("➡️ 폭염의 일상화로 온열질환자가 급증하고 있습니다.")

# 기후우울 차트
cwa, cwb = st.columns(2)
with cwa:
    figW1, axW1 = plt.subplots(figsize=(4.2, 4.2))
    waffle(axW1, 59, title="기후변화를 매우/극도로 우려")
    st.pyplot(figW1, clear_figure=True)
with cwb:
    figW2, axW2 = plt.subplots(figsize=(4.2, 4.2))
    waffle(axW2, 45, title="일상에 부정적 영향을 받음")
    st.pyplot(figW2, clear_figure=True)

st.markdown("""
- **신체 건강**: 폭염으로 인한 온열질환의 위험이 커지고, 대기오염 물질이 정체되면서 천식이나 알레르기 같은 호흡기 질환이 악화될 수 있습니다.
- **정신 건강**: 기후 위기의 심각성을 인지하는 청소년들은 미래에 대한 불안감, 무력감, 분노 등을 느끼며 '기후 우울(Climate Anxiety)'을 겪습니다. **의학 저널 '랜싯'의 연구**에 따르면, 전 세계 **16-25세 청소년의 45%가 기후 변화에 대한 걱정으로 일상생활(학업, 수면, 여가 등)에 부정적인 영향을 받는다**고 답했으며, **59%는 기후 변화를 '매우 또는 극도로' 우려**하는 것으로 나타났습니다.
""")

st.subheader("4) 대응과 미래 세대를 위한 제언")
st.markdown("""
- **정책**: 모든 학교에 냉방 및 환기 시스템을 현대화하고, 기후 변화에 따른 청소년 건강 영향을 추적하는 세분화된 통계를 구축해야 합니다.
- **교육**: 기후변화를 정규 교과목으로 편성하고, 문제 해결 중심의 프로젝트 기반 학습을 확대해야 합니다. 또한, '기후테크'와 같은 새로운 진로 분야에 대한 지도가 필요합니다.
- **청소년 행동**: 플라스틱 저감 캠페인, 기후행동 소송 참여, 지역사회 환경 문제 해결 등 청소년이 주도하는 기후 행동을 적극적으로 지원하고 확산해야 합니다.
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
- Goodman, J., & Park, R. J. (2018). *Heat and Learning*. NBER Working Paper.
- Hickman, C., et al. (2021). Climate anxiety in children and young people and their beliefs about government responses to climate change: a global survey. *The Lancet Planetary Health*.
- 기상청 보도자료 (2024)  
- 한국해양수산개발원 연구보고서  
- Planet03 해양열파 연구 (2021)  
- Newstree, YTN Science 외 기사 및 연구논문  
""")

st.markdown(
    """
    <hr style='border:1px solid #ccc; margin-top:30px; margin-bottom:10px;'/>
    <div style='text-align: center; padding: 10px; color: gray; font-size: 0.9em;'>
        미림마이스터고등학교 1학년 4반 1조 · 지속가능한지구사랑해조
    </div>
    """,
    unsafe_allow_html=True
)