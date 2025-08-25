# streamlit_app.py
# 환경: GitHub Codespaces + Streamlit
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# --- 한글 폰트 강제 등록 (Pretendard-Bold.ttf가 있으면 사용) ---
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

# --- NOAA OISST 데이터 URL ---
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
            lat=slice(-10, 60),  # 아시아 범위
            lon=slice(60, 150)
        ).squeeze()
        da.load()
        if np.all(np.isnan(da.values)):
            return None
        return da
    except Exception as e:
        st.error(f"데이터 불러오기 실패: {e}")
        return None

def plot_sst(da, date):
    if da is None:
        return None
    fig, ax = plt.subplots(
        figsize=(9, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    norm = TwoSlopeNorm(vmin=20, vcenter=30, vmax=34)
    im = da.plot.pcolormesh(
        ax=ax, x="lon", y="lat",
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd", norm=norm, add_colorbar=False
    )
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black")
    try:
        gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels, gl.right_labels = False, False
    except Exception:
        ax.gridlines(color="gray", alpha=0.5, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05, aspect=40)
    cbar.set_label("해수면 온도 (°C)" if "Pretendard" in matplotlib.rcParams["font.family"] else "Sea Surface Temp (°C)")

    ax.set_title(f"해수면 온도: {date.strftime('%Y-%m-%d')}")
    fig.tight_layout()
    return fig

# --- 보고서 시작 ---
st.title("📊 해수면 온도 상승은 고등학생에게 어떠한 영향을 미치는가?")

st.header("1. 조사 동기와 목적")
st.markdown("""
해수면 온도 상승은 기후변화의 핵심 지표 중 하나로, 고등학생들의 현재와 미래 삶 전반에 영향을 미칩니다.  
특히 건강, 학업, 생활환경, 지역사회에 걸친 다층적 영향을 조사하고자 본 연구를 시작하였습니다.
""")

st.header("2. 조사 계획")
st.subheader("(1) 조사 기간")
st.markdown("2025년 1월 ~ 2025년 8월")

st.subheader("(2) 조사 방법과 대상")
st.markdown("""
- 자료 분석: NOAA OISST v2 High Resolution Dataset 활용  
- 문헌 조사: 기상청, 연구 논문, 보도자료 등  
- 대상: 고등학생의 건강·학습·생활에 미치는 영향 전반
""")

st.header("3. 조사 결과")

st.subheader("(1) NOAA OISST v2 High Resolution Dataset을 활용한 해수면 온도 시각화 차트")

# 날짜 선택 위젯
default_date = datetime.date.today() - datetime.timedelta(days=2)
date = st.date_input("날짜 선택", value=default_date, min_value=datetime.date(1981, 9, 1), max_value=default_date)

with st.spinner("데이터 불러오는 중..."):
    da = load_sst(date)

if da is not None:
    fig = plot_sst(da, date)
    if fig:
        st.pyplot(fig, clear_figure=True)
    with st.expander("데이터 미리보기"):
        st.write(da)
else:
    st.warning("데이터를 표시할 수 없습니다. 날짜를 변경하거나 네트워크를 확인하세요.")

st.subheader("(2) 건강과 정신적 영향")
st.markdown("""
- 소아천식 발생률 4–11% 증가, 응급실 방문 17–30% 증가 예상  
- 라임병 환자 79–241% 증가 추정  
- 청소년의 60%가 기후변화에 극도의 걱정을 호소, 학업 집중력 저하
""")

st.subheader("(3) 학업 및 생활 환경 영향")
st.markdown("""
- 폭염으로 학업성취도 4–7% 감소  
- 극한 기상으로 2억4천만 명 학생 수업시간 단축  
- 난류성 어종 증가 → 지역 식문화, 어업 변화  
- 해양 열파 증가 → 태풍·집중호우·폭염 위험 상승
""")

st.header("4. 결론")
st.markdown("""
해수면 온도 상승은 고등학생들의 건강, 학업, 지역사회 전반에 광범위한 영향을 미칩니다.  
따라서 기후 위기 대응 교육과 지역사회 차원의 체계적 대응이 시급합니다.
""")

st.header("5. 참고 자료")
st.markdown("""
[1] 기상청 보도자료  
[2] 한국해양수산개발원 연구보고서  
[3] Newstree 기후변화 기사  
[4] 전 세계 청소년 기후불안 조사 보고서  
[5] Planet03 해양열파 연구  
… (상세 참고문헌은 별첨)
""")
