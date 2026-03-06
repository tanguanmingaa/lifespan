# app.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

st.set_page_config(layout="wide", page_title="Trend Event Detector")

st.title("Trend Event Detector — 平滑样条 + 导数法 (Weekly)")

# ---- 左侧控制面板 ----
st.sidebar.header("1) 数据输入")
use_upload = st.sidebar.checkbox("使用文件上传（优先）", value=True)
uploaded_file = None
if use_upload:
    uploaded_file = st.sidebar.file_uploader("上传 CSV（必须包含 date, ratio 列）", type=["csv"])

default_path = st.sidebar.text_input("或输入本地 CSV 路径（若未上传则使用）",
                                     value="/Users/mingtanguan/Desktop/Trump and tariff.csv")

st.sidebar.header("2) 预处理参数")
start_date_str = st.sidebar.text_input("只保留起始日期（YYYY-MM-DD，留空不裁剪）", value="2024-01-01")
resample_rule = st.sidebar.selectbox("周采样方式", options=["W"], index=0, help="W = weekly")
agg_mode = st.sidebar.selectbox("周聚合方式", options=["mean", "median"], index=0)

st.sidebar.header("3) 检测参数")
deriv_q_up = st.sidebar.slider("上升导数分位 (deriv_q_up)", 0.5, 0.95, 0.75, 0.01)
deriv_q_down = st.sidebar.slider("下降导数分位 (deriv_q_down)", 0.05, 0.5, 0.25, 0.01)
min_event_len = st.sidebar.slider("候选事件最短长度（周）", 2, 52, 12, 1)
s_factor_scale = st.sidebar.number_input("平滑系数经验缩放 (s = n * var(y) * scale)", min_value=0.0, value=0.1, step=0.01)

st.sidebar.write("---")
if st.sidebar.button("运行检测"):
    run_now = True
else:
    run_now = False

# ---- 读取数据 ----
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("已读取上传文件")
    except Exception as e:
        st.sidebar.error(f"读取上传文件失败: {e}")
elif default_path:
    try:
        df = pd.read_csv(default_path)
        st.sidebar.write(f"已读取本地文件: {default_path}")
    except Exception as e:
        st.sidebar.warning(f"无法用默认路径读取文件：{e}\n请上传文件或检查路径。")

if df is None:
    st.info("请上传 CSV 或在左侧填入本地路径并点击 Run（或保存后 Streamlit 会自动刷新）。")
    st.stop()

# ---- 时间处理 ----
if 'date' not in df.columns or 'ratio' not in df.columns:
    st.error("CSV 必须包含 'date' 与 'ratio' 两列。")
    st.write("当前列：", list(df.columns))
    st.stop()

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).set_index('date').sort_index()

if start_date_str:
    try:
        df = df.loc[start_date_str:]
    except Exception:
        st.warning("start_date 格式可能不正确，将不裁剪。")

# ---- 从日频到周频 ----
if agg_mode == "mean":
    weekly = df['ratio'].resample(resample_rule).mean()
else:
    weekly = df['ratio'].resample(resample_rule).median()

weekly = weekly.dropna()
if weekly.empty:
    st.error("resample 后没有数据，请检查原始数据与时间范围。")
    st.stop()

# ---- 可选平滑（轻度） ----
smooth_window = st.sidebar.slider("可选滚动平滑窗口 (周，center=True，0 表示不平滑)", 0, 13, 3)
if smooth_window > 1:
    weekly_smooth = weekly.rolling(window=smooth_window, center=True).mean().dropna()
else:
    weekly_smooth = weekly.copy()

# 标准化
weekly_smooth = (weekly_smooth - weekly_smooth.mean()) / weekly_smooth.std()
dates = weekly_smooth.index
x = np.arange(len(weekly_smooth))
y = weekly_smooth.values
n = len(y)

# ---- Splines & derivative ----
s_emp = max(0.0, n * np.var(y) * float(s_factor_scale))
try:
    sp = UnivariateSpline(x, y, s=s_emp, k=3)
    y_hat = sp(x)
    y_der = sp.derivative(n=1)(x)
except Exception as e:
    st.error(f"Spline 拟合出错: {e}")
    st.stop()

# ---- runs helper ----
def runs(idxs):
    if len(idxs)==0: return []
    runs=[]; a=idxs[0]; p=idxs[0]
    for i in idxs[1:]:
        if i==p+1: p=i
        else:
            runs.append((a,p))
            a=i; p=i
    runs.append((a,p))
    return runs

thr_up = np.quantile(y_der, deriv_q_up)
thr_down = np.quantile(y_der, deriv_q_down)
up_runs = runs(np.where(y_der > thr_up)[0])
down_runs = runs(np.where(y_der < thr_down)[0])

# 找峰值（正->负）
signs = np.sign(y_der)
peaks = [i+1 for i in range(len(signs)-1) if signs[i] > 0 and signs[i+1] < 0]

# 配对拼接事件
candidates = []
for (a,b) in up_runs:
    pk = next((p for p in peaks if p >= a), None)
    if pk is None: continue
    # note: 使用 down_runs 的 start (d[0]) 或 end (d[1]) 取决于你想标哪一端
    end = next((d[1] for d in down_runs if d[0] > pk), None)
    if end is None: continue
    if end - a + 1 >= min_event_len:
        candidates.append({'start_idx': a, 'peak_idx': pk, 'end_idx': end})

cand_df = pd.DataFrame(candidates)

if cand_df.empty:
    st.warning("未找到候选事件（可调整阈值或平滑强度）")
else:
    cand_df['start_date'] = cand_df['start_idx'].apply(lambda i: dates[i].strftime('%Y-%m-%d'))
    cand_df['peak_date']  = cand_df['peak_idx'].apply(lambda i: dates[i].strftime('%Y-%m-%d'))
    cand_df['end_date']   = cand_df['end_idx'].apply(lambda i: dates[i].strftime('%Y-%m-%d'))
    cand_df['start_dt'] = pd.to_datetime(cand_df['start_date'])
    cand_df['peak_dt']  = pd.to_datetime(cand_df['peak_date'])
    cand_df['end_dt']   = pd.to_datetime(cand_df['end_date'])
    cand_df['duration_weeks'] = (cand_df['end_dt'] - cand_df['start_dt']).dt.days // 7

# ---- 左右两列展示 ----
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("时序图（平滑 + 导数 + 拟合）")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(dates, y, label="weekly_smooth (normalized)", linewidth=1.5)
    ax.plot(dates, y_hat, label="spline fit", linewidth=1.2, linestyle='--')
    ax.plot(dates, y_der, label="derivative (scaled)", linewidth=1.0, alpha=0.8)
    # draw vertical lines for events
    if not cand_df.empty:
        for _, row in cand_df.iterrows():
            ax.axvline(pd.to_datetime(row['start_date']), color='green', linestyle='--', alpha=0.8, label='start' if _==0 else "")
            ax.axvline(pd.to_datetime(row['peak_date']), color='orange', linestyle='--', alpha=0.8, label='peak' if _==0 else "")
            ax.axvline(pd.to_datetime(row['end_date']), color='red', linestyle='--', alpha=0.8, label='end' if _==0 else "")
    ax.set_title("Weekly ratio (normalized) + spline + derivative")
    ax.legend(loc='upper left')
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("候选事件列表")
    if cand_df.empty:
        st.write("无候选事件，尝试调参（更低的 min_event_len / 更小的平滑 / 修改分位）")
    else:
        st.dataframe(cand_df[['start_date','peak_date','end_date','duration_weeks']])
        # 允许下载 csv
        buf = StringIO()
        cand_df.to_csv(buf, index=False)
        b = buf.getvalue().encode()
        st.download_button("下载候选事件 CSV", data=b, file_name="candidates.csv")

st.markdown("---")
st.subheader("调试信息（用于快速排查）")
st.write(f"数据点数 (weekly_smooth): {n}")
st.write("up_runs:", up_runs)
st.write("down_runs:", down_runs)
st.write("peaks (indices):", peaks)