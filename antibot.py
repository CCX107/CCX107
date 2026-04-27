import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from impala.dbapi import connect
# 导入你的自定义类
from anti_bot_utils import UnifiedUserBehaviorCleaner

# --- 页面基础设置 ---
st.set_page_config(page_title="🎯 黑产实时监控大盘", layout="wide")

# --- 1. 数据库与全量数据拉取 (一级缓存: 解决5分钟SQL痛点) ---
@st.cache_resource
def get_db_conn():
    pw = os.environ.get('DB_PASSWORD', 'Lx05761081') 
    return connect(
        host='192.168.101.238', port=21050, database='rawdata',
        user='caixinlei@your_company.com',  # 注意：后缀已做通用化处理
        password=pw, 
        auth_mechanism='NOSASL'
    )

@st.cache_data(ttl=86400, show_spinner="⏳ 首次启动：正在从数据仓库拉取近 21 天全量日志，请稍候...")
def fetch_big_data():
    """固定拉取近3周数据存入内存，不再随用户随意点击而重跑"""
    conn = get_db_conn()
    end_s = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_s = (datetime.now() - timedelta(days=22)).strftime('%Y-%m-%d')
    
    sql = f"""
    /*SA(default)*/ SELECT date, distinct_id,
                   hour(TIME) AS hour_time,
                   $city, $os, $province, $browser, $ip,
                   $is_first_day, $is_first_time, $title,
                   $url, $referrer, $is_login_id, $manufacturer
       FROM EVENTS
       WHERE event = '$pageview'
         AND $url not like '%eu.xxxxx.com%'
         AND date BETWEEN '{start_s}' AND '{end_s}'
         AND $lib = 'js'
    """
    df = pd.read_sql(sql, conn)
    # 将日期标准化，方便后续进行比较切片
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# --- 2. 模型推理 (二级缓存: 解决模型重复计算痛点) ---
@st.cache_resource
def load_model():
    return joblib.load('./antibot_pipeline_v1.pkl')

@st.cache_data(show_spinner="🧠 正在对该时段数据进行高维特征提取与孤立森林预测...")
def run_pipeline(df_subset):
    """
    缓存针对特定日期切片的特征提取结果。
    只要日期没变，拖动任何滑动条都不会重跑这部分。
    """
    if df_subset.empty:
        return pd.DataFrame()
        
    pipeline = load_model()
    cleaner = pipeline.named_steps['cleaner']
    model = pipeline.named_steps['model']
    
    # 获取清洗后的特征
    features = cleaner.transform(df_subset)
    # 剥离人工分，扔进孤立森林
    model_cols = [col for col in features.columns if col != 'final_time_risk']
    preds = model.predict(features[model_cols])
    
    # 暂存模型结果在特征表中
    features['model_anomaly'] = preds
    return features

# ==================== 主控逻辑 ====================

st.title("🎯 高维行为探测与黑产监控大盘")

# 1. 后台静默拉取 3 周大池子数据
df_all = fetch_big_data()

# 2. 侧边栏 UI 构建
st.sidebar.header("🗓️ 数据切片与规则引擎")

# 获取池子里的极限日期
if not df_all.empty:
    min_date = df_all['date'].min()
    max_date = df_all['date'].max()
else:
    min_date = (datetime.now() - timedelta(days=7)).date()
    max_date = datetime.now().date()

# 内存滑块：选日期（控制进入模型的切片数据）
selected_dates = st.sidebar.slider(
    "选择分析时段 (极速内存切片)",
    min_value=min_date,
    max_value=max_date,
    value=(max_date - timedelta(days=3), max_date) # 默认看近3天
)

# 阈值滑块：选人工介入线（毫秒级响应，随意拖动）
manual_threshold = st.sidebar.slider(
    "强规则斩杀线 (final_time_risk)", 
    min_value=10.0, max_value=100.0, value=50.0, step=5.0
)

# 3. 实时内存切片
df_sliced = df_all[
    (df_all['date'] >= selected_dates[0]) & 
    (df_all['date'] <= selected_dates[1])
]

# 4. 业务宣判与展示
if df_sliced.empty:
    st.warning("⚠️ 所选日期范围内无数据。")
else:
    # 模型处理
    features = run_pipeline(df_sliced)
    
    if not features.empty:
        # --- 二次加工厂 (不带任何缓存，随手柄实时动态变化) ---
        features['is_model_bot'] = (features['model_anomaly'] == -1)
        features['is_rule_bot'] = (features['final_time_risk'] >= manual_threshold)
        # 最终合并：模型说坏的，或者规则强行枪毙的
        features['final_label'] = features['is_model_bot'] | features['is_rule_bot']
        
        # --- 核心数据大屏 ---
        total_users = len(features)
        total_bots = features['final_label'].sum()
        bot_ratio = (total_bots / total_users) * 100 if total_users > 0 else 0
        
        st.markdown(f"**当前时段切片原始日志量:** `{len(df_sliced):,}` 条 | **去重设备数:** `{total_users:,}`")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("分析设备总数", f"{total_users:,}")
        col2.metric("拦截黑产总数", f"{total_bots:,}", f"{bot_ratio:.2f}% 污染率", delta_color="inverse")
        col3.metric("模型底层抓取", int(features['is_model_bot'].sum()))
        col4.metric("规则实锤强杀", int(features['is_rule_bot'].sum()))
        
        st.divider()
        
        # --- 明细透视区 ---
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.write("#### 🚨 拦截归因分布")
            # 区分一下是谁杀的
            features['bot_type'] = '正常用户'
            features.loc[features['is_model_bot'], 'bot_type'] = '模型判定'
            features.loc[features['is_rule_bot'], 'bot_type'] = '规则强杀'
            features.loc[features['is_model_bot'] & features['is_rule_bot'], 'bot_type'] = '双重实锤'
            st.dataframe(features[features['bot_type'] != '正常用户']['bot_type'].value_counts(), use_container_width=True)
            
        with c2:
            st.write("#### 🥇 规则危险得分 TOP 排名 (按降序)")
            show_cols = ['bot_type', 'final_time_risk', 'cluster_size', 'total_pv']
            display_df = features[features['final_label']].sort_values('final_time_risk', ascending=False)
            st.dataframe(display_df[[col for col in show_cols if col in display_df.columns]].head(15), use_container_width=True)
        
        # --- 导出结果 ---
        bot_list = features[features['final_label']].reset_index()
        csv_data = bot_list.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 导出异常设备完整名单 (CSV)",
            data=csv_data,
            file_name=f'bot_list_{selected_dates[0]}_to_{selected_dates[1]}.csv',
            mime='text/csv'
        )
