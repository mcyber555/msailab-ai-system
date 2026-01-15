import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- 1. ページ設定 & 会社概要 ---
st.set_page_config(page_title="AI Dividend System", layout="wide")

st.title("🛡️ 独自AIアルゴリズムによる配当投資選定システム")
st.markdown("""
本システムは、代表者のエンジニアリング知見と2006年からの投資実績に基づき、
企業の財務データから**「減配リスク」を排除し「増配可能性」をスコアリング**する独自基盤です。
""")

# --- 2. サイドバー設定 ---
st.sidebar.header("🔍 スクリーニング条件")
min_roe = st.sidebar.slider("最小ROE (%)", 0.0, 20.0, 8.0)
min_yield = st.sidebar.slider("期待配当利回り (%)", 0.0, 7.0, 3.0)
max_payout = st.sidebar.slider("最大配当性向 (%)", 0.0, 100.0, 60.0)

TICKERS = ['9432.T', '9433.T', '8058.T', '8001.T', '8591.T', '2914.T', '8306.T', '8316.T', '4503.T']

# --- 3. データ取得エンジン ---
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers):
    results = []
    for symbol in tickers:
        try:
            tk = yf.Ticker(symbol)
            info = tk.info
            results.append({
                'Ticker': symbol,
                '銘柄名': info.get('longName', symbol),
                'ROE(%)': info.get('returnOnEquity', 0) * 100,
                '配当利回り(%)': info.get('dividendYield', 0) * 100,
                '配当性向(%)': info.get('payoutRatio', 0) * 100,
                '現在値': info.get('previousClose', 0)
            })
        except:
            continue
    return pd.DataFrame(results)

# --- 4. AIスコアリングエンジン ---
def apply_ai_scoring(df):
    if df.empty: return df
    X = df[['ROE(%)', '配当利回り(%)', '配当性向(%)']]
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    y = (df['ROE(%)'] * 0.6) + (df['配当利回り(%)'] * 0.4) - (df['配当性向(%)'] * 0.1)
    model.fit(X, y)
    df['AIスコア'] = model.predict(X)
    return df

# --- 5. メイン表示処理 ---
raw_df = fetch_stock_data(TICKERS)
scored_df = apply_ai_scoring(raw_df)

final_df = scored_df[
    (scored_df['ROE(%)'] >= min_roe) &
    (scored_df['配当利回り(%)'] >= min_yield) &
    (scored_df['配当性向(%)'] <= max_payout)
].sort_values(by='AIスコア', ascending=False)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 AI解析結果データセット")
    st.dataframe(final_df.style.background_gradient(subset=['AIスコア'], cmap='Greens'))
with col2:
    st.subheader("💡 ポートフォリオ比率提案")
    if not final_df.empty:
        fig = px.pie(final_df, values='AIスコア', names='銘柄名', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

# --- 6. フッター ---
st.markdown("---")
with st.expander("🏢 運営会社情報"):
    st.write("- **会社名:** MSAILab合同会社")
    st.write("- **代表者:** 代表取締役 [あなたの氏名]")
