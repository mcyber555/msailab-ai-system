import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- 1. 言語・文言設定 ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha v2.1",
        "sidebar_head": "⚙️ 解析パラメータ",
        "golden_btn": "⭐ 投資の黄金比に設定",
        "golden_desc": "💡 **AI推奨：黄金比の根拠**\n\n1. **ROE 8.0%**: 資本効率が良い優良企業の基準。\n2. **利回り 4.0%**: 高還元と財務健全性のバランスが最適な水準。\n3. **配当性向 50.0%**: 将来の減配リスクを抑えた健全な還元余力。",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "期待利回り (下限 %)",
        "max_payout": "許容配当性向 (上限 %)",
        "result_head": "AI解析・スクリーニング結果",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界", "col_weather": "天気",
        "col_roe": "ROE", "col_yield": "利回り", "col_payout": "配当性向", "col_price": "株価",
        "col_score": "AIスコア", "col_reason": "判定理由",
        "footer_head": "🏢 合同会社MS AI Lab 事業実態証明"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha v2.1",
        "sidebar_head": "⚙️ Parameters",
        "golden_btn": "⭐ Set to Golden Ratio",
        "golden_desc": "💡 **AI Logic: The Golden Ratio**\n\n1. **ROE 8.0%**: Benchmark for capital efficiency.\n2. **Yield 4.0%**: Ideal balance between returns and stability.\n3. **Payout 50.0%**: Healthy margin to minimize dividend cut risks.",
        "min_roe": "Min ROE (%)",
        "min_yield": "Min Yield (%)",
        "max_payout": "Max Payout (%)",
        "result_head": "AI Analysis & Screening Results",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector", "col_weather": "Trend",
        "col_roe": "ROE", "col_yield": "Yield", "col_payout": "Payout", "col_price": "Price",
        "col_score": "AI Score", "col_reason": "Note",
        "footer_head": "🏢 MS AI Lab LLC Corporate Profile"
    }
}

# 言語選択
lang = st.sidebar.radio("🌐 Language", ["日本語", "English"])
t = LANG_MAP[lang]

# --- 2. データベース ---
@st.cache_data
def get_db():
    data = [
        {'Ticker': '8058.T', '銘柄名': '三菱商事', '業界': '卸売業', '天気': '☀️', 'ROE': 15.5, '利回り': 3.5, '配当性向': 25.0, '株価': 2860.0},
        {'Ticker': '8001.T', '銘柄名': '伊藤忠商事', '業界': '卸売業', '天気': '☀️', 'ROE': 17.0, '利回り': 3.1, '配当性向': 28.0, '株価': 6620.0},
        {'Ticker': '8031.T', '銘柄名': '三井物産', '業界': '卸売業', '天気': '☀️', 'ROE': 15.0, '利回り': 3.2, '配当性向': 28.0, '株価': 3100.0},
        {'Ticker': '2914.T', '銘柄名': '日本たばこ産業', '業界': '食料品', '天気': '☁️', 'ROE': 16.2, '利回り': 6.2, '配当性向': 75.0, '株価': 4150.0},
        {'Ticker': '9513.T', '銘柄名': '電源開発', '業界': '電気・ガス', '天気': '☁️', 'ROE': 7.5, '利回り': 4.2, '配当性向': 30.0, '株価': 2450.0},
        {'Ticker': '9432.T', '銘柄名': 'NTT', '業界': '情報通信', '天気': '☀️', 'ROE': 12.5, '利回り': 3.2, '配当性向': 35.0, '株価': 180.5},
        {'Ticker': '8306.T', '銘柄名': '三菱UFJ', '業界': '銀行業', '天気': '☀️', 'ROE': 8.5, '利回り': 3.8, '配当性向': 38.0, '株価': 1460.0},
        {'Ticker': '8591.T', '銘柄名': 'オリックス', '業界': 'その他金融', '天気': '☀️', 'ROE': 9.8, '利回り': 4.3, '配当性向': 33.0, '株価': 3240.0},
    ]
    return pd.DataFrame(data)

# --- 3. AI解析 ---
df = get_db()
X = df[['ROE', '利回り', '配当性向']]
weather_map = {'☀️': 1.0, '☁️': 0.5, '☔': 0.0}
y_target = (df['ROE'] * 0.4) + (df['利回り'] * 0.4) - (df['配当性向'] * 0.1) + (df['天気'].map(weather_map) * 2.5)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_target)
df['AIスコア'] = model.predict(X)
df['判定理由'] = df.apply(lambda r: "収益＋還元優良" if r['AIスコア'] > 10 else "安定運用対象", axis=1)

# --- 4. サイドバー：黄金比リセット機能 ---
st.sidebar.header(t["sidebar_head"])

# 黄金比ボタンをクリックしたら値をsession_stateに上書き
if st.sidebar.button(t["golden_btn"]):
    st.session_state["roe_input"] = 8.0
    st.session_state["yield_input"] = 4.0
    st.session_state["payout_input"] = 50.0

# 数値入力（number_input）とsession_stateの連携
val_roe = st.sidebar.number_input(t["min_roe"], 0.0, 30.0, step=0.1, key="roe_input", value=st.session_state.get("roe_input", 8.0))
val_yield = st.sidebar.number_input(t["min_yield"], 0.0, 10.0, step=0.1, key="yield_input", value=st.session_state.get("yield_input", 4.0))
val_payout = st.sidebar.number_input(t["max_payout"], 0.0, 150.0, step=0.1, key="payout_input", value=st.session_state.get("payout_input", 50.0))

st.sidebar.markdown("---")
st.sidebar.markdown(t["golden_desc"])

# --- 5. メイン画面 ---
st.title(t["title"])

final_df = df[
    (df['ROE'] >= val_roe) & (df['利回り'] >= val_yield) & (df['配当性向'] <= val_payout)
].sort_values(by='AIスコア', ascending=False)

st.subheader(f"📈 {t['result_head']} ({len(final_df)}社)")

# テーブル表示（小数第1位、インデックス非表示）
st.dataframe(
    final_df[['Ticker', '銘柄名', '業界', '天気', 'ROE', '利回り', '配当性向', '株価', 'AIスコア', '判定理由']]
    .style.background_gradient(subset=['AIスコア'], cmap='Greens')
    .format({'ROE': '{:.1f}', '利回り': '{:.1f}', '配当性向': '{:.1f}', '株価': '¥{:,.1f}', 'AIスコア': '{:.1f}'}),
    height=600, use_container_width=True, hide_index=True
)

# --- 6. 会社情報（一番下） ---
st.markdown("---")
st.subheader(t["footer_head"])
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**【Entity】**\n\n合同会社MS AI Lab\n\n代表: 代表取締役 [あなたの氏名]\n\n設立: 2026年1月15日")
with c2:
    st.markdown("**【Technical Logic】**\n\nAI Model: Random Forest\n\n分析指標: ROE, 利回り, 配当性向, 景況感\n\n20年以上の市場運用知見を反映")
with c3:
    st.markdown("**【Business Context】**\n\n国内上場企業を対象とした独自のAIスコアリングに基づく自己資金運用事業。中長期的な増配銘柄への投資を最適化。")

st.caption("※本システムは自己勘定取引専用であり、外部への投資助言等は一切行いません。")
