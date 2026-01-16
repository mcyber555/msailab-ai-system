import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- 1. 言語設定辞書 (日本語/English) ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha v1.7",
        "status": "📊 運営主体：合同会社MS AI Lab | 統合分析ステータス: 2026/01/16 更新",
        "sidebar_head": "⚙️ 解析パラメータ",
        "sidebar_sub": "解析基準値を直接入力（0.1単位）",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "期待利回り (下限 %)",
        "max_payout": "許容配当性向 (上限 %)",
        "result_head": "📈 AI解析・スクリーニング結果",
        "col_ticker": "Ticker",
        "col_name": "銘柄名",
        "col_sector": "業界",
        "col_weather": "天気",
        "col_roe": "ROE(%)",
        "col_yield": "利回り(%)",
        "col_payout": "配当性向(%)",
        "col_price": "株価",
        "col_score": "AIスコア",
        "col_reason": "判定理由",
        "chart_title": "AI推奨度：上位銘柄比較",
        "footer_1_head": "**【組織概要】**",
        "footer_1_body": "法人名: 合同会社MS AI Lab  \n代表者: 代表取締役 [あなたの氏名]  \n設立: 2026年1月15日",
        "footer_2_head": "**【技術背景】**",
        "footer_2_body": "AI Model: Random Forest  \nロジック: 財務三表+景況感の多角解析  \n実績: 20年の運用知見をシステム化",
        "footer_3_head": "**【事業内容】**",
        "footer_3_body": "AIスコアリングに基づく自己資金運用事業。増配可能性の高い銘柄への長期投資を最適化。",
        "warning": "※本システムは自己勘定取引専用であり、外部への投資助言等は一切行いません。",
        "weather_sun": "☀️好調", "weather_cloud": "☁️不透明", "weather_rain": "☔苦戦"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis Platform: MSAI-Alpha v1.7",
        "status": "📊 Entity: MS AI Lab LLC | System Status: Updated 2026/01/16",
        "sidebar_head": "⚙️ Algorithm Parameters",
        "sidebar_sub": "Input threshold values (Step: 0.1)",
        "min_roe": "Required ROE (Min %)",
        "min_yield": "Expected Yield (Min %)",
        "max_payout": "Allowable Payout (Max %)",
        "result_head": "📈 AI Analysis & Screening Results",
        "col_ticker": "Ticker",
        "col_name": "Name",
        "col_sector": "Sector",
        "col_weather": "Trend",
        "col_roe": "ROE(%)",
        "col_yield": "Yield(%)",
        "col_payout": "Payout(%)",
        "col_price": "Price",
        "col_score": "AI Score",
        "col_reason": "Analysis Note",
        "chart_title": "AI Recommendation: Top Equities",
        "footer_1_head": "**【Entity】**",
        "footer_1_body": "Name: MS AI Lab LLC  \nCEO: [Your Name]  \nFounded: Jan 15, 2026",
        "footer_2_head": "**【Technology】**",
        "footer_2_body": "AI Model: Random Forest  \nLogic: Multi-factor Financial Analysis  \nExpertise: 20+ years of market experience",
        "footer_3_head": "**【Business】**",
        "footer_3_body": "Proprietary trading based on AI scoring. Optimizing long-term investment in high-dividend stocks.",
        "warning": "Note: This system is for proprietary trading only and does not provide financial advice.",
        "weather_sun": "☀️Stable", "weather_cloud": "☁️Neutral", "weather_rain": "☔Risky"
    }
}

# --- 2. サイドバーでの言語切り替え ---
lang = st.sidebar.radio("🌐 Language / 言語選択", ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. データベース設定 ---
@st.cache_data
def get_integrated_database(current_lang):
    data = [
        {'Ticker': '8058.T', '銘柄名': 'Mitsubishi Corp', '業界': 'Wholesale', '天気': '☀️', 'ROE': 15.5, '利回り': 3.5, '配当性向': 25.0, '株価': 2860.0},
        {'Ticker': '8001.T', '銘柄名': 'ITOCHU', '業界': 'Wholesale', '天気': '☀️', 'ROE': 17.0, '利回り': 3.1, '配当性向': 28.0, '株価': 6620.0},
        {'Ticker': '2914.T', '銘柄名': 'JT', '業界': 'Food', '天気': '☁️', 'ROE': 16.2, '利回り': 6.2, '配当性向': 75.0, '株価': 4150.0},
        {'Ticker': '9513.T', '銘柄名': 'J-POWER', '業界': 'Energy', '天気': '☁️', 'ROE': 7.5, '利回り': 4.2, '配当性向': 30.0, '株価': 2450.0},
        {'Ticker': '9432.T', '銘柄名': 'NTT', '業界': 'Telecom', '天気': '☀️', 'ROE': 12.5, '利回り': 3.2, '配当性向': 35.0, '株価': 180.5},
        {'Ticker': '8306.T', '銘柄名': 'MUFG', '業界': 'Banking', '天気': '☀️', 'ROE': 8.5, '利回り': 3.8, '配当性向': 38.0, '株価': 1460.0},
        {'Ticker': '7203.T', '銘柄名': 'Toyota', '業界': 'Automotive', '天気': '☀️', 'ROE': 11.5, '利回り': 2.8, '配当性向': 30.0, '株価': 2650.0},
        {'Ticker': '9101.T', '銘柄名': 'NYK Line', '業界': 'Shipping', '天気': '☔', 'ROE': 12.0, '利回り': 5.1, '配当性向': 30.0, '株価': 4800.0},
    ]
    # (実際にはここへ35社分のデータを同様に追加)
    return pd.DataFrame(data)

# --- 4. AI解析ロジック ---
df = get_integrated_database(lang)
X = df[['ROE', '利回り', '配当性向']]
weather_map = {'☀️': 1.0, '☁️': 0.5, '☔': 0.0}
weather_val = df['天気'].map(weather_map)
y_target = (df['ROE'] * 0.4) + (df['利回り'] * 0.4) - (df['配当性向'] * 0.1) + (weather_val * 2.5)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y_target)
df['AIスコア'] = model.predict(X)

def generate_rationale(row, current_lang):
    if current_lang == "日本語":
        reasons = []
        if row['ROE'] >= 14.0: reasons.append("圧倒的収益力")
        if row['利回り'] >= 4.0: reasons.append("高還元性")
        if row['天気'] == '☀️': reasons.append("景況良好")
        return "＋".join(reasons) if reasons else "安定評価"
    else:
        reasons = []
        if row['ROE'] >= 14.0: reasons.append("High Profitability")
        if row['利回り'] >= 4.0: reasons.append("High Dividend")
        if row['天気'] == '☀️': reasons.append("Good Outlook")
        return " & ".join(reasons) if reasons else "Stable"

df['判定理由'] = df.apply(lambda r: generate_rationale(r, lang), axis=1)

# --- 5. UI表示 ---
st.title(t["title"])
st.write(t["status"])
st.markdown("---")

# パラメータ入力
min_roe = st.sidebar.number_input(t["min_roe"], value=7.0, step=0.1, format="%.1f")
min_yield = st.sidebar.number_input(t["min_yield"], value=3.5, step=0.1, format="%.1f")
max_payout = st.sidebar.number_input(t["max_payout"], value=90.0, step=0.1, format="%.1f")

# フィルタリング
final_df = df[
    (df['ROE'] >= min_roe) & (df['利回り'] >= min_yield) & (df['配当性向'] <= max_payout)
].sort_values(by='AIスコア', ascending=False)

st.subheader(f"{t['result_head']} ({len(final_df)})")

# テーブル表示
st.dataframe(
    final_df[['Ticker', '銘柄名', '業界', '天気', 'ROE', '利回り', '配当性向', '株価', 'AIスコア', '判定理由']]
    .rename(columns={
        '銘柄名': t['col_name'], '業界': t['col_sector'], '天気': t['col_weather'],
        '利回り': t['col_yield'], '配当性向': t['col_payout'], '株価': t['col_price'],
        'AIスコア': t['col_score'], '判定理由': t['col_reason']
    })
    .style.background_gradient(subset=[t['col_score']], cmap='Greens')
    .format({t['col_roe']: '{:.1f}', t['col_yield']: '{:.1f}', t['col_payout']: '{:.1f}', t['col_price']: '¥{:,.1f}', t['col_score']: '{:.1f}'}),
    height=500, use_container_width=True, hide_index=True
)

# グラフ
if not final_df.empty:
    fig = px.bar(final_df.head(10), x='Ticker', y='AIスコア', color='利回り', title=t["chart_title"], text_auto='.1f')
    st.plotly_chart(fig, use_container_width=True)

# --- 6. フッター (会社情報) ---
st.markdown("---")
st.subheader("🏢 MS AI Lab Information")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"{t['footer_1_head']}\n\n{t['footer_1_body']}")
with c2:
    st.markdown(f"{t['footer_2_head']}\n\n{t['footer_2_body']}")
with c3:
    st.markdown(f"{t['footer_3_head']}\n\n{t['footer_3_body']}")
st.caption(t["warning"])
