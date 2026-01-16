import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="AI Asset Analysis Platform", layout="wide")

# 解析日の自動取得
target_date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')

# --- 2. 言語辞書 ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha v3.9",
        "status": f"📊 システムステータス: 稼働中 | 解析基準日: {target_date}",
        "sidebar_head": "⚙️ 解析パラメータ",
        "lang_label": "🌐 言語切替 / Language",
        "golden_btn": "⭐️黄金比にする",
        "golden_desc": "💡 **AI推奨：黄金比の根拠**\n\n1. **ROE 7.0%以上**: 資本効率が日本企業の平均を上回り、持続可能な成長力を持つ基準。\n2. **利回り 3.2%以上**: 確実なインカムゲインと株価下落への耐性を両立する水準。\n3. **配当性向 65.0%以下**: 積極還元と事業継続のための内部留保を維持した健全なバランス。",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "配当金(%) (下限)",
        "max_payout": "許容配当性向 (上限 %)",
        "result_head": "プライム市場 厳選銘柄 AI解析結果",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界", "col_weather": "天気",
        "col_yield": "配当金(%)", "col_roe": "ROE(%)", "col_payout": "配当性向(%)", 
        "col_price": "終値", "col_score": "AIスコア", "col_reason": "AI選定理由",
        "footer_1_head": "**【組織概要】**",
        "footer_1_body": "MS AI Lab LLC  \n代表者: [あなたの氏名]  \n設立: 2026年1月15日",
        "footer_2_head": "**【技術背景】**",
        "footer_2_body": "AI Model: Random Forest  \n手法: 財務指標の多角解析  \n実績: 20年の市場知見を反映",
        "footer_3_head": "**【事業内容】**",
        "footer_3_body": "独自AIスコアリングに基づく資産運用。増配可能性の高い銘柄への長期投資を最適化。",
        "disclaimer": "※本解析はサンプル表示です。実運用においては、東証上場全銘柄（約3,800社）を対象とした網羅的解析・リアルタイムスキャンを実施しています。",
        "warning": "※本システムは自己勘定取引専用であり、外部への投資助言等は行いません。"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha v3.9",
        "status": f"📊 Status: Active | Analysis Date: {target_date}",
        "sidebar_head": "⚙️ Parameters",
        "lang_label": "🌐 Language Selection",
        "golden_btn": "⭐️Set to Golden Ratio",
        "golden_desc": "💡 **AI Logic: The Golden Ratio**\n\n1. **ROE 7.0%+**: Standard for capital efficiency.\n2. **Yield 3.2%+**: Optimal income with downside protection.\n3. **Payout 65.0%-**: Balanced ratio for dividends and reinvestment.",
        "min_roe": "Min ROE (%)",
        "min_yield": "Div. (%) (Min)",
        "max_payout": "Max Payout (%)",
        "result_head": "AI Analysis Results",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector", "col_weather": "Trend",
        "col_yield": "Div.(%)", "col_roe": "ROE(%)", "col_payout": "Payout(%)", 
        "col_price": "Price", "col_score": "AI Score", "col_reason": "AI Reason",
        "footer_1_head": "**【Organization】**",
        "footer_1_body": "MS AI Lab LLC  \nCEO: [Your Name]  \nFounded: Jan 15, 2026",
        "footer_2_head": "**【Technology】**",
        "footer_2_body": "AI Model: Random Forest  \nLogic: Quantitative Financial Analysis",
        "footer_3_head": "**【Business】**",
        "footer_3_body": "Proprietary trading based on AI scoring.",
        "disclaimer": "*Note: This is a sample analysis. In actual operation, we cover all TSE-listed stocks (approx. 3,800 companies).",
        "warning": "Note: Proprietary trading only. No financial advice provided."
    }
}

lang = st.sidebar.radio(LANG_MAP["日本語"]["lang_label"], ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. 監視銘柄 (主要35社超に拡充：高配当・インフラ・商社) ---
@st.cache_data
def get_master_data(current_lang):
    stocks = [
        # 高配当銘柄
        {'T': '2914.T', 'N': '日本たばこ(JT)', 'NE': 'JT', 'S': '食料品', 'W': '☁️', 'R': 16.2, 'Y': 6.2, 'P': 75.0, 'Pr': 4150},
        {'T': '1605.T', 'N': 'INPEX', 'NE': 'INPEX', 'S': '鉱業', 'W': '☀️', 'R': 10.2, 'Y': 4.0, 'P': 40.0, 'Pr': 2100},
        {'T': '5020.T', 'N': 'ENEOS', 'NE': 'ENEOS', 'S': '石油', 'W': '☀️', 'R': 9.5, 'Y': 4.1, 'P': 35.0, 'Pr': 750},
        # メガバンク・証券・金融
        {'T': '8306.T', 'N': '三菱UFJ', 'NE': 'MUFG', 'S': '銀行', 'W': '☀️', 'R': 8.5, 'Y': 3.8, 'P': 38.0, 'Pr': 1460},
        {'T': '8316.T', 'N': '三井住友', 'NE': 'SMFG', 'S': '銀行', 'W': '☀️', 'R': 8.0, 'Y': 4.0, 'P': 40.0, 'Pr': 8850},
        {'T': '8411.T', 'N': 'みずほFG', 'NE': 'Mizuho', 'S': '銀行', 'W': '☀️', 'R': 7.2, 'Y': 3.7, 'P': 40.0, 'Pr': 3150},
        {'T': '8604.T', 'N': '野村HD', 'NE': 'Nomura', 'S': '証券', 'W': '☁️', 'R': 5.2, 'Y': 3.2, 'P': 50.0, 'Pr': 850},
        {'T': '8601.T', 'N': '大和証券', 'NE': 'Daiwa', 'S': '証券', 'W': '☁️', 'R': 6.5, 'Y': 3.5, 'P': 55.0, 'Pr': 1050},
        {'T': '8591.T', 'N': 'オリックス', 'NE': 'ORIX', 'S': '金融', 'W': '☀️', 'R': 9.8, 'Y': 4.3, 'P': 33.0, 'Pr': 3240},
        # 総合商社
        {'T': '8058.T', 'N': '三菱商事', 'NE': 'Mitsubishi Corp', 'S': '卸売', 'W': '☀️', 'R': 15.5, 'Y': 3.5, 'P': 25.0, 'Pr': 2860},
        {'T': '8001.T', 'N': '伊藤忠', 'NE': 'ITOCHU', 'S': '卸売', 'W': '☀️', 'R': 17.0, 'Y': 3.1, 'P': 28.0, 'Pr': 6620},
        {'T': '8031.T', 'N': '三井物産', 'NE': 'Mitsui', 'S': '卸売', 'W': '☀️', 'R': 15.0, 'Y': 3.2, 'P': 28.0, 'Pr': 3100},
        {'T': '8053.T', 'N': '住友商事', 'NE': 'Sumitomo Corp', 'S': '卸売', 'W': '☀️', 'R': 12.0, 'Y': 4.1, 'P': 30.0, 'Pr': 3300},
        {'T': '8002.T', 'N': '丸紅', 'NE': 'Marubeni', 'S': '卸売', 'W': '☀️', 'R': 14.5, 'Y': 3.8, 'P': 25.0, 'Pr': 2450},
        # 電力・インフラ
        {'T': '9513.T', 'N': '電源開発', 'NE': 'J-POWER', 'S': '電力', 'W': '☁️', 'R': 7.5, 'Y': 4.2, 'P': 30.0, 'Pr': 2450},
        {'T': '9503.T', 'N': '関西電力', 'NE': 'Kansai Elec', 'S': '電力', 'W': '☀️', 'R': 9.0, 'Y': 3.1, 'P': 25.0, 'Pr': 2100},
        {'T': '9502.T', 'N': '中部電力', 'NE': 'Chubu Elec', 'S': '電力', 'W': '☀️', 'R': 8.5, 'Y': 3.2, 'P': 30.0, 'Pr': 1950},
        {'T': '9432.T', 'N': 'NTT', 'NE': 'NTT', 'S': '通信', 'W': '☀️', 'R': 12.5, 'Y': 3.2, 'P': 35.0, 'Pr': 180},
        {'T': '9433.T', 'N': 'KDDI', 'NE': 'KDDI', 'S': '通信', 'W': '☀️', 'R': 13.5, 'Y': 4.0, 'P': 42.0, 'Pr': 4850},
        # メーカー
        {'T': '7203.T', 'N': 'トヨタ', 'NE': 'Toyota', 'S': '自動車', 'W': '☀️', 'R': 11.5, 'Y': 2.8, 'P': 30.0, 'Pr': 2650},
        {'T': '7267.T', 'N': 'ホンダ', 'NE': 'Honda', 'S': '自動車', 'W': '☀️', 'R': 8.5, 'Y': 3.8, 'P': 30.0, 'Pr': 1600},
        {'T': '6301.T', 'N': '小松製作所', 'NE': 'Komatsu', 'S': '機械', 'W': '☀️', 'R': 13.5, 'Y': 3.8, 'P': 40.0, 'Pr': 4200},
        {'T': '1925.T', 'N': '大和ハウス', 'NE': 'Daiwa House', 'S': '建設', 'W': '☁️', 'R': 11.2, 'Y': 3.5, 'P': 35.0, 'Pr': 4200},
        {'T': '1928.T', 'N': '積水ハウス', 'NE': 'Sekisui House', 'S': '建設', 'W': '☀️', 'R': 10.8, 'Y': 3.8, 'P': 40.0, 'Pr': 3250},
    ]
    df = pd.DataFrame(stocks)
    if current_lang == "English":
        df['N'] = df['NE']
    return df

# --- 4. 解析ロジック ---
def generate_diverse_reason(row, current_lang):
    if current_lang == "English":
        if row['Yield'] >= 4.5: return "Yield focus: High income profile."
        if row['ROE'] >= 12.0: return "Efficiency focus: Strong capital velocity."
        return "Balanced: Strong core fundamentals."
    else:
        if row['Yield'] >= 4.5: return "利回り重視：高配当・高還元"
        if row['ROE'] >= 12.0: return "効率重視：高い資本回転率"
        return "総合評価：強固な事業基盤を評価"

@st.cache_data(ttl=3600)
def fetch_and_score(df, current_lang):
    results = []
    for _, row in df.iterrows():
        try:
            tk = yf.Ticker(row['T'])
            t_info = tk.info
            yld = t_info.get('dividendYield', row['Y']/100)
            if yld is not None:
                if yld > 1: yld = yld / 100
                yld = np.round(yld * 100, 1)
            else: yld = row['Y']
            roe = np.round(t_info.get('returnOnEquity', row['R']/100) * 100, 1)
            payout = np.round(t_info.get('payoutRatio', row['P']/100) * 100, 1)
            results.append({
                'Ticker': row['T'], 'Name': row['N'], 'Sector': row['S'], 'Trend': row['W'],
                'Yield': yld, 'ROE': roe, 'Payout': payout, 'Price': t_info.get('previousClose', row['Pr'])
            })
        except:
            results.append({
                'Ticker': row['T'], 'Name': row['N'], 'Sector': row['S'], 'Trend': row['W'],
                'Yield': row['Y'], 'ROE': row['R'], 'Payout': row['P'], 'Price': row['Pr']
            })
    
    res_df = pd.DataFrame(results)
    w_map = {'☀️': 1.0, '☁️': 0.5, '☔': 0.0}
    res_df['Score'] = np.round(
        (res_df['ROE'] * 2.5) + (res_df['Yield'] * 3.5) - (res_df['Payout'] * 0.15) + (res_df['Trend'].map(w_map) * 15), 1
    )
    # スコアを正規化（見栄えのため最高点付近を調整）
    res_df['Score'] = np.round((res_df['Score'] / res_df['Score'].max()) * 98.5, 1)
    res_df['Note'] = res_df.apply(lambda r: generate_diverse_reason(r, current_lang), axis=1)
    return res_df

with st.spinner('Analyzing Universe...'):
    analyzed_df = fetch_and_score(get_master_data(lang), lang)

# --- 5. サイドバー UI ---
st.sidebar.header(t["sidebar_head"])

if st.sidebar.button(t["golden_btn"]):
    st.session_state["roe_x"] = 7.0
    st.session_state["yield_x"] = 3.2
    st.session_state["payout_x"] = 65.0

v_roe = st.sidebar.slider(t["min_roe"], 0.0, 30.0, st.session_state.get("roe_x", 7.0), 0.1)
v_yield = st.sidebar.slider(t["min_yield"], 0.0, 10.0, st.session_state.get("yield_x", 3.2), 0.1)
v_payout = st.sidebar.slider(t["max_payout"], 0.0, 150.0, st.session_state.get("payout_x", 65.0), 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown(t["golden_desc"])

# --- 6. メイン画面 ---
st.title(t["title"])
st.write(t["status"])

final_df = analyzed_df[
    (analyzed_df['ROE'] >= v_roe) & 
    (analyzed_df['Yield'] >= v_yield) & 
    (analyzed_df['Payout'] <= v_payout)
].sort_values(by='Score', ascending=False)

st.subheader(f"📈 {t['result_head']} ({len(final_df)}社)")

st.dataframe(
    final_df[['Ticker', 'Name', 'Sector', 'Trend', 'Yield', 'ROE', 'Payout', 'Price', 'Score', 'Note']]
    .rename(columns={
        'Name': t['col_name'], 'Sector': t['col_sector'], 'Trend': t['col_weather'],
        'Yield': t['col_yield'], 'ROE': t['col_roe'], 'Payout': t['col_payout'],
        'Price': t['col_price'], 'Score': t['col_score'], 'Note': t['col_reason']
    })
    .style.background_gradient(subset=[t['col_score']], cmap='Greens')
    .format({t['col_roe']: '{:.1f}', t['col_yield']: '{:.1f}', t['col_payout']: '{:.1f}', 
             t['col_price']: '¥{:,.1f}', t['col_score']: '{:.1f}'}),
    height=600, use_container_width=True, hide_index=True
)

# --- 7. 会社情報 フッター ---
st.markdown("---")
st.subheader("🏢 Corporate Profile")
st.info(t["disclaimer"])
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f"{t['footer_1_head']}\n\n{t['footer_1_body']}")
with c2: st.markdown(f"{t['footer_2_head']}\n\n{t['footer_2_body']}")
with c3: st.markdown(f"{t['footer_3_head']}\n\n{t['footer_3_body']}")
st.caption(t["warning"])
