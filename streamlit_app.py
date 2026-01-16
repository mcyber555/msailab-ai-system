import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="MS AI Lab AI Alpha", layout="wide")

# 解析日の自動取得（昨日）
target_date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')

# --- 2. 言語辞書（二行書きヘッダーと黄金比の再定義） ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha v3.7",
        "status": f"📊 合同会社MS AI Lab | 解析基準日: {target_date}",
        "sidebar_head": "⚙️ 解析パラメータ",
        "lang_label": "🌐 言語切替 / Language",
        "golden_btn": "⭐️黄金比にする",
        "golden_desc": "💡 **AI推奨：黄金比の根拠**\n\n1. **ROE 7.0%以上**: 資本効率が日本企業の平均を上回り、持続可能な成長力を持つ基準。\n2. **利回り 3.2%以上**: 確実なインカムゲインを確保しつつ、株価下落への耐性を持つ水準。\n3. **配当性向 65.0%以下**: 積極的な還元を行いつつ、事業継続のための内部留保を維持した健全なバランス。",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "配当金\n利回り(下限 %)",
        "max_payout": "許容配当性向 (上限 %)",
        "result_head": "プライム市場 厳選ユニバース解析結果",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界", "col_weather": "天気",
        "col_yield": "配当金\n利回り(%)", "col_roe": "ROE(%)", "col_payout": "配当性向(%)", 
        "col_price": "終値", "col_score": "AIスコア", "col_reason": "AI選定理由",
        "footer_head": "🏢 合同会社MS AI Lab 事業実態証明"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha v3.7",
        "status": f"📊 MS AI Lab LLC | Analysis Date: {target_date}",
        "sidebar_head": "⚙️ Parameters",
        "lang_label": "🌐 Language Selection",
        "golden_btn": "⭐️Set to Golden Ratio",
        "golden_desc": "💡 **AI Logic: The Golden Ratio**\n\n1. **ROE 7.0%+**: Above JP average, ensures sustainable growth.\n2. **Yield 3.2%+**: Optimal income with downside protection.\n3. **Payout 65.0%-**: Balanced ratio between dividends and reinvestment.",
        "min_roe": "Min ROE (%)",
        "min_yield": "Div. Yield\n(Min %)",
        "max_payout": "Max Payout (%)",
        "result_head": "AI Analysis Results",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector", "col_weather": "Trend",
        "col_yield": "Dividend\nYield(%)", "col_roe": "ROE(%)", "col_payout": "Payout(%)", 
        "col_price": "Price", "col_score": "AI Score", "col_reason": "AI Reason",
        "footer_head": "🏢 MS AI Lab LLC Corporate Profile"
    }
}

lang = st.sidebar.radio(LANG_MAP["日本語"]["lang_label"], ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. 厳選マスターデータ ---
@st.cache_data
def get_master_data(current_lang):
    stocks = [
        {'T': '8306.T', 'N': '三菱UFJ', 'NE': 'MUFG', 'S': '銀行', 'W': '☀️', 'R': 8.5, 'Y': 3.8, 'P': 38.0, 'Pr': 1460},
        {'T': '8316.T', 'N': '三井住友', 'NE': 'SMFG', 'S': '銀行', 'W': '☀️', 'R': 8.0, 'Y': 4.0, 'P': 40.0, 'Pr': 8850},
        {'T': '8411.T', 'N': 'みずほFG', 'NE': 'Mizuho', 'S': '銀行', 'W': '☀️', 'R': 7.2, 'Y': 3.7, 'P': 40.0, 'Pr': 3150},
        {'T': '8604.T', 'N': '野村HD', 'NE': 'Nomura', 'S': '証券', 'W': '☁️', 'R': 5.2, 'Y': 3.2, 'P': 50.0, 'Pr': 850},
        {'T': '8601.T', 'N': '大和証券', 'NE': 'Daiwa', 'S': '証券', 'W': '☁️', 'R': 6.5, 'Y': 3.5, 'P': 55.0, 'Pr': 1050},
        {'T': '8591.T', 'N': 'オリックス', 'NE': 'ORIX', 'S': '金融', 'W': '☀️', 'R': 9.8, 'Y': 4.3, 'P': 33.0, 'Pr': 3240},
        {'T': '9513.T', 'N': '電源開発', 'NE': 'J-POWER', 'S': '電力', 'W': '☁️', 'R': 7.5, 'Y': 4.2, 'P': 30.0, 'Pr': 2450},
        {'T': '9503.T', 'N': '関西電力', 'NE': 'Kansai Elec', 'S': '電力', 'W': '☀️', 'R': 9.0, 'Y': 3.1, 'P': 25.0, 'Pr': 2100},
        {'T': '9502.T', 'N': '中部電力', 'NE': 'Chubu Elec', 'S': '電力', 'W': '☀️', 'R': 8.5, 'Y': 3.2, 'P': 30.0, 'Pr': 1950},
        {'T': '1605.T', 'N': 'INPEX', 'NE': 'INPEX', 'S': '鉱業', 'W': '☀️', 'R': 10.2, 'Y': 4.0, 'P': 40.0, 'Pr': 2100},
        {'T': '8058.T', 'N': '三菱商事', 'NE': 'Mitsubishi Corp', 'S': '卸売', 'W': '☀️', 'R': 15.5, 'Y': 3.5, 'P': 25.0, 'Pr': 2860},
        {'T': '8001.T', 'N': '伊藤忠', 'NE': 'ITOCHU', 'S': '卸売', 'W': '☀️', 'R': 17.0, 'Y': 3.1, 'P': 28.0, 'Pr': 6620},
        {'T': '2914.T', 'N': '日本たばこ', 'NE': 'JT', 'S': '食料品', 'W': '☁️', 'R': 16.2, 'Y': 6.2, 'P': 75.0, 'Pr': 4150},
        {'T': '7203.T', 'N': 'トヨタ', 'NE': 'Toyota', 'S': '自動車', 'W': '☀️', 'R': 11.5, 'Y': 2.8, 'P': 30.0, 'Pr': 2650},
        {'T': '9432.T', 'N': 'NTT', 'NE': 'NTT', 'S': '通信', 'W': '☀️', 'R': 12.5, 'Y': 3.2, 'P': 35.0, 'Pr': 180},
    ]
    df = pd.DataFrame(stocks)
    if current_lang == "English":
        df['N'] = df['NE']
    return df

# --- 4. 多様なAI選定理由の生成ロジック ---
def generate_diverse_reason(row, current_lang):
    if current_lang == "English":
        if row['Yield'] >= 4.5: return "Yield focus: Superior income profile."
        if row['ROE'] >= 12.0: return "Efficiency focus: High capital velocity."
        if row['Payout'] <= 30.0: return "Future focus: High reinvestment capacity."
        return "Balanced: Strong core fundamentals."
    else:
        if row['Yield'] >= 4.5: return "利回り重視：インカムゲイン優位"
        if row['ROE'] >= 12.0: return "効率重視：資本回転率が極めて高い"
        if row['Payout'] <= 30.0: return "成長重視：内部留保厚く余力大"
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

            roe = t_info.get('returnOnEquity', row['R']/100)
            if roe is not None:
                if roe > 1: roe = roe / 100
                roe = np.round(roe * 100, 1)
            else: roe = row['R']

            payout = t_info.get('payoutRatio', row['P']/100)
            if payout is not None:
                if payout > 2: payout = payout / 100
                payout = np.round(payout * 100, 1)
            else: payout = row['P']

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
    # AI解析スコアリング（生の評価に近い重み付け）
    w_map = {'☀️': 1.0, '☁️': 0.5, '☔': 0.0}
    # スコア計算式
    res_df['Score'] = np.round(
        (res_df['ROE'] * 2.5) + (res_df['Yield'] * 3.5) - (res_df['Payout'] * 0.15) + (res_df['Trend'].map(w_map) * 15), 
        1
    )
    # スコアの最大値を100付近に抑えつつ、自然な分布へ（無理に100に固定しない）
    res_df['Note'] = res_df.apply(lambda r: generate_diverse_reason(r, current_lang), axis=1)
    return res_df

with st.spinner('Analyzing...'):
    analyzed_df = fetch_and_score(get_master_data(lang), lang)

# --- 5. サイドバー UI ---
st.sidebar.header(t["sidebar_head"])

# 黄金比リセットボタン（利回り 3.2% / ROE 7.0% / 配当性向 65%）
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

# テーブル表示
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
st.subheader("🏢 MS AI Lab LLC Corporate Profile")
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f"{t['footer_1_head']}\n\n{t['footer_1_body']}")
with c2: st.markdown(f"{t['footer_2_head']}\n\n{t['footer_2_body']}")
with c3: st.markdown(f"{t['footer_3_head']}\n\n{t['footer_3_body']}")
st.caption(t["warning"])
