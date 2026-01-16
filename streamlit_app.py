import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="MS AI Lab AI Alpha", layout="wide")

# 解析日の自動取得（前日）
target_date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')

# --- 2. 言語辞書（項目の並び替えと名称変更を反映） ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha v3.5",
        "status": f"📊 運営主体：合同会社MS AI Lab | 解析基準日: {target_date}",
        "sidebar_head": "⚙️ 解析パラメータ",
        "lang_label": "🌐 言語切替 / Language",
        "golden_btn": "⭐️黄金比にする",
        "golden_desc": "💡 **AI推奨：黄金比の根拠**\n\n1. **ROE 8.0%**: 資本効率が良い企業の基準。\n2. **利回り 4.0%**: 還元と健全性の最適バランス。\n3. **配当性向 50.0%**: 減配リスクを抑えた健全な水準。",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "配当金利回り (下限 %)",
        "max_payout": "許容配当性向 (上限 %)",
        "result_head": "プライム市場 厳選ユニバース解析結果",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界", "col_weather": "天気",
        "col_yield": "配当金利回り(%)", "col_roe": "ROE(%)", "col_payout": "配当性向(%)", 
        "col_price": "終値", "col_score": "AIスコア", "col_reason": "AI選定理由",
        "footer_1_head": "**【組織概要】**",
        "footer_1_body": "法人名: 合同会社MS AI Lab  \n代表者: 代表取締役 [あなたの氏名]  \n設立: 2026年1月15日",
        "footer_2_head": "**【技術背景】**",
        "footer_2_body": "AI Model: Random Forest  \n手法: 財務指標の多角解析  \n実績: 20年の市場知見を反映",
        "footer_3_head": "**【事業内容】**",
        "footer_3_body": "独自AIスコアリングに基づく資産運用。増配可能性の高い銘柄への長期投資を最適化。",
        "warning": "※本システムは自己勘定取引専用であり、外部への投資助言等は行いません。"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha v3.5",
        "status": f"📊 Entity: MS AI Lab LLC | Analysis Date: {target_date}",
        "sidebar_head": "⚙️ Parameters",
        "lang_label": "🌐 Language Selection",
        "golden_btn": "⭐️Set to Golden Ratio",
        "golden_desc": "💡 **AI Logic: The Golden Ratio**\n\n1. **ROE 8.0%**: Efficiency benchmark.\n2. **Yield 4.0%**: Optimal dividend balance.\n3. **Payout 50.0%**: Healthy safety margin.",
        "min_roe": "Required ROE (Min %)",
        "min_yield": "Div. Yield (Min %)",
        "max_payout": "Allowable Payout (Max %)",
        "result_head": "AI Analysis Results",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector", "col_weather": "Trend",
        "col_yield": "Div. Yield(%)", "col_roe": "ROE(%)", "col_payout": "Payout(%)", 
        "col_price": "Price", "col_score": "AI Score", "col_reason": "AI Reason",
        "footer_1_head": "**【Organization】**",
        "footer_1_body": "Name: MS AI Lab LLC  \nCEO: [Your Name]  \nFounded: Jan 15, 2026",
        "footer_2_head": "**【Technology】**",
        "footer_2_body": "AI Model: Random Forest  \nLogic: Quantitative Financial Analysis",
        "footer_3_head": "**【Business】**",
        "footer_3_body": "Proprietary trading based on AI scoring. Optimizing long-term investment.",
        "warning": "Note: Proprietary trading only. No financial advice provided."
    }
}

lang = st.sidebar.radio(LANG_MAP["日本語"]["lang_label"], ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. 監視銘柄（電力・銀行・証券を完全網羅 / ダミー名排除） ---
@st.cache_data
def get_master_data(current_lang):
    stocks = [
        # メガバンク・証券・金融
        {'T': '8306.T', 'N': '三菱UFJ', 'NE': 'MUFG', 'S': '銀行/Bank', 'W': '☀️', 'R': 8.5, 'Y': 3.8, 'P': 38.0, 'Pr': 1460},
        {'T': '8316.T', 'N': '三井住友', 'NE': 'SMFG', 'S': '銀行/Bank', 'W': '☀️', 'R': 8.0, 'Y': 4.0, 'P': 40.0, 'Pr': 8850},
        {'T': '8411.T', 'N': 'みずほFG', 'NE': 'Mizuho', 'S': '銀行/Bank', 'W': '☀️', 'R': 7.2, 'Y': 3.7, 'P': 40.0, 'Pr': 3150},
        {'T': '8604.T', 'N': '野村HD', 'NE': 'Nomura', 'S': '証券/Securities', 'W': '☁️', 'R': 5.2, 'Y': 3.2, 'P': 50.0, 'Pr': 850},
        {'T': '8601.T', 'N': '大和証券', 'NE': 'Daiwa', 'S': '証券/Securities', 'W': '☁️', 'R': 6.5, 'Y': 3.5, 'P': 55.0, 'Pr': 1050},
        {'T': '8591.T', 'N': 'オリックス', 'NE': 'ORIX', 'S': '金融/Finance', 'W': '☀️', 'R': 9.8, 'Y': 4.3, 'P': 33.0, 'Pr': 3240},
        # 電力・エネルギー
        {'T': '9513.T', 'N': '電源開発', 'NE': 'J-POWER', 'S': '電力/Utility', 'W': '☁️', 'R': 7.5, 'Y': 4.2, 'P': 30.0, 'Pr': 2450},
        {'T': '9503.T', 'N': '関西電力', 'NE': 'Kansai Elec', 'S': '電力/Utility', 'W': '☀️', 'R': 9.0, 'Y': 3.1, 'P': 25.0, 'Pr': 2100},
        {'T': '9502.T', 'N': '中部電力', 'NE': 'Chubu Elec', 'S': '電力/Utility', 'W': '☀️', 'R': 8.5, 'Y': 3.2, 'P': 30.0, 'Pr': 1950},
        {'T': '1605.T', 'N': 'INPEX', 'NE': 'INPEX', 'S': '鉱業/Mining', 'W': '☀️', 'R': 10.2, 'Y': 4.0, 'P': 40.0, 'Pr': 2100},
        # 主要プライム銘柄
        {'T': '8058.T', 'N': '三菱商事', 'NE': 'Mitsubishi Corp', 'S': '卸売/Trading', 'W': '☀️', 'R': 15.5, 'Y': 3.5, 'P': 25.0, 'Pr': 2860},
        {'T': '2914.T', 'N': '日本たばこ', 'NE': 'JT', 'S': '食料品/Food', 'W': '☁️', 'R': 16.2, 'Y': 6.2, 'P': 75.0, 'Pr': 4150},
        {'T': '7203.T', 'N': 'トヨタ', 'NE': 'Toyota', 'S': '自動車/Auto', 'W': '☀️', 'R': 11.5, 'Y': 2.8, 'P': 30.0, 'Pr': 2650},
        {'T': '9432.T', 'N': 'NTT', 'NE': 'NTT', 'S': '通信/Telecom', 'W': '☀️', 'R': 12.5, 'Y': 3.2, 'P': 35.0, 'Pr': 180},
    ]
    # その他を埋めるための自動補完（ダミー名称は絶対に使わない）
    df = pd.DataFrame(stocks)
    if current_lang == "English":
        df['N'] = df['NE']
    return df

# --- 4. 解析・AIコメント生成ロジック（短文化） ---
def generate_ai_comment(row, current_lang):
    """銀行審査向けに要点を端的に述べる"""
    if current_lang == "English":
        if row['Score'] >= 90: return "Prime: High profit & return."
        if row['Yield'] >= 4: return "Stable: High yield focus."
        return "Target: Balanced growth."
    else:
        if row['Score'] >= 90: return "最優先：高効率・好還元"
        if row['Yield'] >= 4: return "安定：高配当重視"
        return "対象：バランス型成長"

@st.cache_data(ttl=3600)
def fetch_and_score(df, current_lang):
    results = []
    for _, row in df.iterrows():
        try:
            tk = yf.Ticker(row['T'])
            t_info = tk.info
            # yfinanceからの取得値を正規化（0.17 -> 17.0%）
            roe = np.round(t_info.get('returnOnEquity', row['R']/100) * 100, 1)
            yld = np.round(t_info.get('dividendYield', row['Y']/100) * 100, 1)
            payout = np.round(t_info.get('payoutRatio', row['P']/100) * 100, 1)
            price = t_info.get('previousClose', row['Pr'])
            
            # 異常値チェック（スクショ4の300%超え対策）
            if yld > 20: yld = row['Y'] # API異常時はマスターデータを使用
            
            results.append({
                'Ticker': row['T'], 'Name': row['N'], 'Sector': row['S'], 'Trend': row['W'],
                'Yield': yld, 'ROE': roe, 'Payout': payout, 'Price': price
            })
        except:
            results.append({
                'Ticker': row['T'], 'Name': row['N'], 'Sector': row['S'], 'Trend': row['W'],
                'Yield': row['Y'], 'ROE': row['R'], 'Payout': row['P'], 'Price': row['Pr']
            })
    
    res_df = pd.DataFrame(results)
    # AIスコアリング
    X = res_df[['ROE', 'Yield', 'Payout']]
    w_map = {'☀️': 1.0, '☁️': 0.5, '☔': 0.0}
    y_raw = (res_df['ROE'] * 0.4) + (res_df['Yield'] * 0.4) - (res_df['Payout'] * 0.1) + (res_df['Trend'].map(w_map) * 3.0)
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_raw)
    raw_scores = model.predict(X)
    res_df['Score'] = np.round((raw_scores / raw_scores.max()) * 100, 1)
    # 端的なコメント付与
    res_df['Note'] = res_df.apply(lambda r: generate_ai_comment(r, current_lang), axis=1)
    return res_df

with st.spinner('Analyzing...'):
    analyzed_df = fetch_and_score(get_master_data(lang), lang)

# --- 5. サイドバー UI ---
st.sidebar.header(t["sidebar_head"])

if st.sidebar.button(t["golden_btn"]):
    st.session_state["roe_v"] = 8.0
    st.session_state["yield_v"] = 4.0
    st.session_state["payout_v"] = 50.0

v_roe = st.sidebar.slider(t["min_roe"], 0.0, 30.0, st.session_state.get("roe_v", 8.0), 0.1)
v_yield = st.sidebar.slider(t["min_yield"], 0.0, 10.0, st.session_state.get("yield_v", 4.0), 0.1)
v_payout = st.sidebar.slider(t["max_payout"], 0.0, 150.0, st.session_state.get("payout_v", 50.0), 0.1)

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

# テーブル表示 (利回りをROEの左へ、配当金（額）は削除)
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

# --- 7. 会社情報 ---
st.markdown("---")
st.subheader("🏢 MS AI Lab LLC Corporate Profile")
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f"{t['footer_1_head']}\n\n{t['footer_1_body']}")
with c2: st.markdown(f"{t['footer_2_head']}\n\n{t['footer_2_body']}")
with c3: st.markdown(f"{t['footer_3_head']}\n\n{t['footer_3_body']}")
st.caption(t["warning"])
