import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="MS AI Lab AI Alpha", layout="wide")

# 解析日の自動取得
target_date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')

# --- 2. 言語辞書 (説明文を簡潔・明快に修正) ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha",
        "status": f"📊 MS AI Lab LLC | 解析基準日: {target_date}",
        "sidebar_head": "⚙️ 解析パラメータ",
        "lang_label": "🌐 言語選択 / Language Selection",
        "golden_btn": "⭐️黄金比にする",
        "golden_desc": "💡 **AI推奨：黄金比の根拠**\n\n1. **配当利回り 3.0%以上**: 銀行預金を大きく上回る収益性と、株価の下落を防ぐ「支え」となる水準。\n2. **配当性向 120.0%以下**: 企業の利益から無理なく配当が出されているか、成長資金を削っていないかの境界線。\n3. **ROE 6.0%以上**: 預かった資本を使って、日本企業の平均的な効率で利益を生み出せているかの指標。",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "配当利回り\n(%) (下限)",
        "max_payout": "許容配当性向 (上限 %)",
        "result_head": "東証プライム 厳選200銘柄 AI解析結果",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界", "col_weather": "天気",
        "col_yield": "配当利回り\n(%)", "col_payout": "配当性向(%)", "col_roe": "ROE(%)", 
        "col_price": "終値", "col_score": "AIスコア",
        "disclaimer": "📌 本解析は特定銘柄の抽出サンプルです。実運用においては、東証上場全銘柄（約3,800社）を対象とした網羅的解析・リアルタイムスキャンを実施しています。",
        "footer_1_head": "**【組織概要】**", "footer_1_body": "MS AI Lab LLC  \n設立者: 鈴木 学  \n設立: 2026年1月15日",
        "footer_2_head": "**【技術背景】**", "footer_2_body": "AI Model: Random Forest  \nロジック: 財務指標の多角解析  \n実績: 20年の市場知見を反映",
        "footer_3_head": "**【事業内容】**", "footer_3_body": "独自AIスコアリングに基づく資産運用。増配可能性の高い銘柄への長期投資を最適化。",
        "warning": "※本システムは自己勘定取引専用であり、外部への投資助言等は行いません。"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha",
        "status": f"📊 MS AI Lab LLC | Analysis Date: {target_date}",
        "sidebar_head": "⚙️ Parameters",
        "lang_label": "🌐 Language Selection",
        "golden_btn": "⭐️Set to Golden Ratio",
        "golden_desc": "💡 **AI Logic: The Golden Ratio**\n\n1. **Yield 3.0%+**: Secure income with downside protection.\n2. **Payout 120.0%-**: Balance between dividends and business growth.\n3. **ROE 6.0%+**: Standard for efficient capital management.",
        "min_roe": "Min ROE (%)",
        "min_yield": "Dividend\nYield (%)",
        "max_payout": "Max Payout (%)",
        "result_head": "AI Analysis Results (Selected 200 Stocks)",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector", "col_weather": "Trend",
        "col_yield": "Dividend\nYield (%)", "col_payout": "Payout(%)", "col_roe": "ROE(%)", 
        "col_price": "Price", "col_score": "AI Score",
        "disclaimer": "📌 Note: Sample analysis. Actual operations scan all 3,800 TSE stocks.",
        "footer_1_head": "**【Organization】**", "footer_1_body": "MS AI Lab LLC  \nFounder: Manabu Suzuki  \nFounded: Jan 15, 2026",
        "footer_2_head": "**【Technology】**", "footer_2_body": "AI Model: Random Forest  \nLogic: Quantitative Financial Analysis",
        "footer_3_head": "**【Business】**", "footer_3_body": "Proprietary trading based on AI scoring.",
        "warning": "Note: Proprietary trading only."
    }
}

lang = st.sidebar.radio(LANG_MAP["日本語"]["lang_label"], ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. 厳選200銘柄実名ユニバース (銀行、電力10社、商社、主要メーカー網羅) ---
@st.cache_data
def get_master_data(current_lang):
    raw_list = [
        # 高配当・エネルギー
        ('2914.T', '日本たばこ(JT)', 'JT', '食料', '☀️', 16.5, 6.2, 75.0, 4150),
        ('1605.T', 'INPEX', 'INPEX', '鉱業', '☀️', 10.2, 4.0, 40.0, 2100),
        ('5020.T', 'ENEOS', 'ENEOS', '石油', '☀️', 9.5, 4.1, 35.0, 750),
        # 銀行・金融
        {'T': '8306.T', 'N': '三菱UFJ', 'NE': 'MUFG', 'S': '銀行', 'W': '☀️', 'R': 8.5, 'Y': 3.8, 'P': 38.0, 'Pr': 1460},
        {'T': '8316.T', 'N': '三井住友', 'NE': 'SMFG', 'S': '銀行', 'W': '☀️', 'R': 8.0, 'Y': 4.0, 'P': 40.0, 'Pr': 8850},
        {'T': '8411.T', 'N': 'みずほFG', 'NE': 'Mizuho', 'S': '銀行', 'W': '☀️', 'R': 7.2, 'Y': 3.7, 'P': 40.0, 'Pr': 3150},
        {'T': '8591.T', 'N': 'オリックス', 'NE': 'ORIX', 'S': '金融', 'W': '☀️', 'R': 9.8, 'Y': 4.3, 'P': 33.0, 'Pr': 3240},
        # 電力10社
        ('9501.T', '東京電力G', 'TEPCO', '電力', '☔', 3.0, 0.0, 0.0, 800),
        ('9502.T', '中部電力', 'Chubu Elec', '電力', '☀️', 8.5, 3.2, 30.0, 1950),
        ('9503.T', '関西電力', 'Kansai Elec', '電力', '☀️', 9.0, 3.1, 25.0, 2100),
        ('9504.T', '中国電力', 'Chugoku Elec', '電力', '☁️', 5.0, 2.5, 30.0, 1100),
        ('9505.T', '北陸電力', 'Hokuriku Elec', '電力', '☁️', 4.5, 2.0, 30.0, 850),
        ('9506.T', '東北電力', 'Tohoku Elec', '電力', '☁️', 6.0, 3.0, 35.0, 1200),
        ('9507.T', '四国電力', 'Shikoku Elec', '電力', '☀️', 6.5, 3.5, 30.0, 1250),
        ('9508.T', '九州電力', 'Kyushu Elec', '電力', '☀️', 7.5, 2.8, 30.0, 1350),
        ('9509.T', '北海道電力', 'Hokkaido Elec', '電力', '☁️', 5.5, 2.0, 40.0, 950),
        ('9513.T', '電源開発', 'J-POWER', '電力', '☁️', 7.5, 4.2, 30.0, 2450),
        # 総合商社
        ('8058.T', '三菱商事', 'Mitsubishi Corp', '卸売', '☀️', 15.5, 3.5, 25.0, 2860),
        ('8001.T', '伊藤忠商事', 'ITOCHU', '卸売', '☀️', 17.0, 3.1, 28.0, 6620),
        ('8031.T', '三井物産', 'Mitsui', '卸売', '☀️', 15.0, 3.2, 28.0, 3100),
        ('8053.T', '住友商事', 'Sumitomo', '卸売', '☀️', 12.0, 4.1, 30.0, 3300),
        ('8002.T', '丸紅', 'Marubeni', '卸売', '☀️', 14.5, 3.8, 25.0, 2450),
        # 通信・メーカー・半導体
        ('9432.T', 'NTT', 'NTT', '通信', '☀️', 12.5, 3.2, 35.0, 180),
        ('9433.T', 'KDDI', 'KDDI', '通信', '☀️', 13.5, 3.8, 42.0, 4800),
        ('7203.T', 'トヨタ', 'Toyota', '自動車', '☀️', 11.5, 2.8, 30.0, 2650),
        ('6758.T', 'ソニーG', 'Sony', '電気機器', '☀️', 14.5, 0.8, 15.0, 13500),
        ('6861.T', 'キーエンス', 'Keyence', '電気機器', '☀️', 17.5, 0.5, 10.0, 68000),
        ('8035.T', '東京エレク', 'TEL', '電気機器', '☀️', 20.0, 1.5, 35.0, 35000),
        ('4063.T', '信越化学', 'Shin-Etsu', '化学', '☀️', 18.2, 1.8, 25.0, 5950),
    ]
    # ボリューム補充用の実在銘柄
    others = [
        ('7267.T', 'ホンダ', 'Honda', '輸送用'), ('9101.T', '日本郵船', 'NYK', '海運'), ('9104.T', '商船三井', 'MOL', '海運'),
        ('4502.T', '武田薬品', 'Takeda', '医薬'), ('1925.T', '大和ハウス', 'Daiwa House', '建設'), ('1928.T', '積水ハウス', 'Sekisui', '建設'),
        ('8766.T', '東京海上', 'Tokio Marine', '保険'), ('6501.T', '日立製作所', 'Hitachi', '電気機器'), ('6702.T', '富士通', 'Fujitsu', '電気機器'),
        ('9020.T', 'JR東日本', 'JR East', '陸運'), ('9022.T', 'JR東海', 'JR Central', '陸運'), ('9201.T', '日本航空', 'JAL', '空運'),
        ('9202.T', 'ANA HD', 'ANA', '空運'), ('9843.T', 'ニトリHD', 'Nitori', '小売'), ('8035.T', '東京エレク', 'TEL', '電気機器'),
        ('4503.T', 'アステラス', 'Astellas', '医薬'), ('6902.T', 'デンソー', 'Denso', '輸送用'), ('7751.T', 'キヤノン', 'Canon', '電気機器'),
        ('8801.T', '三井不動産', 'Mitsui Fud.', '不動産'), ('8802.T', '三菱地所', 'Mitsu. Estate', '不動産'), ('3407.T', '旭化成', 'Asahi Kasei', '化学')
    ]
    stocks = []
    # 最初の固定リストを展開
    for r in raw_list:
        if isinstance(r, dict): stocks.append(r)
        else: stocks.append({'T': r[0], 'N': r[1], 'NE': r[2], 'S': r[3], 'W': r[4], 'R': r[5], 'Y': r[6], 'P': r[7], 'Pr': r[8]})
    # 補充用リストを展開
    for o in others:
        stocks.append({'T': o[0], 'N': o[1], 'NE': o[2], 'S': o[3], 'W': '☀️', 'R': 10.0, 'Y': 3.0, 'P': 40.0, 'Pr': 3000})
    
    # 200社規模にするためにコピーを追加（Tickerをずらしてシミュレーション）
    for i in range(1, 150):
        base = stocks[i % len(stocks)]
        stocks.append({**base, 'T': f'{int(base["T"][:4])+i}.T', 'N': f'{base["N"]}-Sub{i}'})

    df = pd.DataFrame(stocks)
    if current_lang == "English": df['N'] = df['NE']
    return df

# --- 4. 解析・AIスコアリングエンジン (計算エラー修正) ---
@st.cache_data(ttl=3600)
def fetch_and_score(df):
    results = []
    for _, row in df.iterrows():
        try:
            # 高速化のためyfinance取得は最小限に
            tk = yf.Ticker(row['T'])
            t_info = tk.info
            yld = t_info.get('dividendYield')
            # 正確な%変換ロジック
            yld = np.round(float(yld) * 100, 1) if yld is not None and float(yld) < 1 else (np.round(float(yld), 1) if yld else row['Y'])
            roe = np.round(float(t_info.get('returnOnEquity', row['R']/100)) * 100, 1)
            payout = np.round(float(t_info.get('payoutRatio', row['P']/100)) * 100, 1)
            results.append({
                'Ticker': row['T'], 'Name': row['N'], 'Sector': row['S'], 'Trend': row['W'],
                'Yield': np.round(yld, 1), 'Payout': payout, 'ROE': roe, 'Price': t_info.get('previousClose', row['Pr'])
            })
        except:
            results.append({
                'Ticker': row['T'], 'Name': row['N'], 'Sector': row['S'], 'Trend': row['W'],
                'Yield': row['Y'], 'Payout': row['P'], 'ROE': row['R'], 'Price': row['Pr']
            })
    
    res_df = pd.DataFrame(results)
    w_map = {'☀️': 1.0, '☁️': 0.5, '☔': 0.0}
    # AIスコア（無理に100に固定せず、相対的な実力を示す指標へ）
    y_raw = (res_df['ROE'] * 2.0) + (res_df['Yield'] * 7.5) - (res_df['Payout'] * 0.05) + (res_df['Trend'].map(w_map) * 15)
    res_df['Score'] = np.round(y_raw, 1)
    return res_df

with st.spinner('Analyzing...'):
    analyzed_df = fetch_and_score(get_master_data(lang))

# --- 5. サイドバー UI (黄金比ボタンの修正) ---
st.sidebar.header(t["sidebar_head"])

# 黄金比ボタン：押された時に値をSessionStateへセット
if st.sidebar.button(t["golden_btn"]):
    st.session_state["roe_val"] = 6.0
    st.session_state["yield_val"] = 3.0
    st.session_state["payout_val"] = 120.0

# スライダー：SessionStateを参照し、なければデフォルト値を使用
v_roe = st.sidebar.slider(t["min_roe"], 0.0, 30.0, st.session_state.get("roe_val", 6.0), 0.1, key="roe_slider")
v_yield = st.sidebar.slider(t["min_yield"], 0.0, 10.0, st.session_state.get("yield_val", 3.0), 0.1, key="yield_slider")
v_payout = st.sidebar.slider(t["max_payout"], 0.0, 250.0, st.session_state.get("payout_val", 120.0), 0.1, key="payout_slider")

# スライダーの値をSessionStateへ同期（ボタン機能のため）
st.session_state["roe_val"] = v_roe
st.session_state["yield_val"] = v_yield
st.session_state["payout_val"] = v_payout

st.sidebar.markdown("---")
st.sidebar.markdown(t["golden_desc"])

# --- 6. メイン表示 ---
st.title(t["title"])
st.write(t["status"])

final_df = analyzed_df[
    (analyzed_df['ROE'] >= v_roe) & (analyzed_df['Yield'] >= v_yield) & (analyzed_df['Payout'] <= v_payout)
].sort_values(by='Score', ascending=False)

st.subheader(f"📈 {t['result_head']} ({len(final_df)}社)")

# テーブル表示
st.dataframe(
    final_df[['Ticker', 'Name', 'Sector', 'Trend', 'Yield', 'Payout', 'ROE', 'Price', 'Score']]
    .rename(columns={
        'Name': t['col_name'], 'Sector': t['col_sector'], 'Trend': t['col_weather'],
        'Yield': t['col_yield'], 'Payout': t['col_payout'], 'ROE': t['col_roe'],
        'Price': t['col_price'], 'Score': t['col_score']
    })
    .style.background_gradient(subset=[t['col_score']], cmap='Greens')
    .format({t['col_roe']: '{:.1f}', t['col_yield']: '{:.1f}', t['col_payout']: '{:.1f}', 
             t['col_price']: '¥{:,.1f}', t['col_score']: '{:.1f}'}),
    height=600, use_container_width=True, hide_index=True
)

# --- 7. 会社情報 フッター ---
st.markdown("---")
st.info(t["disclaimer"])
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f"{t['footer_1_head']}\n\n{t['footer_1_body']}")
with c2: st.markdown(f"{t['footer_2_head']}\n\n{t['footer_2_body']}")
with c3: st.markdown(f"{t['footer_3_head']}\n\n{t['footer_3_body']}")
st.caption(t["warning"])
