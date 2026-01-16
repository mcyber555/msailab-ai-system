import streamlit as st
import pandas as pd
import numpy as np

# --- 1. ページ設定 ---
st.set_page_config(page_title="MS AI Lab AI Alpha", layout="wide")

# --- 2. 言語設定辞書 ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha",
        "status": "📊 正常稼働中 | 解析基準日: 2026/01/16",
        "sidebar_head": "⚙️ 解析パラメータ設定",
        "lang_label": "🌐 言語選択 / Language Selection",
        "param_desc": "💡 **各指標の推奨値と設定根拠**\n\n1. **配当利回り 3.0%以上**: 安定した収益確保と下落耐性の基準。\n2. **配当性向 120.0%以下**: 無理のない健全な株主還元の評価基準。\n3. **ROE 6.0%以上**: 資本を効率的に活用できているかの経営指標。",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "配当利回り (%) (下限)",
        "max_payout": "許容配当性向 (上限 %)",
        "result_head": "東証プライム 厳選100銘柄 AI解析結果",
        "unit": "社",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界",
        "col_yield": "配当利回り(%)", "col_payout": "配当性向(%)", "col_roe": "ROE(%)", 
        "col_price": "終値", "col_score": "AIスコア",
        "disclaimer": "📌 本解析結果は、MS AI Labが独自に開発したAIアルゴリズムによる抽出サンプルです。解析精度の担保のため、東証プライム市場より主要100社を厳選しています。実運用においては全上場銘柄を対象とした網羅的解析を実施しています。",
        "footer_1_head": "**【運営組織】**",
        "footer_1_body": "合同会社MS AI Lab  \n設立者: 鈴木 学  \n設立日: 2026年1月15日",
        "footer_2_head": "**【AI解析テクノロジー】**",
        "footer_2_body": "アンサンブル学習アルゴリズム「ランダムフォレスト」を採用。収益性・還元姿勢・財務健全性を多角的に解析し、投資効率を最大化する評価スコアを算出します。",
        "footer_3_head": "**【事業目的】**",
        "footer_3_body": "独自AIスコアリングに基づく資産運用。長期的な増配可能性と企業成長を両立する銘柄への投資を最適化します。",
        "copyright": "© 2026 MS AI Lab LLC. All Rights Reserved.",
        "warning": "※本システムは自己勘定取引専用であり、投資助言等は行いません。"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha",
        "status": "📊 Status: Active | Date: 2026/01/16",
        "sidebar_head": "⚙️ Parameters",
        "lang_label": "🌐 Language Selection",
        "param_desc": "💡 **Logic**\n\n1. **Yield 3.0%+**: Secure income with downside protection.\n2. **Payout 120.0%-**: Dividend sustainability check.\n3. **ROE 6.0%+**: Standard for capital efficiency.",
        "min_roe": "Required ROE (%)",
        "min_yield": "Div. Yield (%)",
        "max_payout": "Max Payout (%)",
        "result_head": "AI Analysis Results (Selected 100 Prime Stocks)",
        "unit": "stocks",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector",
        "col_yield": "Yield(%)", "col_payout": "Payout(%)", "col_roe": "ROE(%)", 
        "col_price": "Price", "col_score": "AI Score",
        "disclaimer": "📌 Note: Sample analysis. We have selected 100 major companies from the TSE Prime Market for reliability. Actual operations scan all 3,800 TSE listed stocks.",
        "footer_1_head": "**【Organization】**",
        "footer_1_body": "MS AI Lab LLC  \nFounder: Manabu Suzuki  \nEstablished: Jan 15, 2026",
        "footer_2_head": "**【AI Technology】**",
        "footer_2_body": "Employs Random Forest algorithm to evaluate metrics and calculate proprietary scores for investment efficiency.",
        "footer_3_head": "**【Business Goal】**",
        "footer_3_body": "Proprietary trading based on AI scoring to optimize growth and dividend potential.",
        "copyright": "© 2026 MS AI Lab LLC. All Rights Reserved.",
        "warning": "Proprietary trading only."
    }
}

lang = st.sidebar.radio(LANG_MAP["日本語"]["lang_label"], ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. 実在100銘柄データベース (手動定義・重複なし) ---
@st.cache_data
def get_verified_data():
    # 100社分の実在データ（途中で途切れないよう整理）
    data = [
        ('2914.T','日本たばこ産業','JT','食料品','Foods',16.5,6.2,75.0,4150),
        ('8306.T','三菱UFJ','MUFG','銀行業','Banking',8.8,3.8,38.0,1480),
        ('8316.T','三井住友FG','SMFG','銀行業','Banking',8.2,4.0,40.0,8900),
        ('8411.T','みずほFG','Mizuho','銀行業','Banking',7.5,3.7,40.0,3180),
        ('8058.T','三菱商事','Mitsubishi','卸売業','Trading',15.5,3.5,25.0,2860),
        ('8001.T','伊藤忠商事','ITOCHU','卸売業','Trading',17.0,3.1,28.0,6620),
        ('8031.T','三井物産','Mitsui','卸売業','Trading',15.0,3.2,28.0,3100),
        ('8053.T','住友商事','Sumitomo','卸売業','Trading',12.5,4.1,30.0,3320),
        ('8002.T','丸紅','Marubeni','卸売業','Trading',14.5,3.8,25.0,2480),
        ('9432.T','日本電信電話','NTT','情報・通信','Telecom',
