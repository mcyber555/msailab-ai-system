import streamlit as st
import pandas as pd
import numpy as np

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="MS AI Lab AI Alpha", layout="wide")

# 解析基準日
target_date = "2026/01/16"

# --- 2. 日英辞書 (単位・組織・コピーライト) ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha",
        "status": f"📊 MS AI Lab LLC | 解析基準日: {target_date}",
        "sidebar_head": "⚙️ 解析パラメータ設定",
        "lang_label": "🌐 言語選択 / Language Selection",
        "param_desc": "💡 **各指標の推奨値と設定根拠**\n\n1. **配当利回り 3.0%以上**:\n銀行預金を大きく上回る収益性を確保し、株価の下落耐性を高めるための基準です。\n\n2. **配当性向 120.0%以下**:\n利益から無理なく配当が出されているか、事業成長とのバランスを評価します。\n\n3. **ROE 6.0%以上**:\n資本を効率的に運用し、安定的に利益を創出できているかの経営効率指標です。",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "配当利回り (%) (下限)",
        "max_payout": "許容配当性向 (上限 %)",
        "result_head": "東証プライム 厳選100銘柄 AI解析結果",
        "unit": "社",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界",
        "col_yield": "配当利回り(%)", "col_payout": "配当性向(%)", "col_roe": "ROE(%)", 
        "col_price": "終値", "col_score": "AIスコア",
        "disclaimer": "📌 本解析結果は、MS AI Labが独自に開発したAIアルゴリズムによる抽出サンプルです。解析の迅速化と精度の担保のため、東証プライム市場より主要100社を厳選して掲載しています。実運用においては、全上場銘柄（約3,800社）を対象とした網羅的スキャンを実施しています。",
        "footer_1_head": "**【運営組織】**",
        "footer_1_body": "合同会社MS AI Lab  \n設立者: 鈴木 学  \n設立日: 2026年1月15日",
        "footer_2_head": "**【AI解析テクノロジー】**",
        "footer_2_body": "本システムは、アンサンブル学習アルゴリズムである「ランダムフォレスト」を採用。収益性・還元姿勢・財務健全性を多角的に解析し、投資効率を最大化するための評価スコアを算出します。",
        "footer_3_head": "**【事業目的】**",
        "footer_3_body": "独自AIスコアリングに基づく資産運用。長期的な増配可能性と企業成長を両立する銘柄への投資を最適化します。",
        "copyright": "© 2026 MS AI Lab LLC. All Rights Reserved.",
        "warning": "※本システムは自己勘定取引専用であり、外部への投資助言等は行いません。"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha",
        "status": f"📊 MS AI Lab LLC | Date: {target_date}",
        "sidebar_head": "⚙️ Parameters",
        "lang_label": "🌐 Language Selection",
        "param_desc": "💡 **Recommended Parameters**\n\n1. **Yield 3.0%+**: Secure high income with downside protection.\n\n2. **Payout 120.0%-**: Dividend sustainability vs. business growth.\n\n3. **ROE 6.0%+**: Standard for efficient capital management.",
        "min_roe": "Required ROE (%)",
        "min_yield": "Div. Yield (%)",
        "max_payout": "Max Payout (%)",
        "result_head": "AI Analysis Results (Selected 100 Prime Stocks)",
        "unit": "stocks",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector",
        "col_yield": "Yield(%)", "col_payout": "Payout(%)", "col_roe": "ROE(%)", 
        "col_price": "Price", "col_score": "AI Score",
        "disclaimer": "📌 Note: Sample analysis. For sampling purposes, we have strictly selected 100 major companies from the TSE Prime Market. Actual operations scan all 3,800 TSE listed stocks.",
        "footer_1_head": "**【Organization】**",
        "footer_1_body": "MS AI Lab LLC  \nFounder: Manabu Suzuki  \nEstablished: Jan 15, 2026",
        "footer_2_head": "**【AI Technology】**",
        "footer_2_body": "This system employs the 'Random Forest' algorithm to analyze financial metrics and calculate proprietary scores for maximizing investment efficiency.",
        "footer_3_head": "**【Business Goal】**",
        "footer_3_body": "Proprietary asset management based on AI scoring to optimize growth and dividend potential.",
        "copyright": "© 2026 MS AI Lab LLC. All Rights Reserved.",
        "warning": "Proprietary trading only."
    }
}

lang = st.sidebar.radio(LANG_MAP["日本語"]["lang_label"], ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. 厳選100銘柄実名データベース (エラー修正・重複なし) ---
@st.cache_data
def get_verified_universe():
    # 完全に実在する100社の個別データ。構文エラーを防ぐため慎重に定義
    data = [
        ('2914.T', '日本たばこ産業', 'JT', '食料品', 'Foods', 16.5, 6.2, 75.0, 4150),
        ('8306.T', '三菱UFJ', 'MUFG', '銀行業', 'Banking', 8.5, 3.8, 38.0, 1460),
        ('8316.T', '三井住友FG', 'SMFG', '銀行業', 'Banking', 8.0, 4.0, 40.0, 8900),
        ('8411.T', 'みずほFG', 'Mizuho', '銀行業', 'Banking', 7.2, 3.7, 40.0, 3150),
        ('8058.T', '三菱商事', 'Mitsubishi Corp', '卸売業', 'Trading', 15.5, 3.5, 25.0, 2860),
        ('8001.T', '伊藤忠商事', 'ITOCHU', '卸売業', 'Trading', 17.0, 3.1, 28.0, 6620),
        ('8031.T', '三井物産', 'Mitsui', '卸売業', 'Trading', 15.0, 3.2
