import streamlit as st
import pandas as pd
import numpy as np

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="MS AI Lab AI Alpha", layout="wide")

# --- 2. 日英辞書 (エラー防止のため文字列を確実にクローズ) ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha",
        "status": "📊 正常稼働中 | 解析対象: 東証プライム厳選100銘柄",
        "sidebar_head": "⚙️ 解析パラメータ設定",
        "lang_label": "🌐 言語選択 / Language Selection",
        "param_desc": "💡 **設定根拠**\n\n1. **配当利回り 3.0%以上**: 安定収益と下落耐性の確保。\n2. **配当性向 120.0%以下**: 健全な還元姿勢の評価。\n3. **ROE 6.0%以上**: 日本企業の平均的な経営効率水準。",
        "min_roe": "要求ROE (下限 %)",
        "min_yield": "利回り (%) (下限)",
        "max_payout": "許容性向 (%) (上限)",
        "result_head": "東証プライム 厳選100銘柄 AI解析結果",
        "unit": "社",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界",
        "col_yield": "利回り(%)", "col_payout": "性向(%)", "col_roe": "ROE(%)", 
        "col_price": "終値", "col_score": "AIスコア",
        "disclaimer": "📌 本解析はMS AI Lab独自開発のAIによる抽出サンプルです。信頼性担保のため主要100社を厳選しています。実運用では全3,800銘柄を解析します。",
        "f1_h": "**【運営組織】**", "f1_b": "合同会社MS AI Lab\n設立者: 鈴木 学\n設立日: 2026年1月15日",
        "f2_h": "**【AI解析テクノロジー】**", "f2_b": "アンサンブル学習（ランダムフォレスト）を採用。財務指標を多角解析し、投資効率を最大化するスコアを算出します。",
        "f3_h": "**【事業目的】**", "f3_b": "独自AIスコアリングに基づく資産運用。長期的な増配と企業成長を両立する投資を最適化します。",
        "copyright": "© 2026 MS AI Lab LLC. All Rights Reserved.",
        "warning": "※本システムは自己勘定取引専用であり、投資助言等は行いません。"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha",
        "status": "📊 Status: Active | Universe: 100 Selected Prime Stocks",
        "sidebar_head": "⚙️ Parameters",
        "lang_label": "🌐 Language Selection",
        "param_desc": "💡 **Logic**\n\n1. **Yield 3.0%+**: Income vs bank rates.\n2. **Payout 120.0%-**: Sustainability check.\n3. **ROE 6.0%+**: Capital efficiency.",
        "min_roe": "Required ROE (%)",
        "min_yield": "Div. Yield (%)",
        "max_payout": "Max Payout (%)",
        "result_head": "AI Analysis Results (Selected 100 Prime Stocks)",
        "unit": "stocks",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector",
        "col_yield": "Yield(%)", "col_payout": "Payout(%)", "col_roe": "ROE(%)", 
        "col_price": "Price", "col_score": "AI Score",
        "disclaimer": "📌 Note: Sample analysis. 100 companies are selected for demo purposes. Actual operations scan all 3,800 TSE stocks.",
        "f1_h": "**【Organization】**", "f1_b": "MS AI Lab LLC\nFounder: Manabu Suzuki\nEstablished: Jan 15, 2026",
        "f2_h": "**【AI Technology】**", "f2_b": "Employs Random Forest algorithm to evaluate metrics and calculate proprietary scores for investment efficiency.",
        "f3_h": "**【Business Goal】**", "f3_b": "Optimizing asset management based on AI scoring for long-term potential.",
        "copyright": "© 2026 MS AI Lab LLC. All Rights Reserved.",
        "warning": "Proprietary trading only."
    }
}

lang = st.sidebar.radio(LANG_MAP["日本語"]["lang_label"], ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. 厳選100銘柄実名データベース (ダブり・ダミー完全排除) ---
@st.cache_data
def get_verified_data():
    # 東証プライムの実在企業100社の実データ
    d = [
        ('2914.T','日本たばこ産業','JT','食料品','Foods',16.5,6.2,75.0,4150),
        ('8306.T','三菱UFJ','MUFG','銀行業','Banking',8.5,3.8,38.0,1460),
        ('8316.T','三井住友FG','SMFG','銀行業','Banking',8.0,4.0,40.0,8900),
        ('8411.T','みずほFG','Mizuho','銀行業','Banking',7.2,3.7,40.0,3150),
        ('8058.T','三菱商事','Mitsubishi','卸売業','Trading',15.5,3.5,25.0,2860),
        ('8001.T','伊藤忠商事','ITOCHU','卸売業','Trading',17.0,3.1,28.0,6620),
        ('8031.T','三井物産','Mitsui','卸売業','Trading',15.0,3.2,28.0,3100),
        ('8053.T','住友商事','Sumitomo','卸売業','Trading',12.0,4.1,30.0,3320),
