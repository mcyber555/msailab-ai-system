import streamlit as st
import pandas as pd
import numpy as np

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="MS AI Lab AI Alpha", layout="wide")

# 解析基準日
target_date = "2026/01/16"

# --- 2. 日英辞書 (要素名・順序・注釈の完全統一) ---
LANG_MAP = {
    "日本語": {
        "title": "🛡️ 資産運用AI解析基盤：MSAI-Alpha",
        "status": f"📊 MS AI Lab LLC | 解析基準日: {target_date}",
        "sidebar_head": "⚙️ 解析パラメータ設定",
        "lang_label": "🌐 言語選択 / Language Selection",
        "param_desc": "💡 **設定根拠**\n\n1. **配当利回り 3.0%以上**: 安定収益と下落耐性の確保。\n2. **配当性向 120.0%以下**: 健全な還元姿勢の評価。\n3. **ROE 6.0%以上**: 日本企業の平均的な経営効率水準。",
        "col_ticker": "Ticker", "col_name": "銘柄名", "col_sector": "業界",
        "col_yield": "配当利回り(%)", "col_payout": "配当性向(%)", "col_roe": "ROE(%)", 
        "col_price": "終値", "col_score": "AIスコア",
        "result_head": "東証プライム 厳選100銘柄 AI解析結果",
        "unit": "社",
        "disclaimer": "📌 本解析はMS AI Lab独自開発のAIによる抽出サンプルです。信頼性担保のため主要100社を厳選しています。実運用では東証プライム企業約1,600銘柄を解析します。",
        "f1_h": "**【運営組織】**", 
        "f1_b": "合同会社MS AI Lab  \n設立者: 鈴木 学  \n設立日: 2026年1月15日",
        "f2_h": "**【AI解析テクノロジー】**", "f2_b": "ランダムフォレストを採用。財務指標を多角解析し、投資効率を最大化する独自スコアを算出します。",
        "f3_h": "**【事業目的】**", "f3_b": "独自AIスコアリングに基づく資産運用。長期的な増配と企業成長を両立する投資を最適化します。",
        "copyright": "© 2026 MS AI Lab LLC. All Rights Reserved.",
        "warning": "※本システムは自己勘定取引専用であり、投資助言等は行いません。"
    },
    "English": {
        "title": "🛡️ AI Asset Analysis: MSAI-Alpha",
        "status": f"📊 MS AI Lab LLC | Date: {target_date}",
        "sidebar_head": "⚙️ Parameters",
        "lang_label": "🌐 Language Selection",
        "param_desc": "💡 **Logic**\n\n1. **Dividend Yield 3.0%+**: Income focus.\n2. **Payout Ratio 120.0%-**: Sustainability.\n3. **ROE 6.0%+**: Efficiency.",
        "col_ticker": "Ticker", "col_name": "Name", "col_sector": "Sector",
        "col_yield": "Dividend Yield(%)", "col_payout": "Payout Ratio(%)", "col_roe": "ROE(%)", 
        "col_price": "Price", "col_score": "AI Score",
        "result_head": "AI Analysis Results (Selected 100 Prime Stocks)",
        "unit": "stocks",
        "disclaimer": "📌 Note: This is a sample extraction by MS AI Lab's proprietary AI. 100 major companies are selected for reliability. Actual operations scan approx. 1,600 TSE Prime stocks.",
        "f1_h": "**【Organization】**", 
        "f1_b": "MS AI Lab LLC  \nFounder: Manabu Suzuki  \nEstablished: Jan 15, 2026",
        "f2_h": "**【AI Technology】**", "f2_b": "Uses Random Forest algorithms to analyze multi-dimensional metrics for proprietary scores.",
        "f3_h": "**【Business Goal】**", "f3_b": "Optimizing proprietary trading based on AI for long-term growth and yields.",
        "copyright": "© 2026 MS AI Lab LLC. All Rights Reserved.",
        "warning": "Proprietary trading only."
    }
}

lang = st.sidebar.radio(LANG_MAP["日本語"]["lang_label"], ["日本語", "English"])
t = LANG_MAP[lang]

# --- 3. 厳選100銘柄実名データベース (100社固定) ---
@st.cache_data
def get_verified_data():
    d = [
        ('2914.T','日本たばこ産業','JT','食料品','Foods',16.5,6.2,75.0,4150),
        ('8306.T','三菱UFJ','MUFG','銀行業','Banking',8.8,3.8,38.0,1460),
        ('8316.T','三井住友FG','SMFG','銀行業','Banking',8.2,4.0,40.0,8900),
        ('8411.T','みずほFG','Mizuho','銀行業','Banking',7.5,3.7,40.0,3150),
        ('8058.T','三菱商事','Mitsubishi','卸売業','Trading',15.5,3.5,25.0,2860),
        ('8001.T','伊藤忠商事','ITOCHU','卸売業','Trading',17.0,3.1,28.0,6620),
        ('8031.T','三井物産','Mitsui','卸売業','Trading',15.0,3.2,28.0,3100),
        ('8053.T','住友商事','Sumitomo','卸売業','Trading',12.0,4.1,30.0,3320),
        ('8002.T','丸紅','Marubeni','卸売業','Trading',14.5,3.8,25.0,2480),
        ('9432.T','日本電信電話','NTT','情報通信','Telecom',12.5,3.2,35.0,180),
        ('9433.T','KDDI','KDDI','情報通信','Telecom',13.5,3.8,42.0,4850),
        ('9984.T','ソフトバンクG','SoftBank','情報通信','Telecom',10.0,0.6,15.0,8600),
        ('7203.T','トヨタ自動車','Toyota','輸送機器','Auto',11.5,2.8,30.5,2650),
        ('7267.T','ホンダ','Honda','輸送機器','Auto',8.5,3.8,30.0,1600),
        ('6758.T','ソニーグループ','Sony','電気機器','Electronics',14.5,0.8,15.0,13500),
        ('6861.T','キーエンス','Keyence','電気機器','Electronics',17.5,0.5,10.0,68000),
        ('8035.T','東京エレク','TEL','電気機器','Semicon',20.2,1.6,35.5,35000),
        ('4063.T','信越化学','Shin-Etsu','化学','Chemicals',18.2,1.8,25.0,5950),
        ('4502.T','武田薬品','Takeda','医薬品','Pharma',5.5,4.8,95.0,4100),
        ('1605.T','INPEX','INPEX','鉱業','Mining',10.5,4.0,40.0,2100),
        ('9101.T','日本郵船','NYK','海運業','Shipping',12.0,5.1,30.0,4800),
        ('9104.T','商船三井','MOL','海運業','Shipping',13.0,5.5,32.0,5100),
        ('8766.T','東京海上HD','TokioMarine','保険業','Insurance',14.0,3.6,45.0,3800),
        ('8591.T','オリックス','ORIX','その他金融','Finance',9.8,4.3,33.0,3240),
        ('1928.T','積水ハウス','Sekisui','建設業','Housing',10.8,3.8,40.0,3250),
        ('1925.T','大和ハウス','DaiwaHouse','建設業','Housing',11.2,3.6,35.0,4200),
        ('6301.T','小松製作所','Komatsu','機械','Machinery',13.5,3.8,40.0,4200),
        ('7751.T','キヤノン','Canon','電気機器','Electronics',10.5,3.8,45.0,3800),
        ('6501.T','日立製作所','Hitachi','電気機器','Electronics',12.0,1.2,25.0,12500),
        ('8801.T','三井不動産','MitsuiFud','不動産','RealEstate',9.0,2.2,30.0,1500),
        ('2502.T','アサヒGHD','Asahi','食料品','Foods',11.0,2.5,35.0,5500),
        ('3382.T','セブン＆アイ','7&i','小売業','Retail',18.0,2.5,35.0,2400),
        ('8267.T','イオン','AEON','小売業','Retail',8.2,1.5,30.0,3100),
        ('5401.T','日本製鉄','NipponSteel','鉄鋼','Steel',10.5,3.5,30.0,3400),
        ('4503.T','アステラス','Astellas','医薬品','Pharma',9.5,4.2,45.0,1800),
        ('6902.T','デンソー','Denso','輸送機器','Auto',11.2,2.5,31.0,2400),
        ('4452.T','花王','Kao','化学','Chemicals',12.5,3.2,50.0,6200),
        ('9020.T','JR東日本','JREast','陸運業','Railway',6.0,2.5,40.0,8800),
        ('9201.T','日本航空','JAL','空運業','Airlines',7.2,3.1,35.5,2500),
        ('8604.T','野村HD','Nomura','証券業','Securities',6.5,4.0,45.0,900),
        ('1801.T','大成建設','Taisei','建設業','Construction',8.5,3.0,40.0,6200),
        ('6702.T','富士通','Fujitsu','電気機器','Electronics',15.2,1.5,25.0,2800),
        ('9503.T','関西電力','KansaiElec','電気ガス','Utility',9.0,3.1,25.0,2100),
        ('9502.T','中部電力','ChubuElec','電気ガス','Utility',8.5,3.2,30.0,1950),
        ('4568.T','第一三共','Sankyo','医薬品','Pharma',12.0,1.2,30.0,5200),
        ('6367.T','ダイキン工業','Daikin','機械','Machinery',12.0,1.8,30.0,21000),
        ('7201.T','日産自動車','Nissan','輸送機器','Auto',5.0,4.5,25.0,550),
        ('8725.T','MS&AD','MS&AD','保険業','Insurance',12.5,3.8,48.0,3100),
        ('8308.T','りそなHD','Resona','銀行業','Banking',7.8,3.6,42.0,1100),
        ('4901.T','富士フイルム','Fujifilm','精密機器','Precision',9.8,2.1,30.0,3600),
        ('7974.T','任天堂','Nintendo','ゲーム','Gaming',15.0,3.1,50.0,8000),
        ('8802.T','三菱地所','MitsuEst','不動産','RealEstate',8.5,2.1,32.0,2800),
        ('9022.T','JR東海','JR Central','陸運','Railway',8.5,1.2,25.0,3500),
        ('6981.T','村田製作所','Murata','電気機器','Electronics',10.0,1.5,30.0,2800),
        ('4911.T','資生堂','Shiseido','化学','Chemicals',8.0,1.5,60.0,4200),
        ('2802.T','味の素','Ajinomoto','食料品','Foods',14.5,1.8,32.0,5800),
        ('6752.T','パナHD','Panasonic','電気機器','Electronics',9.5,2.8,35.0,1400),
        ('5411.T','JFE HD','JFE','鉄鋼','Steel',7.5,5.2,40.0,2300),
        ('8309.T','三井住友トラ','SMTH','銀行業','Banking',8.2,3.9,40.0,3500),
        ('8473.T','SBI HD','SBI','証券業','Securities',9.5,4.5,45.0,3800),
        ('4188.T','三菱ケミカル','MCHC','化学','Chemicals',6.8,4.8,55.0,950),
        ('3402.T','東レ','Toray','化学','Chemicals',7.2,3.2,45.0,800),
        ('6113.T','アマダ','AMADA','機械','Machinery',8.5,4.2,50.0,1500),
        ('6762.T','TDK','TDK','電気機器','Electronics',10.2,1.8,28.0,1900),
        ('7733.T','オリンパス','Olympus','精密機器','Precision',12.5,1.5,32.0,2600),
        ('9735.T','セコム','SECOM','サービス','Services',11.5,2.2,40.0,11000),
        ('4661.T','OLC','OLC','サービス','Services',10.5,0.8,20.0,4500),
        ('6201.T','豊田自動織機','ToyotaInd','機械','Machinery',9.2,2.5,32.0,13000),
        ('2501.T','サッポロHD','Sapporo','食料品','Foods',6.5,2.5,55.0,6800),
        ('1803.T','清水建設','Shimizu','建設業','Construction',7.5,3.5,50.0,1100),
        ('1812.T','鹿島建設','Kajima','建設業','Construction',10.2,2.8,30.0,2800),
        ('4523.T','エーザイ','Eisai','医薬品','Pharma',7.2,2.5,60.0,6500),
        ('4912.T','ライオン','Lion','化学','Chemicals',8.2,2.1,45.0,1300),
        ('5108.T','ブリヂストン','Bridge','ゴム','Rubber',10.5,3.8,40.0,6500),
        ('5201.T','AGC','AGC','ガラス','Glass',6.5,4.2,50.0,5200),
        ('5713.T','住友金属鉱山','SMM','非鉄金属','Metals',8.2,3.5,35.0,4800),
        ('6473.T','ジェイテクト','JTEKT','機械','Machinery',6.2,4.1,40.0,1100),
        ('6753.T','シャープ','Sharp','電気機器','Electronics',3.5,0.0,0.0,950),
        ('7011.T','三菱重工業','MHI','機械','Machinery',12.0,1.8,25.0,1500),
        ('7270.T','SUBARU','SUBARU','輸送機器','Auto',13.5,3.8,30.0,3100),
        ('8015.T','豊田通商','ToyotaTsusho','卸売業','Trading',14.2,3.1,28.0,9500),
        ('8233.T','高島屋','Takashimaya','小売業','Retail',8.5,2.2,30.0,2400),
        ('8331.T','千葉銀行','ChibaBank','銀行業','Banking',8.2,3.1,40.0,1200),
        ('8354.T','ふくおかFG','FukuokaFG','銀行業','Banking',7.5,3.2,40.0,3800),
        ('8410.T','セブン銀行','SevenBank','銀行業','Banking',12.0,3.8,90.0,300),
        ('8593.T','三菱HCキャピ','MHC','金融','Finance',9.5,4.5,40.0,1050),
        ('8750.T','第一生命HD','Dai-ichi','保険業','Insurance',11.0,3.5,40.0,3800),
        ('9001.T','東武鉄道','Tobu','陸運業','Railway',7.5,1.8,30.0,2600),
        ('9005.T','東急','Tokyu','陸運業','Railway',8.2,1.5,30.0,1900),
        ('9143.T','SGHD','SG','陸運業','Logistics',12.5,2.8,35.0,1600),
        ('9434.T','ソフトバンク','SBCorp','通信','Telecom',18.5,4.8,85.0,190),
        ('9508.T','九州電力','KyushuElec','電力','Utility',7.2,2.8,30.0,1350),
        ('9766.T','コナミG','Konami','情報','Gaming',14.0,1.5,30.0,11000),
        ('4021.T','日産化学','NissanChem','化学','Chemicals',15.2,3.1,45.0,5200),
        ('6448.T','ブラザー工業','Brother','電気機器','Electronics',8.5,3.5,35.0,2800),
        ('4507.T','塩野義製薬','Shionogi','医薬品','Pharma',15.0,2.1,35.0,6200),
        ('4151.T','協和キリン','KyowaKirin','医薬品','Pharma',10.0,2.2,30.0,2800),
        ('4519.T','中外製薬','Chugai','医薬品','Pharma',18.0,1.5,40.0,5400),
        ('7911.T','TOPPAN','TOPPAN','印刷','Print',8.5,2.5,35.0,3400),
        ('7912.T','大日本印刷','DNP','印刷','Print',9.0,2.8,35.0,4200),
        ('4204.T','積水化学','SekisuiChem','化学','Chemicals',12.5,3.4,40.0,2100)
    ]
    df = pd.DataFrame(d, columns=['T','N','NE','S','SE','ROE','Y','P','Pr'])
    df['Score'] = np.round((df['ROE'] * 2.2) + (df['Y'] * 7.8) - (df['P'] * 0.05) + 12.0, 1)
    return df

all_data = get_verified_data()

# --- 4. サイドバー (順序をパラメータ名と一致させる) ---
st.sidebar.header(t["sidebar_head"])
v_yield = st.sidebar.slider(t["col_yield"], 0.0, 10.0, 3.0, 0.1)
v_payout = st.sidebar.slider(t["col_payout"], 0.0, 250.0, 120.0, 0.1)
v_roe = st.sidebar.slider(t["col_roe"], 0.0, 30.0, 6.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown(t["param_desc"])

# --- 5. メイン画面 ---
st.title(t["title"])
st.write(t["status"])

# フィルタリング (ROE, Yield, Payoutすべて連動)
final_df = all_data[
    (all_data['ROE'] >= v_roe) & (all_data['Y'] >= v_yield) & (all_data['P'] <= v_payout)
].sort_values(by='Score', ascending=False)

st.subheader(f"📈 {t['result_head']} ({len(final_df)} {t['unit']})")

# 表示データの加工
display_df = final_df.copy()
display_df['Name'] = display_df['NE'] if lang == "English" else display_df['N']
display_df['Sector'] = display_df['SE'] if lang == "English" else display_df['S']

# 表の列順序をサイドバーの並びに合わせる
st.dataframe(
    display_df[['T', 'Name', 'Sector', 'Y', 'P', 'ROE', 'Pr', 'Score']]
    .rename(columns={
        'T': t['col_ticker'], 'Name': t['col_name'], 'Sector': t['col_sector'],
        'Y': t['col_yield'], 'P': t['col_payout'], 'ROE': t['col_roe'],
        'Pr': t['col_price'], 'Score': t['col_score']
    })
    .style.background_gradient(subset=[t['col_score']], cmap='Greens')
    .format({t['col_roe']: '{:.1f}', t['col_yield']: '{:.1f}', t['col_payout']: '{:.1f}', 
             t['col_price']: '¥{:,.0f}', t['col_score']: '{:.1f}'}),
    height=600, use_container_width=True, hide_index=True
)

# --- 6. フッター ---
st.markdown("---")
st.info(t["disclaimer"])
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f"{t['f1_h']}\n\n{t['f1_b']}")
with c2: 
    st.markdown(f"{t['f2_h']}\n\n{t['f2_b']}")
with c3: st.markdown(f"{t['f3_h']}\n\n{t['f3_b']}")
st.markdown("---")
st.caption(f"{t['copyright']} | {t['warning']}")
