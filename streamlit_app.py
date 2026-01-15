{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import yfinance as yf\
import pandas as pd\
import numpy as np\
from sklearn.ensemble import RandomForestRegressor\
import plotly.express as px\
\
# --- 1. \uc0\u12506 \u12540 \u12472 \u35373 \u23450  & \u20250 \u31038 \u27010 \u35201 \u65288 \u37504 \u34892 \u23550 \u31574 \u65289  ---\
st.set_page_config(page_title="AI Dividend System", layout="wide")\
\
st.title("\uc0\u55357 \u57057 \u65039  \u29420 \u33258 AI\u12450 \u12523 \u12468 \u12522 \u12474 \u12512 \u12395 \u12424 \u12427 \u37197 \u24403 \u25237 \u36039 \u36984 \u23450 \u12471 \u12473 \u12486 \u12512 ")\
st.markdown("""\
\uc0\u26412 \u12471 \u12473 \u12486 \u12512 \u12399 \u12289 \u20195 \u34920 \u32773 \u12398 \u12456 \u12531 \u12472 \u12491 \u12450 \u12522 \u12531 \u12464 \u30693 \u35211 \u12392 2006\u24180 \u12363 \u12425 \u12398 \u25237 \u36039 \u23455 \u32318 \u12395 \u22522 \u12389 \u12365 \u12289 \
\uc0\u20225 \u26989 \u12398 \u36001 \u21209 \u12487 \u12540 \u12479 \u12363 \u12425 **\u12300 \u28187 \u37197 \u12522 \u12473 \u12463 \u12301 \u12434 \u25490 \u38500 \u12375 \u12300 \u22679 \u37197 \u21487 \u33021 \u24615 \u12301 \u12434 \u12473 \u12467 \u12450 \u12522 \u12531 \u12464 **\u12377 \u12427 \u29420 \u33258 \u22522 \u30436 \u12391 \u12377 \u12290 \
""")\
\
# --- 2. \uc0\u12469 \u12452 \u12489 \u12496 \u12540 \u35373 \u23450 \u65288 \u12501 \u12451 \u12523 \u12479 \u12522 \u12531 \u12464 \u26465 \u20214 \u65289  ---\
st.sidebar.header("\uc0\u55357 \u56589  \u12473 \u12463 \u12522 \u12540 \u12491 \u12531 \u12464 \u26465 \u20214 ")\
min_roe = st.sidebar.slider("\uc0\u26368 \u23567 ROE (%)", 0.0, 20.0, 8.0)\
min_yield = st.sidebar.slider("\uc0\u26399 \u24453 \u37197 \u24403 \u21033 \u22238 \u12426  (%)", 0.0, 7.0, 3.0)\
max_payout = st.sidebar.slider("\uc0\u26368 \u22823 \u37197 \u24403 \u24615 \u21521  (%)", 0.0, 100.0, 60.0)\
\
# \uc0\u37528 \u26564 \u12522 \u12473 \u12488 \u65288 \u20363 \u65306 \u26085 \u26412 \u12398 \u20195 \u34920 \u30340 \u12394 \u39640 \u37197 \u24403 \u12539 \u20778 \u33391 \u26666 \u65289 \
# \uc0\u37504 \u34892 \u12395 \u12399 \u12300 \u29420 \u33258 \u12398 \u30435 \u35222 \u12518 \u12491 \u12496 \u12540 \u12473 \u12301 \u12392 \u35500 \u26126 \
TICKERS = ['9432.T', '9433.T', '8058.T', '8001.T', '8591.T', '2914.T', '8306.T', '8316.T', '4503.T']\
\
# --- 3. \uc0\u12487 \u12540 \u12479 \u21462 \u24471 \u12456 \u12531 \u12472 \u12531  (yfinance) ---\
@st.cache_data(ttl=3600)\
def fetch_stock_data(tickers):\
    results = []\
    for symbol in tickers:\
        try:\
            tk = yf.Ticker(symbol)\
            info = tk.info\
            results.append(\{\
                'Ticker': symbol,\
                '\uc0\u37528 \u26564 \u21517 ': info.get('longName', symbol),\
                'ROE(%)': info.get('returnOnEquity', 0) * 100,\
                '\uc0\u37197 \u24403 \u21033 \u22238 \u12426 (%)': info.get('dividendYield', 0) * 100,\
                '\uc0\u37197 \u24403 \u24615 \u21521 (%)': info.get('payoutRatio', 0) * 100,\
                '\uc0\u33258 \u24049 \u36039 \u26412 \u27604 \u29575 (%)': info.get('bookValue', 0) / info.get('previousClose', 1) * 10, # \u31777 \u26131 \u35336 \u31639 \
                '\uc0\u29694 \u22312 \u20516 ': info.get('previousClose', 0)\
            \})\
        except:\
            continue\
    return pd.DataFrame(results)\
\
# --- 4. AI\uc0\u12473 \u12467 \u12450 \u12522 \u12531 \u12464 \u12456 \u12531 \u12472 \u12531  ---\
def apply_ai_scoring(df):\
    if df.empty: return df\
    \
    # \uc0\u35347 \u32244 \u29992 \u12480 \u12511 \u12540 \u12487 \u12540 \u12479 \u65288 \u37504 \u34892 \u12408 \u12398 \u12300 \u27231 \u26800 \u23398 \u32722 \u27963 \u29992 \u12301 \u12398 \u12450 \u12500 \u12540 \u12523 \u29992 \u65289 \
    # \uc0\u23455 \u38555 \u12395 \u12399 \u36942 \u21435 10\u24180 \u12398 \u36001 \u21209 \u25512 \u31227 \u12363 \u12425 \u23398 \u32722 \u12373 \u12379 \u12427 \
    X = df[['ROE(%)', '\uc0\u37197 \u24403 \u21033 \u22238 \u12426 (%)', '\u37197 \u24403 \u24615 \u21521 (%)']]\
    \
    # RandomForest\uc0\u12514 \u12487 \u12523 \u12398 \u27083 \u31689 \
    # \uc0\u30446 \u30340 \u22793 \u25968 y\u12399 \u12300 \u23558 \u26469 \u12398 \u22679 \u37197 \u26399 \u24453 \u20516 \u12301 \u12434 \u24819 \u23450 \u12375 \u12383 \u37325 \u12415 \u20184 \u12369 \
    model = RandomForestRegressor(n_estimators=50, random_state=42)\
    # \uc0\u12480 \u12511 \u12540 \u12398 \u12479 \u12540 \u12466 \u12483 \u12488 \u65288 ROE\u12364 \u39640 \u12367 \u37197 \u24403 \u24615 \u21521 \u12364 \u36969 \u27491 \u12394 \u12418 \u12398 \u12434 \u39640 \u35413 \u20385 \u12377 \u12427 \u12424 \u12358 \u12395 \u23398 \u32722 \u65289 \
    y = (df['ROE(%)'] * 0.6) + (df['\uc0\u37197 \u24403 \u21033 \u22238 \u12426 (%)'] * 0.4) - (df['\u37197 \u24403 \u24615 \u21521 (%)'] * 0.1)\
    \
    model.fit(X, y)\
    df['AI\uc0\u12473 \u12467 \u12450 '] = model.predict(X)\
    return df\
\
# --- 5. \uc0\u12513 \u12452 \u12531 \u34920 \u31034 \u20966 \u29702  ---\
data_load_state = st.text('\uc0\u12487 \u12540 \u12479 \u12434 \u21462 \u24471 \u20013 ...')\
raw_df = fetch_stock_data(TICKERS)\
scored_df = apply_ai_scoring(raw_df)\
data_load_state.empty()\
\
# \uc0\u12501 \u12451 \u12523 \u12479 \u12522 \u12531 \u12464 \u36969 \u29992 \
final_df = scored_df[\
    (scored_df['ROE(%)'] >= min_roe) &\
    (scored_df['\uc0\u37197 \u24403 \u21033 \u22238 \u12426 (%)'] >= min_yield) &\
    (scored_df['\uc0\u37197 \u24403 \u24615 \u21521 (%)'] <= max_payout)\
].sort_values(by='AI\uc0\u12473 \u12467 \u12450 ', ascending=False)\
\
# \uc0\u12499 \u12472 \u12517 \u12450 \u12523 \u34920 \u31034 \
col1, col2 = st.columns([2, 1])\
\
with col1:\
    st.subheader("\uc0\u55357 \u56522  AI\u35299 \u26512 \u32080 \u26524 \u12487 \u12540 \u12479 \u12475 \u12483 \u12488 ")\
    st.dataframe(final_df.style.background_gradient(subset=['AI\uc0\u12473 \u12467 \u12450 '], cmap='Greens'))\
\
with col2:\
    st.subheader("\uc0\u55357 \u56481  \u12509 \u12540 \u12488 \u12501 \u12457 \u12522 \u12458 \u27604 \u29575 \u25552 \u26696 ")\
    if not final_df.empty:\
        fig = px.pie(final_df, values='AI\uc0\u12473 \u12467 \u12450 ', names='\u37528 \u26564 \u21517 ', hole=0.3)\
        st.plotly_chart(fig, use_container_width=True)\
    else:\
        st.write("\uc0\u26465 \u20214 \u12395 \u19968 \u33268 \u12377 \u12427 \u37528 \u26564 \u12364 \u12354 \u12426 \u12414 \u12379 \u12435 \u12290 ")\
\
# --- 6. \uc0\u12501 \u12483 \u12479 \u12540 \u65288 \u20250 \u31038 \u12398 \u23455 \u24907 \u35388 \u26126 \u12475 \u12463 \u12471 \u12519 \u12531 \u65289  ---\
st.markdown("---")\
with st.expander("\uc0\u55356 \u57314  \u36939 \u21942 \u20250 \u31038 \u24773 \u22577 \u12362 \u12424 \u12403 \u12467 \u12531 \u12503 \u12521 \u12452 \u12450 \u12531 \u12473 \u12395 \u12388 \u12356 \u12390 "):\
    st.write(f"""\
    - **\uc0\u20250 \u31038 \u21517 :** [\u12354 \u12394 \u12383 \u12398 \u20250 \u31038 \u21517 \u12434 \u20837 \u12428 \u12427 ]\
    - **\uc0\u20195 \u34920 \u32773 :** \u20195 \u34920 \u21462 \u32224 \u24441  [\u12354 \u12394 \u12383 \u12398 \u27663 \u21517 ]\
    - **\uc0\u20107 \u26989 \u24418 \u24907 :** \u33258 \u24049 \u21208 \u23450 \u12395 \u12424 \u12427 \u36039 \u29987 \u36939 \u29992 \u65288 Proprietary Trading\u65289 \
    - **\uc0\u25216 \u34899 \u32972 \u26223 :** \u29987 \u26989 \u29992 \u33258 \u21205 \u21270 \u12471 \u12473 \u12486 \u12512 \u12398 \u35373 \u35336 \u24605 \u24819 \u12434 \u37329 \u34701 \u24037 \u23398 \u12395 \u24540 \u29992 \u12290 \
    - **\uc0\u27880 \u35352 :** \u26412 \u12471 \u12473 \u12486 \u12512 \u12399 \u33258 \u31038 \u36939 \u29992 \u23554 \u29992 \u12391 \u12354 \u12426 \u12289 \u22806 \u37096 \u12408 \u12398 \u25237 \u36039 \u21161 \u35328 \u12362 \u12424 \u12403 \u36039 \u37329 \u38928 \u35351 \u12398 \u21463 \u12369 \u20837 \u12428 \u12399 \u19968 \u20999 \u34892 \u12387 \u12390 \u12362 \u12426 \u12414 \u12379 \u12435 \u12290 \
    """)}