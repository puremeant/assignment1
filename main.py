import os
import requests
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import pandas_datareader.famafrench as ff

# define winsorizer function
def winsorize_series(x):
    lower = x.quantile(0.01)
    upper = x.quantile(0.99)
    return x.clip(lower, upper)

# --- DATA SETTING STARTS --- #

# STOCK file #
## (Stock) Automatic downlaod of US stock file from Google Drive
!pip install gdown
import gdown
url = "https://drive.google.com/file/d/186hbPx4C3I7DefqLEvo2-HwlNkne3roC/view"
gdown.download(url, "stock.csv", quiet=False, fuzzy=True)
df_stock = pd.read_csv("stock.csv")

## (Stock) 컬럼을 우선 정리하고 날짜가 들어 있는 MthCalDt 컬럼을 날짜 형태로 변환 (period M 함수 적용)
df_stock.columns = df_stock.columns.str.strip().str.lower()
df_stock = df_stock.rename(columns={'mthcaldt': 'date'})
df_stock['date'] = pd.to_datetime(df_stock['date']).dt.to_period('M')
## change column 'mthret' to 'ret' (its name and into numeric value)
df_stock = df_stock.rename(columns={'mthret': 'ret'}) 
df_stock['ret'] = pd.to_numeric(df_stock['ret'], errors='coerce')

# Fama-French Factors #
## (F_F) 파마프렌치 팩터는 홈페이지에서 직접 가져오기
df_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench', start='2017-01-01', end='2025-12-31')
df_momentum = web.DataReader('F-F_Momentum_Factor', 'famafrench', start='2017-01-01', end='2025-12-31')

## (F_F) 파마프렌치 팩터 파일에서 날짜가 들어 있는 index 컬럼을 날짜 형태로 변환
df_factor = df_factors[0].reset_index()
df_mom = df_momentum[0].reset_index()
df_factor = df_factor.rename(columns={'Date': 'date'})
df_mom = df_mom.rename(columns={'Date': 'date'})
# 안전하게 astype str으로 해두기
df_factor['date'] = pd.to_datetime(df_factor['date'].astype(str)).dt.to_period('M') 
df_mom['date'] = pd.to_datetime(df_mom['date'].astype(str)).dt.to_period('M')

# 파일 병합 #
## 파마프렌치 두개 파일 병합하고 나서 stock 파일과 병합; 일치하는 것만 합치는 경우 how='inner'
df_facmom = pd.merge(df_factor, df_mom, on='date', how='inner')
df_stockfac = pd.merge(df_stock, df_facmom, on='date', how='inner')

## change unit in % into decimals
factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF','Mom']
df_stockfac[factor_cols] = df_stockfac[factor_cols] / 100

# --- DATA SETTING ENDS --- #
# excess return → winsorize → create CAT --- #

# Excess return 만들기 
df_stockfac['RF']= pd.to_numeric(df_stockfac['RF'], errors='coerce')
df_stockfac['excess_ret']= df_stockfac['ret'] - df_stockfac['RF']

# Winsorize
df_stockfac['excess_ret'] = df_stockfac.groupby('date')['excess_ret'].transform(winsorize_series)

# Create CAT
## CAT용 데이터 따로 만들기: C로 시작하는 티커 = L, T로 시작하는 티커 = S로 설정하는 'LS' 컬럼 생성
df_cat= df_stockfac.copy()
df_cat['ticker'] = df_cat['ticker'].astype(str).str.strip().str.upper()
df_cat['LS']= np.where(df_cat['ticker'].str.startswith('C'), 'L', np.where(df_cat['ticker'].str.startswith('T'), 'S', None))

#'LS'값 있는것만 남겨서 CAT = L - S
df_cat = df_cat[df_cat['LS'].isin(['L','S'])]
cat= df_cat.groupby(['date', 'LS'])['excess_ret'].mean().unstack()
cat['CAT'] = cat['L'] - cat['S']

## 다시 stockfac에 붙이기
df_stockfac = df_stockfac.merge(cat['CAT'], on='date', how='left')
df_stockfac.head()
