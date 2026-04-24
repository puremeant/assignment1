#사용하는 
import os
import requests
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import pandas_datareader.famafrench as ff

#코드 실행시 US stock 파일 자동 다운로드 
!pip install gdown
import gdown
url = "https://drive.google.com/uc?id=13qlkL0mEDhB5On3wsme17JOXwT7FGD97"
gdown.download(url, "US_stocks_2017_2025_CRSP.csv", quiet=False)
df_stock = pd.read_csv("US_stocks_2017_2025_CRSP.csv")

#파마프렌치 팩터는 홈페이지에서 직접 가져오기
import pandas_datareader.data as web
import pandas_datareader.famafrench as ff
datasets = ff.get_available_datasets()
