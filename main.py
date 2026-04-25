import os
import requests
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import pandas_datareader.famafrench as ff
import gdown
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# Define a winsorizer function
def winsorize_series(x):
    lower = x.quantile(0.01)
    upper = x.quantile(0.99)
    return x.clip(lower, upper)

# ---  Part A. 1. DATA SETTING --- #

# STOCK file #
## (Stock) Automatic downlaod of US stock file from Google Drive
!pip install gdown
url = "https://drive.google.com/file/d/186hbPx4C3I7DefqLEvo2-HwlNkne3roC/view"
gdown.download(url, "stock.csv", quiet=False, fuzzy=True)
df_stock = pd.read_csv("stock.csv")

## (Stock) Clean up and convert a 'MthCalDt' column into date format by period M function
df_stock.columns = df_stock.columns.str.strip().str.lower()
df_stock = df_stock.rename(columns={'mthcaldt': 'date'})
df_stock['date'] = pd.to_datetime(df_stock['date']).dt.to_period('M')
## change column 'mthret' to 'ret' (its name and into numeric value)
df_stock = df_stock.rename(columns={'mthret': 'ret'})
df_stock['ret'] = pd.to_numeric(df_stock['ret'], errors='coerce')

# Fama-French Factors #
## (F_F) File from the webpage directly
df_factor = web.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench', start='2017-01-01', end='2025-12-31')
df_momentum = web.DataReader('F-F_Momentum_Factor', 'famafrench', start='2017-01-01', end='2025-12-31')

## (F_F) Convert a 'Date' (index) column into date format and rename the columns
df_factor = df_factor[0].reset_index()
df_mom = df_momentum[0].reset_index()
df_factor = df_factor.rename(columns={'Date': 'date'})
df_mom = df_mom.rename(columns={'Date': 'date'})
# 'date' column into string type and a period M function
df_factor['date'] = pd.to_datetime(df_factor['date'].astype(str)).dt.to_period('M')
df_mom['date'] = pd.to_datetime(df_mom['date'].astype(str)).dt.to_period('M')

# Merging files #
## Merge two factor files and then merge with Stock file on 'data'; 'inner' to keep only matching records
df_facmom = pd.merge(df_factor, df_mom, on='date', how='inner')
df_stockfac = pd.merge(df_stock, df_facmom, on='date', how='inner')

## Change the unit % into decimals
factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF','Mom']
df_stockfac[factor_cols] = df_stockfac[factor_cols] / 100

# STOCKFAC file created
## for check: df_stockfac.head()

# ---  Part A. 2. CAT --- #

# excess return → winsorize → create CAT

## Excess return
df_stockfac['RF']= pd.to_numeric(df_stockfac['RF'], errors='coerce')
df_stockfac['excess_ret']= df_stockfac['ret'] - df_stockfac['RF']

## Winsorize
df_stockfac['excess_ret'] = df_stockfac.groupby('date')['excess_ret'].transform(winsorize_series)

## Create CAT
### Create a separate dataset for CAT: A 'LS' column where tickers starting with C are set to 'L', and T to 'S'
df_cat= df_stockfac.copy()
df_cat['ticker'] = df_cat['ticker'].astype(str).str.strip().str.upper() #to avoid case sensitivity 
df_cat['LS']= np.where(df_cat['ticker'].str.startswith('C'), 'L', np.where(df_cat['ticker'].str.startswith('T'), 'S', None))

## CAT= L - S; Filter out rows with missing 'LS' values and calculate the avg 'excess_ret' for stocks in each 'L' and 'S' group by date
df_cat = df_cat[df_cat['LS'].isin(['L','S'])]
cat= df_cat.groupby(['date', 'LS'])['excess_ret'].mean().unstack()
cat= cat.dropna(subset=['L', 'S']) #to avoid NaN
cat['CAT'] = cat['L'] - cat['S'] 

## Merge with a stockfac file
df_stockfac = df_stockfac.merge(cat['CAT'], on='date', how='left')



# ---  Part A. 3. REGRESSION --- #

## Store regression results incrementally in a 'results' column.
results = []
factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom', 'CAT']

## For each stock, sorting by date.
for permno, one_stock in df_stockfac.groupby('permno'):
    one_stock = one_stock.sort_values('date').reset_index(drop=True)

    # Rolling regression starts from t=24, because I use a 24-month estimation window.
    # That is, stocks with fewer than 24 months of prior data are excluded from rolling estimation. 
    # The first rolling estimates are generated from 2019 onward.
    for t in range (24, len(one_stock)):
        current = one_stock.loc[t, 'date']
        window = one_stock.iloc[t-24:t]
        reg_data = window[['excess_ret'] + factor_cols].dropna()

        # Exclude cases with fewer than 10 months of data (such as when a company is delisted)
        if len(reg_data) < 10:
            continue

        # Run a regression with excess return as the dependent and factors as independent variables.
        Y = reg_data['excess_ret']
        X = reg_data[factor_cols]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()

        # Create columns for the estimated betas and append the results
        results.append({
            'permno': permno, 'date': current,
            'beta_mkt': model.params['Mkt-RF'],
            'beta_smb': model.params['SMB'],
            'beta_hml': model.params['HML'],
            'beta_rmw': model.params['RMW'],
            'beta_cma': model.params['CMA'],
            'beta_mom': model.params['Mom'],
            'beta_cat': model.params['CAT']})
## Convert the accumulated results into a DataFrame
df_beta = pd.DataFrame(results)

df_beta.to_pickle('./df_beta.pkl')
df_stockfac.to_pickle('./df_stockfac.pkl')

## For check: df_beta.head()

# ---  Part B. 1. Cross-sectional Regression  --- #

## Set the dependent and independent variables
df_panel = pd.merge(df_stockfac, df_beta, on=['permno', 'date'], how='inner')
beta_cols= ['beta_mkt', 'beta_smb', 'beta_hml','beta_rmw', 'beta_cma', 'beta_mom', 'beta_cat']
reg_panel = df_panel[['date', 'permno', 'excess_ret'] + beta_cols].dropna()


# ---  Part B. 2. Four Specifications  --- #

## (a) Pooled OLS

y = reg_panel['excess_ret']
X = reg_panel[beta_cols]
X = sm.add_constant(X)

model_pooled = sm.OLS(y, X).fit()
print(model_pooled.summary())

## (b) Fama-MacBeth
### Since the Fama–MacBeth approach estimates a cross-sectional regression for each month, 
### we need to group the panel by date and run OLS within each monthly cross-section.

lambda_values = []
 
for month, month_data in reg_panel.groupby('date'):

    if len(month_data) <= len(beta_cols)+1:
        continue

    y = month_data['excess_ret']
    X = month_data[beta_cols]
    X = sm.add_constant(X)
    
    model_FM = sm.OLS(y, X).fit()
    params = model_FM.params
    params.name = month #store lambda values for each month
    lambda_values.append(params)

df_lambda = pd.DataFrame(lambda_values)
df_lambda.index.name = 'date'
### For check: df_lambda.head()

## The problem of getting lambdas in F-M is that its assumption on independent lambdas
### That is, standard error is under-estimated and t-stat is over-estimated, since lambdas are actually correlated.
### To address this issue, use Newey-West corrected standard errors, getting standard errors for each lambda in each month.

### autocovariance = 4 months 
### (lag0: Var(λt), lag1: Cov(λt, λt-1), lag2: Cov(λt, λt-2), lag3: Cov(λt, λt-3), lag4: Cov(λt, λt-4)
nw_lags = 4
nw_results = []

for lambda_col in df_lambda.columns:
    y = df_lambda[lambda_col].dropna()
    X = pd.DataFrame({'const':1}, index = y.index)

    ## HAC: heteroskedasticity + autocorrelation consistent
    model_NW = sm.OLS(y, X).fit(cov_type='HAC',cov_kwds={'maxlags': nw_lags})
    nw_results.append({
        'factor': lambda_col,
        'lambda_mean': model_NW.params['const'],
        'newey_west_se': model_NW.bse['const'],
        't_stat': model_NW.tvalues['const'],
        'p_value': model_NW.pvalues['const']})
    
nw_results = pd.DataFrame(nw_results)
print(nw_results)

## (c) Pooled OLS with date and stock fixed effects

y = reg_panel['excess_ret']
X = reg_panel[beta_cols]
X = sm.add_constant(X)

## similar to (a), but add C(permno) as fixed stock effect and C(date) as fixed date effect
reg_panel['date_str'] = reg_panel['date'].astype(str) # Change 'date' from Period[M] to string type
model_pooled_FE = smf.ols('excess_ret ~ beta_mkt + beta_smb + beta_hml + beta_rmw + beta_cma + beta_mom + beta_cat + C(permno) + C(date_str)', data=reg_panel).fit()
print(model_pooled_FE.summary())


## (d) (c) with standard errors two-way clustered by date and stock
### Considering both autocorrelation and cross-sectional correlation
model_pooled_FE_cluster = smf.ols('excess_ret ~ beta_mkt + beta_smb + beta_hml + beta_rmw + beta_cma + beta_mom + beta_cat + C(permno) + C(date_str)', data=reg_panel
                                  ).fit(cov_type='cluster', cov_kwds= {'groups': reg_panel[['permno', 'date_str']]})
print(model_pooled_FE_cluster.summary())


# ---  Part C. 1. Sort stocks into 25 portfolios  --- #

## Choose HML as a second factor
df_port = df_panel.copy()

## Create 5 different portfolios based on quantiles
df_port['cat_q'] = df_port.groupby('date')['beta_cat'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates ='drop'))
df_port['hml_q'] = df_port.groupby('date')['beta_hml'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
# Numbering portfolio from 1 to 25
df_port['portfolio_no'] = df_port['cat_q'] * 5 + df_port['hml_q'] + 1



# ---  Part C. 2. Equal-weighted monthly excess returns  --- #

## Returns by portfolio
port_returns = df_port.groupby(['date', 'portfolio_no'])['excess_ret'].mean().reset_index()



# ---  Part C. 3. Time-series regression --- #

## Create columns: date, 7 factors, 'port_returns' 
factor_cols = ['Mkt-RF','SMB','HML','RMW','CMA','Mom','CAT']
df_ts = port_returns.merge(df_port[['date'] + factor_cols].drop_duplicates(), on='date', how='left')

ts_results = []

for p, data in df_ts.groupby('portfolio_no'):
    y = data['excess_ret']
    X = sm.add_constant(data[factor_cols])

    model_ts = sm.OLS(y,X).fit()
    ts_results.append({'portfolio_no': p, 
                    'alpha': model_ts.params['const'], 
                    'alpha_t': model_ts.tvalues['const']})
    
df_alpha = pd.DataFrame(ts_results)
## For check: df_alpha.head()



# ---  Part C. 4. GRS TEST --- #

# Test the joint null hypothesis that all 25 alphas are zero 

## residuals
residuals =[]

for p, data in df_ts.groupby('portfolio_no'):
    y = data['excess_ret']
    X = sm.add_constant(data[factor_cols])

    model_r = sm.OLS(y, X).fit()
    residuals.append(model_r.resid.values)
residuals = np.array(residuals)

## Stats
sigma = np.cov(residuals)
alpha = df_alpha['alpha'].values.reshape(-1,1) #alpha vector

factors = df_ts[factor_cols].drop_duplicates().sort_values('date')
mu_f = factors.mean().values.reshape(-1,1) #mu vector
sigma_f = np.cov(factors.T) #covariance table

inv_sigma = np.linalg.pinv(sigma)
inv_sigma_f = np.linalg.pinv(sigma_f)

## GRS statistics

T = residuals.shape[1]
N = residuals.shape[0]
K = len(factor_cols)

numerator = alpha.T @ inv_sigma @ alpha
denominator = 1 + mu_f.T @ inv_sigma_f @ mu_f

GRS = ((T - N - K) / N) * (numerator / denominator)
GRS = float(GRS)

## p-value
from scipy.stats import f
p_value = 1 - f.cdf(GRS, N, T-N-K)



# ---  Part C. 5. Report --- #

## (a) GRS F-statistic and its p-value.
print("GRS F-statistic:", GRS)
print("p-value:", p_value)

## (b) The average absolute intercept across the 25 portfolios
avg_abs_alpha = np.mean(np.abs(df_alpha['alpha']))
print("Average absolute alpha:", avg_abs_alpha)

## (c) a 5 by 5 heatmap
### Add portfolio information to an alpha dataframe
df_alpha = df_alpha.merge(df_port[['portfolio_no', 'cat_q', 'hml_q']].drop_duplicates(), on='portfolio_no', how='left')
### pivot table
heatmap_data = df_alpha.pivot(index='cat_q', columns='hml_q', values='alpha')

### heat map
plt.figure(figsize=(6,5))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0)
plt.title("Alpha Heatmap (5X5 Portfolios)")
plt.xlabel("HML Quintile")
plt.ylabel("CAT Quintile")
plt.show()



# ---  Part C. 6. Discuss --- #
