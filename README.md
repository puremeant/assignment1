## Project Overview
This project implements a seven-factor asset pricing model using U.S. stock data from 2017–2025.
The main goal is to:
- Construct the CAT factor (long C-tickers, short T-tickers)
- Estimate rolling 24-month factor loadings
- Test whether these factors explain cross-sectional stock returns

## Data
- Stock data: CRSP monthly returns (2017–2025)
  From https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/stock-version-2/monthly-stock-file/
  Choose 'permno, ticker, mthcaldt, and mthret' as variables after reading variable description in Stock - Version 2 (CIZ)
- Factor data: Fama-French 5 factors + Momentum (from Ken French Data Library)
  The stock dataset is automatically downloaded from Google Drive when running the code.

## Methodology
### Part A: Data Construction
- Construct CAT factor (C long, T short)
- Compute excess returns
- Estimate rolling 24-month betas for each stock
### Part B: Cross-sectional Regression
- Regress monthly excess returns on estimated betas
- Four specifications:
  - Pooled OLS
  - Fama-MacBeth
  - OLS with fixed effects
  - OLS with clustered standard errors
### Part C: GRS Test
- Sort stocks into 25 portfolios based on betas
- Run time-series regressions
- Perform GRS test to evaluate model performance

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Run the main script: python main.py

## Output
- Regression results table
- Summary statistics of CAT factor
- GRS test results
- 5x5 heatmap of portfolio alphas
