## Project Overview
This project implements a seven-factor asset pricing model using U.S. stock data from 2017–2025.
The main goal is to:
- Construction of a custom CAT factor
- Estimation of rolling betas (24-month window)
- Cross-sectional regressions (Pooled OLS, Fama-MacBeth)
- Portfolio sorting (5×5 portfolios based on CAT and HML betas)
- Time-series regressions
- GRS test (Gibbons, Ross, and Shanken, 1989)
- Comparison between 7-factor and 6-factor models

## Requirements
Install dependencies using:
pip install -r requirements.txt

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Run the main script: python main.py

## Data
- Stock data: CRSP monthly returns (2017–2025)
  - From https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/stock-version-2/monthly-stock-file/
  - Choose 'permno, ticker, mthcaldt, and mthret' as variables after reading variable description in Stock - Version 2 (CIZ)
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

## Output
- Regression results table
- Summary statistics of CAT factor
- Average absolute alpha across 25 portfolios
- GRS F-statistic and p-value
- 5x5 heatmap of portfolio alphas
- Comparison between 7-factor and 6-factor models

## Key Results Interpretation
- GRS test evaluates whether all portfolio alphas are jointly zero
- Lower average absolute alpha indicates better model fit
- Heatmaps help identify which factor dimension is not fully captured
