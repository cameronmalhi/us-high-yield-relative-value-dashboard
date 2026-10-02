# us-high-yield-relative-value-dashboard
Python dashboard analysing relative-value dislocations between US high-yield credit, equities, volatility and rates
## What it does

The dashboard compares:
- US High Yield OAS
- S&P 500
- VIX
- US 10-Year Treasury yield

It uses a rolling walk-forward regression to estimate the weekly HY spread move implied by wider market conditions. The difference between the actual and predicted spread move is converted into a residual z-score to identify potential relative-value dislocations.

## Features

- Live market data from FRED and Yahoo Finance
- Walk-forward regression
- Residual z-score signals
- Driver decomposition
- 4-week and 12-week historical signal analysis
- Market regime analysis
- Scenario testing

## Built with

Python, Pandas, NumPy, Plotly and Streamlit
## Live Demo

[Open the US High Yield Relative Value Dashboard](https://us-high-yield-relative-value-dashboard-7i4u2ebxela5lzfef8jck2.streamlit.app/)
