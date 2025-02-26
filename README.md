# Spotify Streaming Analytics Project

## Overview
This project analyzes Spotify streaming data to uncover audience patterns, measure campaign impact, and forecast future performance. Using statistical analysis and machine learning techniques, it delivers actionable insights for optimizing music release and promotional strategies.

## Data Description
The analysis uses Spotify audience timeline data containing:
- Daily streams
- Monthly listeners 
- Dates spanning 2023-2025

## Methodology
The project employs various analytical techniques:
1. **Exploratory Data Analysis**: Identifying trends, seasonality, and anomalies in streaming data
2. **Statistical Regression**: Measuring the impact of external factors on streaming performance
3. **Regression Diagnostics**: Validating statistical assumptions with residual analysis
4. **Machine Learning Forecasting**: Using Random Forest to predict future streaming performance
5. **Time Series Cross-Validation**: Ensuring robust model validation with multiple time-based folds
6. **Interactive Dashboard**: Visualizing insights and forecasts for business decision-making

## Key Findings
- Streaming performance shows clear day-of-week patterns with Tuesday being the strongest day
- Growth rate of 6.9% in monthly listeners over the analysis period
- Time series forecasting identifies 30-day moving averages as the strongest predictor of future streams
- Cross-validation confirms model reliability with consistent performance across time periods

## Running the Dashboard
To run the interactive dashboard:

1. Install required packages:
   ```
   pip install dash plotly pandas numpy statsmodels scikit-learn
   ```

2. Execute the dashboard script:
   ```
   python spotify_dashboard.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8052
   ```
