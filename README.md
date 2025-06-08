# Spotify Analytics Project

## Overview
This project analyzes Spotify streaming data from **Sakkaris** (my own band) to uncover audience patterns, measure campaign impact, and forecast future performance. Using statistical analysis and machine learning techniques, it delivers actionable insights for optimizing music release and promotional strategies.

## Data Description
The analysis uses proprietary Spotify for Artists data from Sakkaris, including:
- **Daily streams**
- **Monthly listeners**
- **Dates spanning 2023–2025**
- **Campaign data** (Marquee, Showcase, Discovery campaigns)

> **Note:** All data is sourced from Sakkaris’ Spotify for Artists dashboard.

## Methodology
The project employs a range of analytical techniques:
- **Exploratory Data Analysis:** Identifying trends, seasonality, and anomalies in streaming data
- **Statistical Regression:** Measuring the impact of external factors (e.g., campaigns) on streaming performance
- **Regression Diagnostics:** Validating statistical assumptions with residual analysis
- **Machine Learning Forecasting:** Using Random Forest to predict future streaming performance
- **Time Series Cross-Validation:** Ensuring robust model validation with multiple time-based folds
- **Interactive Dashboard:** Visualizing insights and forecasts for business decision-making
- **Automated Reporting:** Generating HTML reports and interactive visualizations

## Key Findings
- Streaming performance shows clear day-of-week patterns, with **Tuesday** being the strongest day
- **Monthly listener growth** of 19.7% (updated for monthly aggregation) over the analysis period
- Time series forecasting identifies **30-day moving averages** as the strongest predictor of future streams
- Cross-validation confirms model reliability with consistent performance across time periods
- Campaign analysis quantifies the impact and ROI of Marquee and Showcase campaigns

## Running the Dashboard

1. **Install required packages:**
    ```bash
    pip install dash plotly pandas numpy statsmodels scikit-learn
    ```

2. **(Optional) Run the full analysis and generate reports/visualizations:**
    ```bash
    python spotify_analytics_project.py
    ```
    This will create updated data and visualizations in the `output/` folder.

3. **Run the interactive dashboard:**
    ```bash
    python spotify_dashboard.py
    ```

4. **Open your web browser and navigate to:**
    ```
    http://127.0.0.1:8052
    ```

## Project Structure

- `spotify_analytics_project.py` — Main analysis and reporting script (data cleaning, EDA, modeling, report generation)
- `spotify_dashboard.py` — Interactive dashboard for exploring insights and forecasts
- `data/` — Raw data files (not included in repo)
- `output/` — Generated reports and visualizations

## About Sakkaris

This project is based on real streaming and campaign data from my band, **Sakkaris**. We're an indie rock/shoegaze band based in Los Angeles, CA. All analysis and insights are tailored to our actual Spotify for Artists data.

---
