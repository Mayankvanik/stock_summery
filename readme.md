# Stock Market Analysis and Prediction

This project offers a comprehensive analysis of stock market data, integrating machine learning models for price prediction, fundamental analysis, and sentiment analysis from recent news articles.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Machine Learning Models](#machine-learning-models)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The Stock Market Analysis and Prediction project aims to provide users with insightful analyses of stock performance through:

- Technical indicators such as current price, 52-week range, market capitalization, 50-day and 200-day moving averages, and volume analysis.
- Fundamental analysis including valuation metrics (e.g., P/E ratio), financial health indicators, and performance metrics like Return on Equity (ROE) and profit margins.
- Price predictions utilizing machine learning models.
- Sentiment analysis of recent news articles using advanced language models to assess strengths, weaknesses, risks, opportunities, and provide investment recommendations.

## Features

1. **Technical Analysis**
   - Displays current stock price, 52-week range, market capitalization, 50-day and 200-day moving averages, and volume analysis.

2. **Fundamental Analysis**
   - **Valuation:** Calculates Price-to-Earnings (P/E) ratio.
   - **Financial Health:** Assesses metrics such as debt-to-equity ratio and liquidity ratios.
   - **Performance:** Evaluates Return on Equity (ROE) and profit margins.

3. **Price Prediction**
   - Employs machine learning models to forecast future stock prices based on historical data.

4. **News Sentiment Analysis**
   - Fetches the latest news articles related to the selected stock.
   - Utilizes language models to analyze sentiment and provide insights into strengths, weaknesses, risks, opportunities, and investment recommendations for both short-term and long-term horizons.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Mayankvanik/stock_summery.git
   cd STOCK_ANALYSIS
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **API Keys Configuration**
   - Obtain necessary API keys for stock data and news retrieval.
   - Add openAI API key in app_html_main file in api /analyze-stock/:

5. **Run the project**
    - python3 app_html_main.py

## Usage

1. **Data Collection**
   - Run the data collection script to fetch historical stock data and recent news articles:
     ```bash
     python data_collection.py --ticker SYMBOL
     ```
     Replace `SYMBOL` with the stock ticker of interest.

2. **Analysis and Prediction**
   - Execute the main analysis script:
     ```bash
     python main_analysis.py --ticker SYMBOL
     ```
     This will generate technical and fundamental analyses, perform price prediction, and conduct news sentiment analysis.

3. **Results**
   - The results, including charts and analysis summaries, will be saved in the `results/SYMBOL/` directory.


## Data Sources

- **Stock Data:** yahoo fianance
- **News Articles:** Tavily search

## Machine Learning Models

- **Price Prediction:** Utilizes Random Forest networks for time series forecasting.
- **Sentiment Analysis:** Employs transformer-based language models to assess sentiment in news articles.

