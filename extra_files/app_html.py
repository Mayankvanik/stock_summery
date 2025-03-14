import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime, timedelta
import openai  # For LLM analysis
from openai import OpenAI
from tavily import TavilyClient
from fpdf import FPDF
from jinja2 import Environment, FileSystemLoader
import re
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        
    def fetch_technical_data(self):
        """Fetch technical indicators and market data"""
        # Get historical data for past year
        hist = self.stock.history(period="1y")
        
        # Calculate technical indicators
        data = {
            'Current_Price': hist['Close'].iloc[-1],
            '52W_High': hist['High'].max(),
            '52W_Low': hist['Low'].min(),
            '50_Day_MA': hist['Close'].rolling(window=50).mean().iloc[-1],
            '30_Day_MA': hist['Close'].rolling(window=30).mean().iloc[-1],
            '200_Day_MA': hist['Close'].rolling(window=200).mean().iloc[-1],
            'Volume': hist['Volume'].iloc[-1],
            'Avg_Volume_3M': hist['Volume'].iloc[-63:].mean(),
            'Price_Change_1M': (hist['Close'].iloc[-1] / hist['Close'].iloc[-22] - 1) * 100,
            'RSI': self._calculate_rsi(hist['Close']).iloc[-1]
        }
        return data
    
    def fetch_fundamental_data(self):
        """Fetch fundamental company data"""
        info = self.stock.info
        fundamentals = {
            'Market_Cap': info.get('marketCap'),
            'PE_Ratio': info.get('trailingPE'),
            'Forward_PE': info.get('forwardPE'),
            'PEG_Ratio': info.get('pegRatio'),
            'Debt_to_Equity': info.get('debtToEquity'),
            'Current_Ratio': info.get('currentRatio'),
            'ROE': info.get('returnOnEquity'),
            'Profit_Margin': info.get('profitMargins'),
            'Revenue_Growth': info.get('revenueGrowth'),
            'Dividend_Yield': info.get('dividendYield'),
            'company_name': info.get('longName'),
        }
        return fundamentals
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def ml_prediction(self, days_ahead=30):
        """Make price prediction using Random Forest"""
        hist = self.stock.history(period="2y")
        
        # Create features
        df = pd.DataFrame()
        df['Price'] = hist['Close']
        df['Volume'] = hist['Volume']
        df['50_MA'] = df['Price'].rolling(window=50).mean()
        df['200_MA'] = df['Price'].rolling(window=200).mean()
        df['RSI'] = self._calculate_rsi(df['Price'])
        
        # Create target (future return)
        #df['Target'] = df['Price'].shift(-days_ahead) / df['Price'] - 1
        
        # Remove NaN values
        df = df.dropna()
        df.to_csv('aa02.csv',index=False)
        
        # Split data
        X = df.drop(['Price'], axis=1)
        y = df['Price']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        #model.fit(X[:-days_ahead], y[:-days_ahead])
        model.fit(X[:], y[:])
        
        # Make prediction
        latest_features = X.iloc[-1:]
        predicted_return = model.predict(latest_features)[0]
        current_price = df['Price'].iloc[-1]
        predicted_price = float(current_price * (1 + predicted_return))
        
        return {
    
            'Predicted_Price': predicted_return,
            "cureent":current_price
        }
    
    def tavily_news(self,k=4):

        # Get the full company name from the 'info' attribute
        company_info = self.stock.info
        company_name = company_info.get('longName')

        change_month = self.fetch_technical_data()
        change_month = change_month['Price_Change_1M']
        print("Full company name:", company_name , change_month)
        tavily_client = TavilyClient(api_key="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")
        sentiment = ''
        if change_month > 10:
            sentiment = 'hike'
        elif change_month < -9:
            sentiment = 'drop'
        else:
            sentiment = ''

        response = tavily_client.search(f"{company_name} share {sentiment} news?")
        all_news = list(map(lambda i, x: f"News {i+1}: {x['content']}", range(len(response['results'])), response['results']))[:k]
        print(f"{company_name} share {sentiment} news?",'========',all_news)
        return all_news

    def llm_analysis(self, openai_key):
        """Generate LLM analysis using technical and fundamental data"""
        openai.api_key = openai_key
        client = OpenAI(api_key = openai_key)
        
        # Combine all data
        technical = self.fetch_technical_data()
        fundamental = self.fetch_fundamental_data()
        prediction = self.ml_prediction()
        news = self.tavily_news() #'11111111111' #
        
        # Create prompt for LLM
        prompt = f"""
        Analyze the following stock data for {self.symbol}:

        Technical Indicators:
        {technical}

        Fundamental Data:
        {fundamental}

        ML Prediction:
        {prediction}

        Latest News:
        {news}

        Please provide:
        1. Overall analysis of the stock's current position
        2. Key strengths and weaknesses
        3. Potential risks and opportunities
        4. Investment recommendation (short-term and long-term)
        5. If Stock is High or low recently and any relation with news than metion sentement of result.
        """
        #6. All report give me with dict and value pair
        # Get LLM analysis
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional stock analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        
        output = response.choices[0].message.content
        
        response02 = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "just make key value pair dict formate from the content with key (strengths, weakness ,risk , opportunities, investment_recommendation_long-term, investment_recommendation_short-term, and sentiment) with out losing information of value include all information in value"},
                {"role": "user", "content": output}
            ]
        )
        output02 = response02.choices[0].message.content
        print('➡ output02 type:', type(output02))
        print('ooooooo',output,'pppppppp',output02)
        return  output02

def analyze_stock(symbol, openai_key):
    """Main function to run complete analysis"""
    analyzer = StockAnalyzer(symbol)
    
    results = {
        'Technical_Data': analyzer.fetch_technical_data(),
        'Fundamental_Data': analyzer.fetch_fundamental_data(),
        'ML_Prediction': analyzer.ml_prediction(),
        'LLM_Analysis': analyzer.llm_analysis(openai_key)
    }
    return results



def save_to_file(results, symbol, filename="stock_analysis.txt"):
    """Save analysis results to a file with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'a') as f:
        # Add separator if file is not empty
        if f.tell() > 0:
            f.write('\n' + '='*50 + '\n\n')
            
        f.write(f"Stock Analysis for {symbol} - {current_time}\n\n")
        
        # Write Technical Data
        f.write("Technical Data:\n")
        for key, value in results['Technical_Data'].items():
            f.write(f"{key}: {value}\n")
        
        # Write Fundamental Data
        f.write("\nFundamental Data:\n")
        for key, value in results['Fundamental_Data'].items():
            f.write(f"{key}: {value}\n")
        
        # Write ML Prediction
        f.write("\nML Prediction:\n")
        for key, value in results['ML_Prediction'].items():
            f.write(f"{key}: {value}\n")
        
        # Write LLM Analysis
        f.write("\nLLM Analysis:\n")
        f.write(results['LLM_Analysis'])
        f.write("\n")

def render_report_html(data , template_path='templates', output_path='stock_report02.html'):
    #env = Environment(loader=FileSystemLoader(template_path))
    env = Environment(
    loader=FileSystemLoader(template_path),
    autoescape=True  # Helps with character escaping in HTML
    )
    template = env.get_template('report_template.html')

    data = data

    input_str=data['LLM_Analysis']
    print('fffffffffff',input_str)
    json_match = re.search(r'\{.*\}', input_str, re.DOTALL)
    json_str = json_match.group(0) 
    data02 = json.loads(json_str)

    
    current_date = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html_content = template.render(
        stock_symbol=data['Fundamental_Data']['company_name'],#"JPPOWER.NS",
        current_date=current_date,
        metrics=data['Technical_Data'],
        fundamentals=data['Fundamental_Data'],
        prediction=data['ML_Prediction'],
        recommendation=data02
    )
    
    return html_content
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(html_content)

def analyze_stock(symbol, openai_key):
    """Main function to run complete analysis"""
    analyzer = StockAnalyzer(symbol)
    
    results = {
        'Technical_Data': analyzer.fetch_technical_data(),
        'Fundamental_Data': analyzer.fetch_fundamental_data(),
        'ML_Prediction': analyzer.ml_prediction(),
        'LLM_Analysis': analyzer.llm_analysis(openai_key)
    }
    
    return results

if __name__ == "__main__":
    # Configuration
    SYMBOL = "BAJAJ-AUTO.NS"
    API_KEY = 'sk-proj-ac5LJh6fUwThrixXKhZyT3BlbkFJ6MklVui6AG9o95YXCvmB'
    OUTPUT_FILE = "stock_analysis02.txt"
    
    try:
        # Get analysis results
        results = analyze_stock(SYMBOL, API_KEY)
        render_report_html(data = results)
        # Print to console
        print("Technical Data:", results['Technical_Data'])
        print("\nFundamental Data:", results['Fundamental_Data'])
        print("\nML Prediction:", results['ML_Prediction'])
        print("\nLLM Analysis:", results['LLM_Analysis'])
        
        # Save to file
        # save_to_file(results01, SYMBOL, OUTPUT_FILE)
        # print(f"\nResults have been saved to {OUTPUT_FILE}")
        
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        
        # Save error to file
        # with open(OUTPUT_FILE, 'a') as f:
        #     if f.tell() > 0:
        #         f.write('\n' + '='*50 + '\n\n')
        #     f.write(f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n{error_message}\n")


#             Please provide analysis in the following JSON format:

# {{
#     "overall_analysis": {{
#         "stock_name": "",
#         "current_price": "",
#         "price_movement": "",  # Up/Down with percentage from previous close
#         "market_position": "",  # Current market positioning based on MA
#         "predicted_price": ""   # ML model prediction
#     }},

#     "technical_analysis": {{
#         "rsi": {{
#             "value": "",
#             "indication": ""  # Overbought/Oversold/Neutral
#         }}
#     }},

#     "key_metrics": {{
#         "strengths": {{
#             "strength_1": "",
#             "strength_2": "",
#             "strength_3": ""
#         }},
#         "weaknesses": {{
#             "weakness_1": "",
#             "weakness_2": "",
#             "weakness_3": ""
#         }}
#     }},

#     "investment_recommendation": {{
#         "short_term": {{
#             "action": "",       # Buy/Sell/Hold
#             "rationale": "",
#             "risk_level": ""    # High/Medium/Low
#         }},
#         "long_term": {{
#             "action": "",       # Buy/Sell/Hold
#             "rationale": "",
#             "risk_level": ""    # High/Medium/Low
#         }}
#     }},

#     "sentiment_analysis": {{
#         "price_trend": "",      # High/Low recently
#         "news_impact": "",      # Positive/Negative/Neutral based on news
#         "overall_sentiment": "" # Bullish/Bearish/Neutral
#     }}
# }}"""