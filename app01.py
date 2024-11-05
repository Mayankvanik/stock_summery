import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime, timedelta
import openai  # For LLM analysis
from openai import OpenAI

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
            'Dividend_Yield': info.get('dividendYield')
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
        df['Target'] = df['Price'].shift(-days_ahead) / df['Price'] - 1
        df.to_csv('aa.csv',index=False)
        # Remove NaN values
        df = df.dropna()
        
        # Split data
        X = df.drop(['Target', 'Price'], axis=1)
        y = df['Target']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[:-days_ahead], y[:-days_ahead])
        
        # Make prediction
        latest_features = X.iloc[-1:]
        predicted_return = model.predict(latest_features)[0]
        current_price = df['Price'].iloc[-1]
        predicted_price = float(current_price * (1 + predicted_return))
        
        return {
            'Predicted_Return': predicted_return * 100,
            'Predicted_Price': predicted_price
        }
    
    def llm_analysis(self, openai_key):
        """Generate LLM analysis using technical and fundamental data"""
        openai.api_key = openai_key
        client = OpenAI(api_key = openai_key)
        
        # Combine all data
        technical = self.fetch_technical_data()
        fundamental = self.fetch_fundamental_data()
        prediction = self.ml_prediction()
        
        # Create prompt for LLM
        prompt = f"""
        Analyze the following stock data for {self.symbol}:
        
        Technical Indicators:
        {technical}
        
        Fundamental Data:
        {fundamental}
        
        ML Prediction:
        {prediction}
        
        Please provide:
        1. Overall analysis of the stock's current position
        2. Key strengths and weaknesses
        3. Potential risks and opportunities
        4. Investment recommendation (short-term and long-term)
        """
        
        # Get LLM analysis
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional stock analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content

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

#results = analyze_stock("AAPL",  )

# Add this if you want to run the script directly
# if __name__ == "__main__":
#     results = analyze_stock("ZOMATO.NS", "sk-proj-wpDzOQZMS9YzK2kHOqgzT3BlbkFJVC3FICDwsGZNDuFUvULL")
#     print("Technical Data:", results['Technical_Data'])
#     print("\nFundamental Data:", results['Fundamental_Data'])
#     print("\nML Prediction:", results['ML_Prediction'])
#     print("\nLLM Analysis:", results['LLM_Analysis'])

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
    OUTPUT_FILE = "stock_analysis.txt"
    
    try:
        # Get analysis results
        results = analyze_stock(SYMBOL, API_KEY)
        
        # Print to console
        print("Technical Data:", results['Technical_Data'])
        print("\nFundamental Data:", results['Fundamental_Data'])
        print("\nML Prediction:", results['ML_Prediction'])
        print("\nLLM Analysis:", results['LLM_Analysis'])
        
        # Save to file
        save_to_file(results, SYMBOL, OUTPUT_FILE)
        print(f"\nResults have been saved to {OUTPUT_FILE}")
        
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        
        # Save error to file
        with open(OUTPUT_FILE, 'a') as f:
            if f.tell() > 0:
                f.write('\n' + '='*50 + '\n\n')
            f.write(f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n{error_message}\n")