# Import statements
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import openai
from openai import OpenAI
from tavily import TavilyClient
from fpdf import FPDF
from jinja2 import Environment, FileSystemLoader
import re
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import requests

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
    
    # ... [All other StockAnalyzer methods remain the same] ...

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
        'Current_Price': round(hist['Close'].iloc[-1], 2),
        '52W_High': round(hist['High'].max(), 2),
        '52W_Low': round(hist['Low'].min(), 2),
        '50_Day_MA': round(hist['Close'].rolling(window=50).mean().iloc[-1], 2),
        '30_Day_MA': round(hist['Close'].rolling(window=30).mean().iloc[-1], 2),
        '200_Day_MA': round(hist['Close'].rolling(window=200).mean().iloc[-1], 2),
        'Volume': round(hist['Volume'].iloc[-1], 2),
        'Avg_Volume_3M': round(hist['Volume'].iloc[-63:].mean(), 2),
        'Price_Change_1M': round((hist['Close'].iloc[-1] / hist['Close'].iloc[-22] - 1) * 100, 2),
        'RSI': round(self._calculate_rsi(hist['Close']).iloc[-1], 2)
        }
        return data
    
    def fetch_fundamental_data(self):
        """Fetch fundamental company data"""
        info = self.stock.info
        fundamentals = {
            'Market_Cap': round(info.get('marketCap'), 2) if info.get('marketCap') is not None else None,
            'PE_Ratio': round(info.get('trailingPE'), 2) if info.get('trailingPE') is not None else None,
            'Forward_PE': round(info.get('forwardPE'), 2) if info.get('forwardPE') is not None else None,
            'PEG_Ratio': round(info.get('pegRatio'), 2) if info.get('pegRatio') is not None else None,
            'Debt_to_Equity': round(info.get('debtToEquity'), 2) if info.get('debtToEquity') is not None else None,
            'Current_Ratio': round(info.get('currentRatio'), 2) if info.get('currentRatio') is not None else None,
            'ROE': round(info.get('returnOnEquity'), 2) if info.get('returnOnEquity') is not None else None,
            'Profit_Margin': round(info.get('profitMargins'), 2) if info.get('profitMargins') is not None else None,
            'Revenue_Growth': round(info.get('revenueGrowth'), 2) if info.get('revenueGrowth') is not None else None,
            'Dividend_Yield': round(info.get('dividendYield'), 2) if info.get('dividendYield') is not None else None,
            'company_name': info.get('longName')
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
        predicted_return = round(model.predict(latest_features)[0], 2)
        current_price = df['Price'].iloc[-1]
        predicted_price = float(current_price * (1 + predicted_return))
        
        return {
            'Predicted_Price': predicted_return,
            "current":current_price
        }
    
    def get_all_data(self):
        technical = self.fetch_technical_data()
        fundamental = self.fetch_fundamental_data()
        prediction = self.ml_prediction()
        news, news_links = self.tavily_news()
        return technical, fundamental, prediction, news, news_links

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
        news_link = list(map(lambda i, x: f"{x['url']}", range(len(response['results'])), response['results']))[:k]
        print(f"{company_name} share {sentiment} news?",'========',all_news)
        return all_news,news_link

    def dummy_llm_analysis(self, openai_key):
        """Generate LLM analysis using technical and fundamental data"""
        openai.api_key = openai_key
        client = OpenAI(api_key = openai_key)
        
        # Combine all data
        technical = self.fetch_technical_data()
        fundamental = self.fetch_fundamental_data()
        prediction = self.ml_prediction()
        news,url = self.tavily_news() #'11111111111' #
        
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
                {"role": "system", "content": "just make key value pair dict formate from the content with exact key name(strengths, weaknesses ,risk , opportunities, investment_recommendation_long-term, investment_recommendation_short-term, and sentiment) with out losing information of value include all information in value"},
                {"role": "user", "content": output}
            ]
        )
        output02 = response02.choices[0].message.content
        print('➡ output02 type:', type(output02))
        print('ooooooo',output,'pppppppp',output02)
        return  output02

    def llm_analysis(self, technical, fundamental, prediction, news,openai_key):
        """Generate LLM analysis using technical and fundamental data"""
        openai.api_key = openai_key
        client = OpenAI(api_key=openai_key)

        # Fetch all data
        #technical, fundamental, prediction, news, news_links = self.get_all_data(openai_key)

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
                {"role": "system", "content": "just make key value pair dict formate from the content with exact key name(strengths, weaknesses ,risk , opportunities, investment_recommendation_long-term, investment_recommendation_short-term, and sentiment) with out losing information of value include all information in value"},
                {"role": "user", "content": output}
            ]
        )
        output02 = response02.choices[0].message.content
        return output02

# Create FastAPI app
app = FastAPI()

# Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# origins = ["http://localhost:8000"]  # Add your allowed origins here
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["*"],
# )


# Setup templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class StockRequest(BaseModel):
    symbol: str

def setup_directories():
    """Create necessary directories and template files"""
    # Create templates directory
    Path("templates").mkdir(exist_ok=True)
    
    # Create static directory
    Path("static").mkdir(exist_ok=True)
    
    # Create default template file
    template_path = Path("templates/report_template.html")
    # if not template_path.exists():
    #     default_template = """
    #     <!DOCTYPE html>
    #     <html>
    #     <head>
    #         <title>Stock Analysis Report - {{stock_symbol}}</title>
    #     </head>
    #     <body>
    #         <h1>Stock Analysis Report for {{stock_symbol}}</h1>
    #         <p>Generated on: {{current_date}}</p>
            
    #         <h2>Technical Metrics</h2>
    #         <ul>
    #         {% for key, value in metrics.items() %}
    #             <li>{{key}}: {{value}}</li>
    #         {% endfor %}
    #         </ul>
            
    #         <h2>Fundamental Data</h2>
    #         <ul>
    #         {% for key, value in fundamentals.items() %}
    #             <li>{{key}}: {{value}}</li>
    #         {% endfor %}
    #         </ul>
            
    #         <h2>Price Prediction</h2>
    #         <ul>
    #         {% for key, value in prediction.items() %}
    #             <li>{{key}}: {{value}}</li>
    #         {% endfor %}
    #         </ul>
            
    #         <h2>Analysis and Recommendations</h2>
    #         <ul>
    #         {% for key, value in recommendation.items() %}
    #             <li>{{key}}: {{value}}</li>
    #         {% endfor %}
    #         </ul>
    #     </body>
    #     </html>
    #     """
    #     template_path.write_text(default_template)

    # Create index.html
    index_path = Path("templates/index.html")
    if not index_path.exists():
        index_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Analyzer</title>
        </head>
        <body>
            <h1>Stock Analyzer</h1>
            <form id="stockForm">
                <input type="text" id="symbol" placeholder="Enter stock symbol">
                <button type="submit">Analyze</button>
            </form>
            <div id="results"></div>

            <script>
                document.getElementById('stockForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const symbol = document.getElementById('symbol').value;
                    const response = await fetch('/analyze-stock/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ symbol: symbol })
                    });
                    const html = await response.text();
                    document.getElementById('results').innerHTML = html;
                });
            </script>
        </body>
        </html>
        """
        index_path.write_text(index_template)

def analyze_stock(symbol: str, openai_key: str):
    """Run complete stock analysis"""
    analyzer = StockAnalyzer(symbol)
    
    technical, fundamental, prediction, news, news_links = analyzer.get_all_data()
    
    results = {
        'Technical_Data': technical,
        'Fundamental_Data': fundamental,
        'ML_Prediction': prediction,
        'News_link': news_links,
        'LLM_Analysis': analyzer.llm_analysis(technical, fundamental, prediction, news,openai_key)
    }
    return results

def render_report_html(data , template_path='templates', output_path='stock_report02.html'):
    """Render HTML report from analysis results"""
    env = Environment(
        loader=FileSystemLoader(template_path),
        autoescape=True
    )
    template = env.get_template('report_template.html')

    # Parse LLM analysis JSON
    input_str = data['LLM_Analysis']
    print('fffffffffff',input_str)
    json_match = re.search(r'\{.*\}', input_str, re.DOTALL)
    json_str = json_match.group(0)
    data02 = json.loads(json_str)

    news_links = []
    for i, url in enumerate(data['News_link'], 1):
        news_links.append({
            'index': f'News {i}',
            'url': url
        })
    current_date = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('➡ news_links:', news_links)
    html_content = template.render(
        stock_symbol=data['Fundamental_Data']['company_name'],#"JPPOWER.NS",
        current_date=current_date,
        metrics=data['Technical_Data'],
        fundamentals=data['Fundamental_Data'],
        prediction=data['ML_Prediction'],
        news_links = news_links,
        recommendation=data02
    )
    return html_content

# import httpx
# @app.get("/api/yahoo-search")
# async def yahoo_search(query: str):
#     async with httpx.AsyncClient() as client:
#         response = await client.get(
#             f"https://query2.finance.yahoo.com/v1/finance/search", 
#             params={"q": query},
#             headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}
#         )
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         raise HTTPException(status_code=response.status_code, detail="Error fetching data from Yahoo Finance")

# @app.get("/proxy-yahoo-search")
# async def proxy_yahoo_search(query: str):
#     url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#     return response.json()

# async def fetch_suggestions(query: str):
#     async with httpx.AsyncClient() as client:
#         response = await client.get(
#             f"https://query2.finance.yahoo.com/v1/finance/search",
#             params={"q": query},
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
#             }
#         )
#         return response.json()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search")
def search_stock(query: str):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    params = {
        "q": query
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        # Print the results in a readable format
        print(f"\nResults for '{query}':")
        for quote in data.get('quotes', []):
            print(f"Symbol: {quote.get('symbol')} -> Name: {quote.get('shortname') or quote.get('longname')}")
            
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     """Serve the main page"""
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/suggestions", response_class=HTMLResponse)
# async def get_suggestions(request: Request, query: str):
#     """Handle stock suggestion requests"""
#     try:
#         suggestions = await fetch_suggestions(query)
#         return HTMLResponse(content=suggestions)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/analyze-stock/", response_class=HTMLResponse)
async def analyze_stock_endpoint(stock_request: StockRequest):
    """Handle stock analysis requests"""
    try:
        symbol = stock_request.symbol
        API_KEY = 'sk-proj-ac5LJh6fUwThrixXKhZyT3BlbkFJ6MklVui6AG9o95YXCvmB'  # Replace with your actual API key

        results = analyze_stock(symbol, API_KEY)
        html_content = render_report_html(results)
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Setup directories and files
    setup_directories()
    
    # Run the application
    uvicorn.run("app_html_main:app", host="0.0.0.0", port=8080, reload=True)


       

