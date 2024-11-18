from fastapi import FastAPI,  Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import yfinance as yf
import plotly.graph_objects as go
from plotly.io import to_json
import os
import uvicorn
import json
# # Create FastAPI app
# app = FastAPI()

# # Setup Jinja2 template renderer
# templates = Jinja2Templates(directory="templates")

# # StockAnalyzer class to handle stock analysis and plotting
class StockAnalyzer:
    def __init__(self, symbol, period=1):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        self.period = period
    
    def plot_stock_data(self):
        """Create and return stock data in JSON format for Plotly"""
        # Get historical data for the given period
        hist = self.stock.history(period=f"{self.period}y")
        
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='OHLC'
        ))
        
        fig.update_layout(
            title=f'{self.symbol} Stock Price',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark'
        )
        
        # Convert plot to JSON
        return to_json(fig)

# def save_plot_data_to_txt(data, file_path='plot_data_chart.txt'):
#     # Check if data['plot'] is a dictionary or a string
#     if isinstance(data, dict):
#         plot_data_str = json.dumps(data, indent=4)
#     else:
#         plot_data_str = data  # Assume it's already a JSON string if not a dictionary
    
#     # Write the JSON string to a file
#     with open(file_path, 'w') as file:
#         file.write(plot_data_str)
    
#     print(f"Plot data saved to {file_path}")

# # FastAPI route to serve the stock chart page
# @app.get("/stock/{symbol}", response_class=HTMLResponse)
# async def stock_page(request: Request, symbol: str, period: int = 1):
#     # Create StockAnalyzer object for the requested stock symbol
#     stock_analyzer = StockAnalyzer(symbol, period)
    
#     # Get the plot data as JSON
#     plot_data = stock_analyzer.plot_stock_data()
#     print('➡ plot_data type:', plot_data)
#     save_plot_data_to_txt(plot_data)
#     # Render HTML page with the chart
#     return templates.TemplateResponse("newindex.html", {
#         "request": request,
#         "plot_data": plot_data
#     })

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class StockRequest(BaseModel):
    symbol: str

# GET route
@app.get("/stock/{symbol}", response_class=HTMLResponse)
async def get_stock(request: Request, symbol: str):
    
    plot_data = await generate_plot_data(symbol)  # Your data generation function
    return templates.TemplateResponse(
        "newindex.html",
        {
            "request": request,
            "symbol": symbol,
            "plot_data": plot_data
        }
    )

# POST route
@app.post("/api/stock")
async def post_stock(stock_request: StockRequest,request: Request,):
    symbol = "AAPL"
    stock_analyzer = StockAnalyzer(symbol, period=1)
    
    # Get the plot data as JSON
    plot_data = stock_analyzer.plot_stock_data()
    print('➡ plot_data type:', plot_data)
    #plot_data = await generate_plot_data(stock_request.symbol)  # Your data generation function
    return templates.TemplateResponse(
        "newindex.html",
        {
            "request": request,

            "plot_data": plot_data
        }
    )

# Example plot data generation function
async def generate_plot_data(symbol: str):
    # Replace this with your actual data generation logic
    return {
        "data": [{
            "x": ["2024-01", "2024-02", "2024-03"],
            "y": [100, 110, 105],
            "type": "scatter"
        }],
        "layout": {
            "title": f"{symbol} Stock Price",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Price"}
        }
    }

# Run the app (run with 'uvicorn <filename>:app --reload')
if __name__ == "__main__":
 
    # Run the application
    uvicorn.run("chart:app", host="0.0.0.0", port=8000, reload=True)

