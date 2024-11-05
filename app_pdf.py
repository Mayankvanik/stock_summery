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
    
    def tavily_news(self):

        # Get the full company name from the 'info' attribute
        company_info = self.stock.info
        company_name = company_info.get('longName')

        print("Full company name:", company_name)
        tavily_client = TavilyClient(api_key="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")
        response = tavily_client.search(f"{company_name} share news?")
        all_news = list(map(lambda i, x: f"News {i+1}: {x['content']}", range(len(response['results'])), response['results']))
        print('========',all_news)
        return all_news

    def llm_analysis(self, openai_key):
        """Generate LLM analysis using technical and fundamental data"""
        openai.api_key = openai_key
        client = OpenAI(api_key = openai_key)
        
        # Combine all data
        technical = self.fetch_technical_data()
        fundamental = self.fetch_fundamental_data()
        prediction = self.ml_prediction()
        news = self.tavily_news()
        
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
        6. Don't use any symbol
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

# from fpdf import FPDF
# from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        # Title styling
        self.set_font("Arial", "B", 14)
        self.set_text_color(33, 150, 243)  # Soft blue
        self.cell(0, 10, "Stock Analysis Report", ln=True, align='C')
        self.set_font("Arial", "", 10)
        self.set_text_color(100, 100, 100)  # Grey date
        self.cell(0, 10, 'Generated on: 28-10-2024', ln=True, align='C')
        self.ln(10)
        
    def add_section(self, title, content):
        # Section Title
        self.set_text_color(48, 63, 159)  # Indigo for headers
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True, align='L')
        self.set_line_width(0.5)
        self.set_draw_color(48, 63, 159)  # Line color
        self.line(10, self.get_y(), 200, self.get_y())  # Add an underline
        self.ln(4)

    def add_data_box(self, label, value):
        # Draw a box for each key-value pair
        self.set_draw_color(200, 200, 200)  # Light gray border
        self.set_fill_color(245, 245, 245)  # Light fill color
        self.set_font("Arial", "B", 10)
        self.cell(0, 10, f"{label}: {value}", ln=True, border=1, fill=True)
        self.ln(4)  

        # # Content Styling
        # self.set_text_color(55, 71, 79)  # Dark gray text
        # self.set_font("Arial", "", 10)
        # content = "\n".join([f"{key}: {value}" for key, value in content.items()]) if isinstance(content, dict) else content
        # self.multi_cell(0, 8, content)
        # self.ln(10)

def generate_pdf_report(results, filename='Stock_Analysis_Report.pdf'):
    pdf = PDFReport()
    pdf.add_page()

    # Styled Sections
    pdf.add_section("Technical Data Overview", str(results['Technical_Data']))
    pdf.add_section("Fundamental Insights", str(results['Fundamental_Data']))
    pdf.add_section("Machine Learning Prediction", str(results['ML_Prediction']))
    pdf.add_section("LLM-Driven Analysis", str(results['LLM_Analysis']))

    # Save PDF with a stylish print statement
    pdf.output(filename)
    print(f"The report has been beautifully saved as '{filename}'!")



# class PDFReport(FPDF):
#     def header(self):
#         self.set_font('Arial', 'B', 14)
#         self.cell(0, 10, 'Stock Analysis Report', 0, 1, 'C')
#         self.set_font('Arial', 'I', 10)
#         self.cell(0, 10, 'Generated on: 28-10-2024', 0, 1, 'C')
#         self.ln(10)

#     def add_section(self, title, content):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, title, 0, 1)
#         self.set_font('Arial', '', 10)
#         self.multi_cell(0, 10, content)
#         self.ln(10)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 8)
#         self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# def generate_pdf_report(results, filename='StockAnalysisReport.pdf'):
#     pdf = PDFReport()
#     pdf.add_page()

#     # Adding each section
#     pdf.add_section("Technical Data", str(results['Technical_Data']))
#     pdf.add_section("Fundamental Data", str(results['Fundamental_Data']))
#     pdf.add_section("ML Prediction", str(results['ML_Prediction']))
#     pdf.add_section("LLM Analysis", results['LLM_Analysis'])

#     # Save PDF
#     pdf.output(filename)
#     print(f"Report saved as {filename}")

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

def render_report_html(data, template_path='templates', output_path='stock_report02.html'):
    #env = Environment(loader=FileSystemLoader(template_path))
    env = Environment(
    loader=FileSystemLoader(template_path),
    autoescape=True  # Helps with character escaping in HTML
    )
    template = env.get_template('report_template.html')
    
    current_date = "2024-10-28 12:08:06" # datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = template.render(
        stock_symbol="JPPOWER.NS",
        current_date=current_date,
        metrics=data['Technical_Data'],
        fundamentals=data['Fundamental_Data'],
        prediction=data['ML_Prediction'],
        recommendation=data['LLM_Analysis']
    )
    
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
    SYMBOL = "IEX.NS"
    API_KEY = 'sk-proj-ac5LJh6fUwThrixXKhZyT3BlbkFJ6MklVui6AG9o95YXCvmB'
    OUTPUT_FILE = "stock_analysis02.txt"
    
    try:
        # Get analysis results
        results = analyze_stock(SYMBOL, API_KEY)
        #generate_pdf_report(results)
        render_report_html(data=results)
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