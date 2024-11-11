# from fastapi import FastAPI
# import requests

# app = FastAPI()

# @app.get("/search")
# def search_finance(query: str):
#     url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
#     response = requests.get(url)
#     return response.json()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

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


# def search_finance(query: str):
#     url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         return {"error": str(e)}
#     except (ValueError, json.decoder.JSONDecodeError):
#         return {"error": "Could not parse response from Yahoo Finance API"}

if __name__ == "__main__":

    
    # Run the application
    uvicorn.run("stream:app", host="0.0.0.0", port=8000, reload=True)


###

# <!DOCTYPE html>
# <html>
# <head>
#     <title>Finance Search</title>
#     <style>
#         /* Add some basic styling */
#         body {
#             font-family: Arial, sans-serif;
#             padding: 20px;
#         }
#         input {
#             padding: 10px;
#             font-size: 16px;
#             width: 300px;
#         }
#         #results {
#             margin-top: 20px;
#         }
#         .result-item {
#             cursor: pointer;
#             padding: 5px;
#         }
#         .result-item:hover {
#             background-color: #f1f1f1;
#         }
#     </style>
# </head>
# <body>
#     <input type="text" id="searchInput" placeholder="Search stock symbol or company name">
#     <div id="results"></div>

#     <script>
#         const searchInput = document.getElementById('searchInput');
#         const resultsContainer = document.getElementById('results');

#         searchInput.addEventListener('input', async () => {
#             const query = searchInput.value;
#             const response = await fetch(`/search?query=${query}`);
#             const data = await response.json();
#             if (data.error) {
#                 resultsContainer.innerHTML = `<div style="color: red;">${data.error}</div>`;
#             } else {
#                 // Update the DOM with the search results
#                 resultsContainer.innerHTML = '';
#                 for (const quote of data.quotes) {
#                     const symbol = quote.symbol;
#                     const name = quote.shortname || quote.longname;
#                     const resultItem = document.createElement('div');
#                     resultItem.classList.add('result-item');
#                     resultItem.textContent = `${symbol} - ${name}`;
#                     resultItem.addEventListener('click', () => {
#                         resultsContainer.innerHTML = `<div>Selected symbol: ${symbol}</div>`;
#                     });
#                     resultsContainer.appendChild(resultItem);
#                 }
#             }
#         });
#     </script>
# </body>
# </html>