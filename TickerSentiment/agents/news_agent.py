import requests
from bs4 import BeautifulSoup

# 1. Use the specific ticker URL
url = "https://finviz.com/quote.ashx?t=AAPL"
headers = {'User-Agent': 'Mozilla/5.0'}

# 2. Fetch the page
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# 3. Find the news table
news_table = soup.find(id='news-table')

# 4. Loop through the rows to get the headlines
for row in news_table.findAll('tr'):
    a_tag = row.find('a')
    if a_tag:
        headline = a_tag.text
        link = a_tag['href']
        # The time is in the adjacent cell
        time_text = row.td.text.strip()
        print(f"{time_text} | {headline} | {link}")