import requests

def fetch_reddit_json(url):
    # Standardize URL
    if ".json" not in url:
        url = url.split('?')[0].rstrip('/') + ".json"
    
    # Fake a real browser header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"Scraper error: {e}")
        return None