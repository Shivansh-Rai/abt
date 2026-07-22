import requests


def fetch_json(url: str):
    """Fetch a URL and return the parsed JSON response."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
