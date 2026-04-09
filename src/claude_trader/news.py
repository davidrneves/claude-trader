"""News feed integration via Alpaca News API.

Fetches recent headlines per symbol and feeds them to the sentiment agent.
"""

import structlog
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

log = structlog.get_logger()


class NewsFeed:
    """Fetches news headlines from Alpaca for sentiment analysis."""

    def __init__(self, api_key: str, secret_key: str) -> None:
        self._client = NewsClient(api_key=api_key, secret_key=secret_key)

    def get_headlines(self, symbol: str, limit: int = 10) -> list[str]:
        """Get recent news headlines for a symbol."""
        try:
            req = NewsRequest(symbols=symbol, limit=limit)
            news = self._client.get_news(req)
            headlines = []
            for articles in news.data.values():
                for a in articles:
                    headlines.append(a.headline)
            log.info("news_fetched", symbol=symbol, count=len(headlines))
            return headlines
        except Exception as e:
            log.warning("news_fetch_failed", symbol=symbol, error=str(e))
            return []
