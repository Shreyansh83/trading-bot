from financedatabase import Equities
from functools import lru_cache
import pandas as pd
import re

@lru_cache(maxsize=1)
def _load_equities() -> pd.DataFrame:
    """Load equity database for India once and cache it."""
    eq = Equities()
    return eq.select(country="India", exclude_exchanges=False)


def _normalize_query(q: str) -> str:
    """Remove exchange prefixes and Fyers suffixes."""
    q = q.strip()
    q = re.sub(r"^(NSE:|BSE:)", "", q, flags=re.IGNORECASE)
    q = re.sub(r"-EQ$", "", q, flags=re.IGNORECASE)
    return q

def search_symbols(query: str, limit: int = 10) -> pd.DataFrame:
    """Search Indian equities by symbol or name.

    Parameters
    ----------
    query: str
        Substring to search for.
    limit: int
        Maximum number of results to return.
    """
    if not query:
        return pd.DataFrame()
    df = _load_equities()
    # Only keep symbols listed on NSE or BSE to ensure Fyers compatibility
    df = df[df["exchange"].isin(["NSE", "BSE"])]
    q = _normalize_query(query)
    mask = df.index.str.contains(q, case=False) | df["name"].str.contains(q, case=False)
    return df[mask].head(limit)

def to_fyers_symbol(symbol: str, exchange: str | None = None) -> str:
    """Convert FinanceDatabase symbol and exchange to Fyers format."""
    if symbol.endswith(".NS") or (exchange and exchange.upper() == "NSE"):
        base = symbol[:-3] if symbol.endswith(".NS") else symbol
        return f"NSE:{base}-EQ"
    if symbol.endswith(".BO") or (exchange and exchange.upper() == "BSE"):
        base = symbol[:-3] if symbol.endswith(".BO") else symbol
        return f"BSE:{base}-EQ"
    return symbol
