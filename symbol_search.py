from financedatabase import Equities
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=1)
def _load_equities() -> pd.DataFrame:
    """Load equity database for India once and cache it."""
    eq = Equities()
    return eq.select(country="India")

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
    df = df[df.index.str.endswith((".NS", ".BO"))]
    mask = df.index.str.contains(query, case=False) | df["name"].str.contains(query, case=False)
    return df[mask].head(limit)

def to_fyers_symbol(symbol: str) -> str:
    """Convert FinanceDatabase symbol to Fyers format."""
    if symbol.endswith(".NS"):
        return f"NSE:{symbol[:-3]}-EQ"
    if symbol.endswith(".BO"):
        return f"BSE:{symbol[:-3]}-EQ"
    return symbol
