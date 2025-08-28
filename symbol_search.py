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
    df = df[df["exchange"].isin(["NSE", "BSE"])]

    # Normalize Fyers-style inputs (e.g. "NSE:SBIN-EQ")
    q = query.strip().upper()
    for prefix in ("NSE:", "BSE:"):
        if q.startswith(prefix):
            q = q[len(prefix):]
    if q.endswith("-EQ"):
        q = q[:-3]

    mask = (
        df.index.str.contains(q, case=False, regex=False)
        | df["name"].str.contains(q, case=False, regex=False)
    )
    return df[mask].head(limit)

def to_fyers_symbol(symbol: str) -> str:
    """Convert FinanceDatabase symbol to Fyers format."""
    if symbol.endswith(".NS"):
        return f"NSE:{symbol[:-3]}-EQ"
    if symbol.endswith(".BO"):
        return f"BSE:{symbol[:-3]}-EQ"
    return symbol
