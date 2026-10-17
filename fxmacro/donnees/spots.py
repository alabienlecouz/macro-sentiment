"""Récupération des taux de change spot (journaliers) via yfinance."""

import pandas as pd
import yfinance as yf

from fxmacro.donnees.cache import obtenir_avec_cache
from fxmacro.donnees.http import ErreurTelechargement


def colonne_cloture(donnees: pd.DataFrame, ticker: str) -> pd.Series:
    """Cours de clôture d'un ticker, quel que soit le format renvoyé par yfinance.

    Selon la version et le nombre de tickers demandés, `donnees["Close"]` est une Series
    ou un DataFrame (une colonne par ticker).
    """
    if donnees.empty or "Close" not in donnees:
        return pd.Series(dtype=float)
    cloture = donnees["Close"]
    if isinstance(cloture, pd.DataFrame):
        if ticker in cloture.columns:
            cloture = cloture[ticker]
        elif cloture.shape[1] == 1:
            cloture = cloture.iloc[:, 0]
        else:
            return pd.Series(dtype=float)
    return cloture.dropna()


def telecharger_spot(ticker: str, date_debut: str) -> pd.Series:
    """Télécharge les cours de clôture journaliers d'un ticker yfinance (ex. 'EURUSD=X')."""
    donnees = yf.download(ticker, start=date_debut, progress=False, auto_adjust=False)
    cloture = colonne_cloture(donnees, ticker)
    if cloture.empty:
        # Erreur « réseau » : permet le repli sur le dernier cache
        raise ErreurTelechargement(f"Aucune donnée yfinance pour {ticker}")
    cloture.index = pd.to_datetime(cloture.index).tz_localize(None)
    cloture.name = ticker
    return cloture


def obtenir_spot(paire: str, config: dict, forcer: bool = False) -> pd.Series:
    """Spot journalier d'une paire de la config, en passant par le cache."""
    ticker = config["paires"][paire]["ticker"]
    return obtenir_avec_cache(f"spot_{ticker}",
                              lambda: telecharger_spot(ticker, config["general"]["date_debut"]),
                              config, forcer)
