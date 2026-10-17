"""Derniers cours disponibles (intrajournaliers, différés de quelques minutes) via yfinance."""

import warnings

import pandas as pd
import yfinance as yf

from fxmacro.donnees.spots import colonne_cloture

COLONNES = ["cours", "horodatage", "intervalle"]


def derniers_cours(tickers: list[str]) -> pd.DataFrame:
    """Dernier cours et son horodatage pour chaque ticker (index : ticker).

    Une seule requête yfinance pour tous les tickers (5 min sur 5 jours) ; ceux qui
    manquent sont redemandés en journalier. Le week-end, c'est le dernier cours du
    vendredi. Les tickers introuvables sont absents du résultat (jamais d'exception :
    le tableau de bord se rabat alors sur le spot mensuel).
    """
    lignes: dict[str, dict] = {}
    for intervalle, periode in (("5m", "5d"), ("1d", "1mo")):
        manquants = [t for t in tickers if t not in lignes]
        if not manquants:
            break
        try:
            donnees = yf.download(manquants, period=periode, interval=intervalle,
                                  progress=False, auto_adjust=False)
        except Exception as erreur:  # bibliothèque tierce : erreurs réseau et de format variées
            warnings.warn(f"yfinance ({intervalle}) : {erreur.__class__.__name__}: {erreur}")
            continue
        for ticker in manquants:
            cloture = colonne_cloture(donnees, ticker)
            if not cloture.empty:
                lignes[ticker] = {"cours": float(cloture.iloc[-1]),
                                  "horodatage": pd.Timestamp(cloture.index[-1]), "intervalle": intervalle}
    resultat = pd.DataFrame.from_dict(lignes, orient="index", columns=COLONNES)
    resultat.index.name = "ticker"
    return resultat
