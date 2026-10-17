"""Récupération des séries macro (CPI, taux) depuis FRED.

Avec une clé API (.env) on passe par l'API officielle ; sinon on se rabat sur
l'export CSV public de FRED, pratique pour tester sans clé.
"""

import io
import warnings

import pandas as pd

from fxmacro.config import cle_fred
from fxmacro.donnees.cache import obtenir_avec_cache
from fxmacro.donnees.http import telecharger

URL_API = "https://api.stlouisfed.org/fred/series/observations"
URL_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _via_api(id_serie: str, date_debut: str, cle: str) -> pd.Series:
    parametres = {
        "series_id": id_serie,
        "api_key": cle,
        "file_type": "json",
        "observation_start": date_debut,
    }
    # La clé passe en paramètre d'URL : `telecharger` la masque dans les messages d'erreur
    reponse = telecharger(URL_API, params=parametres, timeout=30)
    observations = reponse.json()["observations"]
    dates = pd.to_datetime([o["date"] for o in observations])
    # FRED code les valeurs manquantes par "."
    valeurs = pd.to_numeric([o["value"] for o in observations], errors="coerce")
    return pd.Series(valeurs, index=dates)


def _via_csv(id_serie: str, date_debut: str) -> pd.Series:
    reponse = telecharger(URL_CSV, params={"id": id_serie}, timeout=30)
    tableau = pd.read_csv(io.StringIO(reponse.texte), index_col=0, parse_dates=True)
    serie = pd.to_numeric(tableau.iloc[:, 0], errors="coerce")
    return serie[serie.index >= date_debut]


def telecharger_serie_fred(id_serie: str, date_debut: str) -> pd.Series:
    """Télécharge une série FRED (valeurs manquantes -> NaN)."""
    cle = cle_fred()
    if cle:
        serie = _via_api(id_serie, date_debut, cle)
    else:
        warnings.warn("Pas de FRED_API_KEY dans .env : utilisation de l'export CSV public.")
        serie = _via_csv(id_serie, date_debut)
    serie.name = id_serie
    serie.index.name = "date"
    return serie


def obtenir_serie_fred(id_serie: str, config: dict, forcer: bool = False) -> pd.Series:
    """Série FRED en passant par le cache."""
    return obtenir_avec_cache(f"fred_{id_serie}",
                              lambda: telecharger_serie_fred(id_serie, config["general"]["date_debut"]),
                              config, forcer)
