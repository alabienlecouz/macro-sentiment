"""Récupération de données depuis la BRI (BIS), API SDMX publique sans clé.

Jeux de données utilisés :
- WS_LONG_CPI : CPI mensuels, base 2010 = 100 (clé "M.{pays}.628")
- WS_CBPOL    : taux directeurs des banques centrales, en % (clé "M.{pays}")
"""

import io

import pandas as pd

from fxmacro.donnees.cache import obtenir_avec_cache
from fxmacro.donnees.http import telecharger

URL_BRI = "https://stats.bis.org/api/v1/data/{jeu}/{cle}"
ENTETE_CSV = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}


def lire_csv_bri(texte: str) -> pd.Series:
    """Transforme la réponse CSV SDMX de la BRI en série indexée au début du mois."""
    tableau = pd.read_csv(io.StringIO(texte))
    dates = pd.to_datetime(tableau["TIME_PERIOD"], format="%Y-%m")
    valeurs = pd.to_numeric(tableau["OBS_VALUE"], errors="coerce")
    return pd.Series(valeurs.to_numpy(), index=dates).sort_index()


def telecharger_bri(jeu: str, cle: str, date_debut: str) -> pd.Series:
    """Télécharge une série mensuelle BRI (ex. jeu='WS_CBPOL', cle='M.JP')."""
    reponse = telecharger(URL_BRI.format(jeu=jeu, cle=cle), params={"startPeriod": date_debut[:4]},
                          headers=ENTETE_CSV, timeout=60)
    serie = lire_csv_bri(reponse.texte)
    serie.name = f"{jeu}_{cle}"
    serie.index.name = "date"
    return serie


def obtenir_bri(jeu: str, cle: str, config: dict, forcer: bool = False) -> pd.Series:
    """Série BRI en passant par le cache."""
    return obtenir_avec_cache(f"bri_{jeu}_{cle}",
                              lambda: telecharger_bri(jeu, cle, config["general"]["date_debut"]),
                              config, forcer)


def obtenir_cpi_bri(pays: str, config: dict, forcer: bool = False) -> pd.Series:
    """CPI mensuel d'un pays (code ISO à 2 lettres, ex. 'JP')."""
    return obtenir_bri("WS_LONG_CPI", f"M.{pays}.628", config, forcer)


def obtenir_taux_directeur_bri(pays: str, config: dict, forcer: bool = False) -> pd.Series:
    """Taux directeur mensuel en % (code BRI, ex. 'US', 'XM' pour la zone euro)."""
    return obtenir_bri("WS_CBPOL", f"M.{pays}", config, forcer)
