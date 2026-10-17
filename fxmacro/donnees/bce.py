"""Données de la BCE (API Data Portal, sans clé).

Sert à compléter la balance courante de la zone euro avant 2013, période que
l'OCDE ne couvre pas pour la zone euro.
"""

import pandas as pd

from fxmacro.donnees.cache import obtenir_avec_cache
from fxmacro.donnees.http import telecharger
from fxmacro.donnees.sdmx import lire_csv_sdmx

URL_BCE = "https://data-api.ecb.europa.eu/service/data/{flux}/{cle}"

# Balance courante de la zone euro à 19 (I8), en millions d'EUR, non désaisonnalisée
BC_ZONE_EURO = ("BP6", "Q.N.I8.W1.S1.S1.T.B.CA._Z._Z._Z.EUR._T._X.N")
# PIB nominal de la zone euro (composition variable), millions d'EUR, désaisonnalisé
PIB_ZONE_EURO = ("MNA", "Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N")


def telecharger_bce(flux: str, cle: str, date_debut: str) -> pd.Series:
    reponse = telecharger(URL_BCE.format(flux=flux, cle=cle),
                          params={"format": "csvdata", "startPeriod": date_debut[:4], "detail": "dataonly"},
                          timeout=120)
    serie = lire_csv_sdmx(reponse.texte)
    serie.name = f"{flux}_{cle}"
    return serie


def obtenir_bce(flux: str, cle: str, config: dict, forcer: bool = False) -> pd.Series:
    return obtenir_avec_cache(f"bce_{flux}_{cle}",
                              lambda: telecharger_bce(flux, cle, config["general"]["date_debut"]),
                              config, forcer)


def balance_courante_zone_euro_pct_pib(config: dict, forcer: bool = False) -> pd.Series:
    """Balance courante / PIB en %, cumulée sur 4 trimestres (la saisonnalité s'annule)."""
    bc = obtenir_bce(*BC_ZONE_EURO, config, forcer)
    pib = obtenir_bce(*PIB_ZONE_EURO, config, forcer)
    return (100 * bc.rolling(4).sum() / pib.rolling(4).sum()).dropna()
