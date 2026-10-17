"""Fondamentaux macro du modèle BEER, par devise, en fréquence mensuelle temps réel."""

import pandas as pd

from fxmacro.donnees.alignement import raccorder, trimestriel_vers_mensuel
from fxmacro.donnees.bce import balance_courante_zone_euro_pct_pib
from fxmacro.donnees.ocde import obtenir_balance_courante_ocde, obtenir_termes_echange_ocde


def _zones(config: dict) -> list[str]:
    return [params["zone_ocde"] for params in config["fondamentaux"].values()]


def obtenir_balance_courante(devise: str, config: dict) -> pd.Series:
    """Balance courante en % du PIB, moyenne sur 4 trimestres, mensualisée."""
    params = config["fondamentaux"][devise]
    trimestrielle = obtenir_balance_courante_ocde(_zones(config), config)[params["zone_ocde"]]
    trimestrielle = trimestrielle.rolling(4).mean().dropna()
    if params.get("complement_bce"):
        # L'OCDE ne couvre la zone euro qu'à partir de 2013 : on complète avec la BCE
        trimestrielle = raccorder(trimestrielle, balance_courante_zone_euro_pct_pib(config))
    return trimestriel_vers_mensuel(trimestrielle, config["beer"]["decalage_trimestriel_mois"])


def obtenir_termes_echange(devise: str, config: dict) -> pd.Series:
    """Termes de l'échange (prix des exportations / prix des importations), mensualisés."""
    zone = config["fondamentaux"][devise]["zone_ocde"]
    trimestriels = obtenir_termes_echange_ocde(_zones(config), config)[zone]
    return trimestriel_vers_mensuel(trimestriels, config["beer"]["decalage_trimestriel_mois"])
