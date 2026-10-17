"""Point d'entrée unique pour les CPI, quelle que soit la source (FRED ou BRI)."""

import pandas as pd

from fxmacro.donnees.alignement import decaler_publication
from fxmacro.donnees.bri import obtenir_cpi_bri
from fxmacro.donnees.fred import obtenir_serie_fred


def obtenir_cpi_brut(devise: str, config: dict, forcer: bool = False) -> pd.Series:
    """CPI d'une devise tel que publié (sans décalage)."""
    params = config["cpi"][devise]
    if params["source"] == "fred":
        return obtenir_serie_fred(params["id"], config, forcer)
    if params["source"] == "bri":
        return obtenir_cpi_bri(params["id"], config, forcer)
    raise ValueError(f"Source CPI inconnue pour {devise} : {params['source']}")


def obtenir_cpi(devise: str, config: dict, forcer: bool = False) -> pd.Series:
    """CPI d'une devise, décalé de son délai de publication (utilisable en temps réel)."""
    return decaler_publication(obtenir_cpi_brut(devise, config, forcer), config["cpi"][devise]["decalage"])
