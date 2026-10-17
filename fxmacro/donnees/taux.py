"""Taux d'intérêt par devise (pour l'instant : taux directeurs BRI)."""

import pandas as pd

from fxmacro.donnees.bri import obtenir_taux_directeur_bri


def combler_regimes_sans_cible(serie: pd.Series, valeur: float | None) -> pd.Series:
    """Remplit les mois sans taux cible par une valeur fixe, sur toute la période couverte.

    Ex. : la BoJ n'avait pas de taux cible pendant ses régimes d'assouplissement
    quantitatif ; le taux du marché monétaire était alors quasi nul.
    Sans `valeur`, la série est seulement remise au pas mensuel (trous = NaN).
    """
    mensuel = serie.resample("MS").last()
    if valeur is None:
        return mensuel
    return mensuel.fillna(valeur)


def obtenir_taux(devise: str, config: dict, forcer: bool = False) -> pd.Series:
    """Taux directeur mensuel d'une devise, en %."""
    params = config["taux_directeurs"][devise]
    serie = obtenir_taux_directeur_bri(params["id"], config, forcer)
    return combler_regimes_sans_cible(serie, params.get("si_absent"))
