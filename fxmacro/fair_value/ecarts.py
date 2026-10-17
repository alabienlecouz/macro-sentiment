"""Mesures d'écart entre le spot et une fair value (communes à tous les modèles)."""

import numpy as np
import pandas as pd


def ecart_pct(spot: pd.Series, fair_value: pd.Series) -> pd.Series:
    """Écart relatif du spot à la fair value : S / FV - 1.

    Positif : la devise de base (XXX dans XXXYYY) est surévaluée.
    Négatif : elle est sous-évaluée.
    """
    return spot / fair_value - 1


def zscore_glissant(serie: pd.Series, fenetre: int, min_periodes: int | None = None) -> pd.Series:
    """z(t) = (x(t) - moyenne glissante) / écart-type glissant, sur `fenetre` observations.

    La fenêtre inclut l'observation t et uniquement le passé : pas d'information future.
    """
    roulant = serie.rolling(fenetre, min_periods=min_periodes or fenetre)
    return (serie - roulant.mean()) / roulant.std()


def ecart_log(spot: pd.Series, fair_value: pd.Series) -> pd.Series:
    """Écart logarithmique ln(S / FV) ≈ écart en % pour de petits écarts, mais symétrique."""
    return np.log(spot / fair_value)


def zscore_ancre(ecart: pd.Series, periode: tuple[str | None, str | None] = (None, None)) -> pd.Series:
    """z(t) = écart(t) / σ, sans retirer de moyenne : la référence reste la fair value (écart nul).

    σ est l'écart-type de l'écart sur la période de calibration. Contrairement au
    z-score glissant, il mesure la distance au modèle, pas au passé récent.
    """
    debut, fin = periode
    ecart_type = ecart.loc[debut:fin].std()
    return ecart / ecart_type
