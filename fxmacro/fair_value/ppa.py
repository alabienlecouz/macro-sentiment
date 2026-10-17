"""Parité de pouvoir d'achat (PPA) relative.

Pour une paire XXXYYY (unités de YYY pour 1 XXX) :

    S_ppa(t) = k * P_YYY(t) / P_XXX(t)

Si les prix montent plus vite dans le pays YYY, il faut plus de YYY pour 1 XXX :
la devise YYY doit se déprécier. La constante k fixe le niveau (la PPA relative
ne dit rien du niveau absolu, seulement des variations). Deux façons de la caler :

- "date"    : k = S(t0) * P_XXX(t0) / P_YYY(t0), on suppose le spot "juste" en t0.
- "moyenne" : on suppose que la PPA tient en moyenne sur une longue période ;
              log k = moyenne[ log S(t) - log(P_YYY(t) / P_XXX(t)) ].
              Autrement dit, le taux de change réel est en moyenne à l'équilibre.
"""

import numpy as np
import pandas as pd


def ratio_prix(cpi_base: pd.Series, cpi_cotation: pd.Series) -> pd.Series:
    """Rapport des niveaux de prix P_YYY / P_XXX."""
    return cpi_cotation / cpi_base


def calibrer_constante(
    spot: pd.Series,
    ratio: pd.Series,
    methode: str = "moyenne",
    periode: tuple[str | None, str | None] = (None, None),
    date_base: str | None = None,
) -> float:
    """Calcule la constante k de la PPA relative."""
    log_reel = np.log(spot) - np.log(ratio)  # log du taux de change "réel" non normalisé
    if methode == "moyenne":
        debut, fin = periode
        fenetre = log_reel.loc[debut:fin].dropna()
        if fenetre.empty:
            raise ValueError("Période de calibration vide")
        return float(np.exp(fenetre.mean()))
    if methode == "date":
        if date_base is None:
            raise ValueError("date_base requise pour la méthode 'date'")
        # Première date disponible à partir de date_base
        valeurs = log_reel.loc[date_base:].dropna()
        if valeurs.empty:
            raise ValueError(f"Aucune donnée après {date_base}")
        return float(np.exp(valeurs.iloc[0]))
    raise ValueError(f"Méthode inconnue : {methode}")


def fair_value_ppa(
    spot: pd.Series,
    cpi_base: pd.Series,
    cpi_cotation: pd.Series,
    methode: str = "moyenne",
    periode: tuple[str | None, str | None] = (None, None),
    date_base: str | None = None,
) -> pd.Series:
    """Série de fair value PPA, sur le même index que les entrées (supposées alignées)."""
    ratio = ratio_prix(cpi_base, cpi_cotation)
    k = calibrer_constante(spot, ratio, methode, periode, date_base)
    fair_value = k * ratio
    fair_value.name = "fv_ppa"
    return fair_value
