"""Parité des taux d'intérêt, couverte (CIP) et non couverte (UIP).

Pour une paire XXXYYY (unités de YYY pour 1 XXX), avec des taux annuels en décimal :

- CIP (arbitrage) : F(t, h) = S(t) * ((1 + i_YYY) / (1 + i_XXX)) ** h
  La devise au taux le plus élevé cote en déport à terme.

- UIP (hypothèse d'anticipations) : E_t[S(t+h)] = F(t, h).
  Fair value « UIP » d'aujourd'hui = forward observé il y a h :
      FV_uip(t) = S(t-h) * ((1 + i_YYY(t-h)) / (1 + i_XXX(t-h))) ** h
  Aucune information future : tout est connu en t-h.
"""

import pandas as pd


def pourcent_vers_decimal(taux_pct: pd.Series) -> pd.Series:
    """3.75 (%) -> 0.0375."""
    return taux_pct / 100


def forward_theorique(spot: pd.Series, i_base: pd.Series, i_cotation: pd.Series,
                      horizon_annees: float) -> pd.Series:
    """Forward CIP à l'horizon donné (taux en décimal)."""
    return spot * ((1 + i_cotation) / (1 + i_base)) ** horizon_annees


def prime_terme_annualisee(spot: pd.Series, forward: pd.Series, horizon_annees: float) -> pd.Series:
    """(F / S) ** (1 / h) - 1, proche de i_YYY - i_XXX.

    Le portage (carry) d'une position longue XXX financée en YYY en est l'opposé.
    """
    return (forward / spot) ** (1 / horizon_annees) - 1


def decaler_mois(serie: pd.Series, mois: int) -> pd.Series:
    """Ramène la valeur de t-mois à la date t (index mensuel au début du mois)."""
    decalee = serie.copy()
    decalee.index = decalee.index + pd.DateOffset(months=mois)
    return decalee.reindex(serie.index)


def fair_value_uip(spot: pd.Series, i_base: pd.Series, i_cotation: pd.Series,
                   horizon_mois: int) -> pd.Series:
    """Ce que l'UIP prévoyait il y a `horizon_mois` pour aujourd'hui (NaN au début)."""
    forward = forward_theorique(spot, i_base, i_cotation, horizon_mois / 12)
    fair_value = decaler_mois(forward, horizon_mois)
    fair_value.name = "fv_uip"
    return fair_value
