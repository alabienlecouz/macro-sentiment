"""PPA ajustée de la productivité (effet Balassa-Samuelson).

Un pays dont la productivité progresse plus vite voit ses salaires, donc les prix de ses
services non échangeables, monter : son niveau général des prix et son taux de change réel
s'apprécient durablement. La fair value n'est plus une constante réelle :

    q(t) = ln S(t) + ln P_XXX(t) - ln P_YYY(t) = a + b · (y_XXX(t) - y_YYY(t)) + e(t),   b > 0 attendu

avec y = ln(PIB réel par habitant), approximation standard de la productivité.
FV(t) = exp(a + b · Δy(t)) · P_YYY(t) / P_XXX(t), et ln(S / FV) = e(t).
"""

import numpy as np
import pandas as pd

from fxmacro.fair_value.beer import ResultatBEER, estimer_beer, fair_value_beer, taux_change_reel
from fxmacro.signaux.backtest import regression_q_temps_reel


def productivite_relative(prod_base: pd.Series, prod_cotation: pd.Series) -> pd.Series:
    """Δy = ln(PIB par habitant de XXX) - ln(PIB par habitant de YYY)."""
    return (np.log(prod_base) - np.log(prod_cotation)).rename("productivite_relative")


def taux_reel_et_productivite(tableau: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Taux de change réel q et régresseur Δy, à partir du tableau d'une paire."""
    q = taux_change_reel(tableau["spot"], tableau["cpi_base"], tableau["cpi_cotation"])
    return q, productivite_relative(tableau["prod_base"], tableau["prod_cotation"]).to_frame()


def estimer_ppa_productivite(tableau: pd.DataFrame) -> tuple[pd.Series, ResultatBEER]:
    """Estimation sur tout l'échantillon : fair value et résultats (b, t de Newey-West, cointégration)."""
    q, dy = taux_reel_et_productivite(tableau)
    resultat = estimer_beer(q, dy)
    return fair_value_beer(resultat.q_estime, tableau["cpi_base"], tableau["cpi_cotation"]), resultat


def ppa_productivite_temps_reel(tableau: pd.DataFrame, debut: str, min_obs: int = 60) -> pd.DataFrame:
    """a et b ré-estimés à chaque date sur les seules données passées : écart ln(S / FV) et z."""
    q, dy = taux_reel_et_productivite(tableau)
    return regression_q_temps_reel(q, dy, debut, min_obs)
