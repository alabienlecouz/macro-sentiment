"""Banc d'essai commun des fair values : une fair value « price » bien si le cours revient vers elle.

Pour chaque modèle, l'écart en temps réel e(t) = ln(S(t) / FV(t)) est calculé avec une
fair value recalibrée à chaque date sur les seules données passées.

Critère 1 — pouvoir prédictif : régression r(t→t+h) = α + β·e(t) + ε, avec
r(t→t+h) = ln S(t+h) - ln S(t). β = -1 : écart entièrement refermé en h mois ; β = 0 : la fair
value n'apporte rien. Les rendements à h mois se chevauchent : écarts-types de Newey-West
(h retards) par paire, et de Driscoll-Kraay en panel (robustes au chevauchement ET aux chocs
communs à toutes les paires, ex. un mouvement du dollar).

Critère 2 — retour vers la fair value : test ADF de stationnarité de e(t) et demi-vie d'un AR(1).

Référence naïve : « fair value » = moyenne mobile du cours sur 5 ans. Un modèle qui ne fait pas
mieux qu'elle n'apporte rien de plus qu'un simple retour à la moyenne.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from fxmacro.fair_value.beer import demi_vie
from fxmacro.fair_value.ppa_productivite import ppa_productivite_temps_reel
from fxmacro.signaux.backtest import ppa_temps_reel, z_beer_temps_reel


def ecart_moyenne_mobile(spot: pd.Series, fenetre_mois: int) -> pd.Series:
    """Écart du cours à sa moyenne mobile (en log) sur les `fenetre_mois` derniers mois, passé seul."""
    log_spot = np.log(spot)
    return log_spot - log_spot.rolling(fenetre_mois, min_periods=fenetre_mois).mean()


def ecarts_temps_reel(tableau: pd.DataFrame, debut: str, fenetre_reference_mois: int,
                      modeles: tuple[str, ...] = ("ppa", "beer", "moyenne_mobile")) -> pd.DataFrame:
    """Écart ln(S / FV) en temps réel de chaque modèle demandé, sur les dates ≥ début.

    Le BEER a besoin des fondamentaux (taux, termes de l'échange, balance courante) ; sur
    l'historique long, seuls la PPA et la référence naïve sont disponibles.
    """
    calculs = {
        "ppa": lambda: ppa_temps_reel(tableau, debut)["ecart"],
        "beer": lambda: z_beer_temps_reel(tableau, debut)["ecart"],
        "moyenne_mobile": lambda: ecart_moyenne_mobile(tableau["spot"], fenetre_reference_mois),
        "ppa_productivite": lambda: ppa_productivite_temps_reel(tableau, debut)["ecart"],
    }
    ecarts = pd.DataFrame({modele: calculs[modele]() for modele in modeles})
    return ecarts.loc[ecarts.index >= pd.Timestamp(debut)]


def rendement_futur(spot: pd.Series, horizon_mois: int) -> pd.Series:
    """ln S(t+h) - ln S(t), aligné sur la date t (NaN quand t+h n'est pas encore connu)."""
    log_spot = np.log(spot)
    return log_spot.shift(-horizon_mois) - log_spot


def regression_predictive(ecart: pd.Series, rendement: pd.Series, horizon_mois: int) -> dict:
    """β, t de Newey-West (h retards), R² et nombre d'observations (total et indépendantes ≈ n / h)."""
    donnees = pd.concat([rendement.rename("r"), ecart.rename("e")], axis=1).dropna()
    if len(donnees) < 3 * horizon_mois:
        return {"beta": np.nan, "t": np.nan, "r2": np.nan, "n": len(donnees), "n_indep": len(donnees) // horizon_mois}
    modele = sm.OLS(donnees["r"], sm.add_constant(donnees["e"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon_mois})
    return {"beta": float(modele.params["e"]), "t": float(modele.tvalues["e"]), "r2": float(modele.rsquared),
            "n": int(modele.nobs), "n_indep": int(modele.nobs) // horizon_mois}


def standardiser_temps_reel(ecart: pd.Series, min_obs: int = 24) -> pd.Series:
    """Écart divisé par son écart-type passé (fenêtre croissante) : pas d'information future."""
    return ecart / ecart.expanding(min_periods=min_obs).std()


def regression_panel(ecarts: dict[str, pd.Series], rendements: dict[str, pd.Series], horizon_mois: int,
                     effets_fixes: bool = False, standardiser: bool = False) -> dict:
    """β commun à toutes les paires (MCO empilés), t de Driscoll-Kraay.

    Version 1 (par défaut) : MCO empilés bruts. Version 2 : effets fixes par paire (écart et
    rendement centrés sur la moyenne de chaque paire) et écarts standardisés en temps réel.
    La version 2 a été ajoutée APRÈS avoir vu les résultats de la version 1, où USDJPY (écart
    moyen +0,22, écart-type 2 à 3 fois celui des autres paires) dominait la pente commune.
    """
    morceaux = []
    for p in ecarts:
        e = standardiser_temps_reel(ecarts[p]) if standardiser else ecarts[p]
        bloc = pd.DataFrame({"r": rendements[p], "e": e}).dropna()
        if effets_fixes:
            bloc = bloc - bloc.mean()
        morceaux.append(bloc.assign(paire=p))
    empile = pd.concat(morceaux).sort_index()
    if empile.empty:
        return {"beta": np.nan, "t": np.nan, "r2": np.nan, "n": 0, "n_dates_indep": 0}
    dates = pd.factorize(empile.index)[0]
    modele = sm.OLS(empile["r"].to_numpy(), sm.add_constant(empile["e"].to_numpy())).fit(
        cov_type="hac-groupsum", cov_kwds={"time": dates, "maxlags": horizon_mois})
    return {"beta": float(modele.params[1]), "t": float(modele.tvalues[1]), "r2": float(modele.rsquared),
            "n": int(modele.nobs), "n_dates_indep": int(empile.index.nunique()) // horizon_mois}


def stationnarite(ecart: pd.Series) -> dict:
    """p-value du test ADF (avec constante) et demi-vie (mois) d'un AR(1) sur l'écart."""
    serie = ecart.dropna()
    if len(serie) < 24:
        return {"p_adf": np.nan, "demi_vie_mois": np.nan}
    return {"p_adf": float(adfuller(serie, regression="c", autolag="AIC", result_object=False)[1]),
            "demi_vie_mois": demi_vie(serie)}
