"""Modèle BEER (Behavioural Equilibrium Exchange Rate, Clark & MacDonald 1998).

Pour une paire XXXYYY, le taux de change réel (en log) est :

    q(t) = ln S(t) + ln P_XXX(t) - ln P_YYY(t)

q monte quand XXX s'apprécie en termes réels. On le régresse sur des fondamentaux
relatifs (XXX moins YYY) :

    q(t) = a + b1 · diff_taux_reel(t) + b2 · diff_termes_echange(t) + b3 · diff_balance_courante(t) + e(t)

Signes attendus, tous positifs :
- taux réel plus élevé       -> afflux de capitaux -> appréciation
- termes de l'échange meilleurs -> revenus d'exportation plus élevés -> appréciation
- excédent courant           -> accumulation d'actifs étrangers -> appréciation

q et les fondamentaux sont en général non stationnaires (I(1)). Une régression MCO en
niveaux n'a de sens que s'ils sont cointégrés, c'est-à-dire si le résidu e(t) est
stationnaire. Sinon, c'est une « régression fallacieuse » (Granger & Newbold, 1974).
On le vérifie avec le test d'Engle-Granger.

Fair value : ln FV(t) = q_estimé(t) - ln P_XXX(t) + ln P_YYY(t), et ln(S / FV) = e(t).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def taux_change_reel(spot: pd.Series, cpi_base: pd.Series, cpi_cotation: pd.Series) -> pd.Series:
    """ln S + ln P_XXX - ln P_YYY."""
    return np.log(spot) + np.log(cpi_base) - np.log(cpi_cotation)


def inflation_glissement_annuel(cpi: pd.Series) -> pd.Series:
    """Inflation sur 12 mois en % (index mensuel)."""
    return 100 * (cpi / cpi.shift(12) - 1)


def inflation_publiee(cpi_publie: pd.Series) -> pd.Series:
    """Inflation sur 12 mois calculée sur le seul CPI publié (mois non publiés = NaN).

    À calculer AVANT tout report : reporter le niveau du CPI puis calculer l'inflation
    diviserait un CPI ancien par une base de 12 mois plus récente (inflation inventée).
    C'est la valeur d'inflation calculée qu'on reporte ensuite.
    """
    return inflation_glissement_annuel(cpi_publie.resample("MS").last())


def taux_reel(taux_nominal_pct: pd.Series, cpi: pd.Series) -> pd.Series:
    """Taux réel ex post en % : taux nominal moins inflation sur 12 mois."""
    return taux_nominal_pct - inflation_glissement_annuel(cpi)


def construire_regresseurs(tableau: pd.DataFrame) -> pd.DataFrame:
    """Fondamentaux relatifs (base moins cotation) à partir des colonnes du tableau de la paire.

    Le taux réel utilise les colonnes `inflation_*` (calculées sur le CPI publié, puis
    reportées) ; à défaut, l'inflation est calculée sur les colonnes `cpi_*` du tableau.
    """
    def reel(cote: str) -> pd.Series:
        if f"inflation_{cote}" in tableau:
            return tableau[f"taux_{cote}"] - tableau[f"inflation_{cote}"]
        return taux_reel(tableau[f"taux_{cote}"], tableau[f"cpi_{cote}"])

    return pd.DataFrame({
        "diff_taux_reel": reel("base") - reel("cotation"),
        "diff_termes_echange": np.log(tableau["tde_base"]) - np.log(tableau["tde_cotation"]),
        "diff_balance_courante": tableau["bc_base"] - tableau["bc_cotation"],
    }, index=tableau.index)


@dataclass
class ResultatBEER:
    """Ce qu'on garde de l'estimation, pour l'afficher et le réutiliser."""
    coefficients: pd.Series
    t_stats_hac: pd.Series
    r2: float
    p_value_cointegration: float
    demi_vie_mois: float
    nb_obs: int
    q_estime: pd.Series


def demi_vie(residus: pd.Series) -> float:
    """Demi-vie (en périodes) d'un écart, à partir d'un AR(1) sur les résidus : ln(0,5) / ln(rho)."""
    r = residus.dropna()
    rho = np.corrcoef(r.iloc[1:], r.iloc[:-1])[0, 1]
    if rho <= 0 or rho >= 1:
        return float("inf") if rho >= 1 else 0.0
    return float(np.log(0.5) / np.log(rho))


def estimer_beer(q: pd.Series, regresseurs: pd.DataFrame,
                 periode: tuple[str | None, str | None] = (None, None)) -> ResultatBEER:
    """MCO de q sur les fondamentaux, écarts-types de Newey-West, test d'Engle-Granger."""
    donnees = pd.concat([q.rename("q"), regresseurs], axis=1).dropna()
    debut, fin = periode
    echantillon = donnees.loc[debut:fin]
    y = echantillon["q"]
    X = sm.add_constant(echantillon[regresseurs.columns])

    # Les résidus sont très autocorrélés : écarts-types HAC (Newey-West), 12 retards
    modele = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    # Engle-Granger : ADF sur les résidus, avec les valeurs critiques de MacKinnon adaptées
    _, p_coint, _ = coint(y, echantillon[regresseurs.columns], trend="c")

    q_estime = modele.predict(sm.add_constant(donnees[regresseurs.columns], has_constant="add"))
    return ResultatBEER(
        coefficients=modele.params,
        t_stats_hac=modele.tvalues,
        r2=float(modele.rsquared),
        p_value_cointegration=float(p_coint),
        demi_vie_mois=demi_vie(modele.resid),
        nb_obs=int(modele.nobs),
        q_estime=q_estime.reindex(q.index),
    )


def fair_value_beer(q_estime: pd.Series, cpi_base: pd.Series, cpi_cotation: pd.Series) -> pd.Series:
    """Repasse du taux réel d'équilibre au taux nominal : FV = exp(q_estimé) · P_YYY / P_XXX."""
    fair_value = np.exp(q_estime) * cpi_cotation / cpi_base
    fair_value.name = "fv_beer"
    return fair_value
