"""Portefeuille G10 value + carry : est-on rémunéré pour porter le risque de change ?

Convention : une série mensuelle indexée au début du mois m représente la fin du mois m.
Les poids décidés à la fin du mois t s'appliquent au rendement de t à t+1 (indexé t+1).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fxmacro.donnees.alignement import journalier_vers_mensuel
from fxmacro.donnees.bri import obtenir_bri
from fxmacro.donnees.cache import obtenir_avec_cache
from fxmacro.donnees.historique_long import config_long, paires_long, tableau_long
from fxmacro.donnees.spots import telecharger_spot
from fxmacro.signaux.apprentissage import taux_long
from fxmacro.signaux.backtest import ppa_temps_reel


# ---------- Données ----------

def cours_fin_mois(devise: str, config: dict) -> pd.Series:
    """Dollars pour une unité de la devise, fin de mois (BRI ; euro synthétique avant 1999)."""
    code = config["portefeuille"]["devises"][devise]
    par_dollar = obtenir_bri("WS_XRU", f"M.{code}.{devise}.E", config_long(config))
    return (1 / par_dollar).rename(devise)


def ecart_ppa_devise(devise: str, config: dict, debut: str) -> pd.Series:
    """Sous-évaluation de la devise face au dollar selon la PPA en temps réel (+ = bon marché).

    L'écart d'une paire est ln(S / FV) : positif quand la devise de base est surévaluée.
    """
    for paire, infos in config_long(config)["paires"].items():
        if paire not in paires_long(config) or "USD" not in (infos["base"], infos["cotation"]):
            continue
        if devise in (infos["base"], infos["cotation"]):
            ecart = ppa_temps_reel(tableau_long(paire, config), debut)["ecart"]
            return (-ecart if infos["base"] == devise else ecart).rename(devise)
    raise ValueError(f"Aucune paire contre le dollar pour {devise}")


def rendements_mensuels_actions(ticker: str, config: dict) -> pd.Series:
    """Rendement logarithmique mensuel de l'indice actions (cours de fin de mois)."""
    journalier = obtenir_avec_cache(f"actions_{ticker}", lambda: telecharger_spot(ticker, "1970-01-01"), config)
    fin_mois = journalier_vers_mensuel(journalier, "fin")
    return np.log(fin_mois).diff().rename("actions")


# ---------- Rendements et signaux ----------

def rendements_totaux(cours: pd.DataFrame, taux: pd.DataFrame, taux_usd: pd.Series) -> pd.DataFrame:
    """rx(t+1) = (i_k(t) - i_USD(t)) / 12 + ln S_k(t+1) - ln S_k(t), taux en %."""
    carry = taux.sub(taux_usd, axis=0) / 1200
    return carry.shift(1) + np.log(cours).diff()


def score_combine(carry: pd.DataFrame, value: pd.DataFrame) -> pd.DataFrame:
    """Moyenne des deux signaux standardisés en coupe (chaque mois, sur les devises disponibles)."""
    def standardiser(signal: pd.DataFrame) -> pd.DataFrame:
        return signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1), axis=0)
    return (standardiser(carry) + standardiser(value)) / 2


def poids_long_short(signal: pd.DataFrame, nb: int, min_devises: int) -> pd.DataFrame:
    """Achat des `nb` devises au signal le plus élevé, vente des `nb` plus faibles, poids égaux.

    Un mois avec moins de `min_devises` signaux disponibles n'a aucune position.
    """
    poids = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date, ligne in signal.iterrows():
        disponibles = ligne.dropna()
        if len(disponibles) < min_devises:
            continue
        classes = disponibles.sort_values()
        poids.loc[date, classes.index[-nb:]] = 1 / nb
        poids.loc[date, classes.index[:nb]] = -1 / nb
    return poids


def rendement_portefeuille(poids: pd.DataFrame, rendements: pd.DataFrame, cout_pb: float,
                           cout_roll_pb: float = 0.0) -> pd.DataFrame:
    """Rendement brut et net de coûts. Les poids de t s'appliquent au rendement indexé t+1.

    Coûts payés sur ce même rendement : `cout_pb` par unité de rotation |w(t) - w(t-1)|, et
    `cout_roll_pb` par mois sur le nominal brut Σ|w(t)| (renouvellement des contrats à terme à 1 mois).
    """
    poids_appliques = poids.shift(1)
    brut = (poids_appliques * rendements.reindex(poids.index)).sum(axis=1, min_count=1)
    rotation = poids.diff().abs().sum(axis=1).shift(1)
    nominal = poids_appliques.abs().sum(axis=1)
    actif = nominal > 0
    resultat = pd.DataFrame({"brut": brut, "rotation": rotation})
    resultat["net"] = resultat["brut"] - (cout_pb * resultat["rotation"] + cout_roll_pb * nominal) / 10_000
    return resultat[actif]


def ecart_taux_portefeuille(poids: pd.DataFrame, carry: pd.DataFrame) -> pd.Series:
    """Écart de taux moyen (points) entre les devises achetées et les devises vendues, chaque mois."""
    achetees = carry.where(poids > 0).mean(axis=1)
    vendues = carry.where(poids < 0).mean(axis=1)
    return (achetees - vendues).rename("ecart_taux")


# ---------- Statistiques ----------

def statistiques(r: pd.Series) -> dict:
    """Rendement et volatilité annualisés, Sharpe, t de Newey-West, pire mois, baisse maximale, asymétrie."""
    r = r.dropna()
    t = sm.OLS(r.to_numpy(), np.ones(len(r))).fit(cov_type="HAC", cov_kwds={"maxlags": 3}).tvalues[0]
    cumul = r.cumsum()
    return {"n_mois": len(r), "rendement_annuel_%": 1200 * r.mean(), "volatilite_%": 100 * np.sqrt(12) * r.std(),
            "sharpe": float(np.sqrt(12) * r.mean() / r.std()), "t_newey_west": float(t),
            "pire_mois_%": 100 * r.min(), "baisse_max_%": 100 * float((cumul - cumul.cummax()).min()),
            "asymetrie": float(r.skew())}


def prime_de_risque(r: pd.Series, actions: pd.Series) -> dict:
    """Comportement de la stratégie selon le marché actions : mois de baisse (< 0) contre mois de hausse."""
    d = pd.concat([r.rename("r"), actions], axis=1).dropna()
    baisse, hausse = d[d["actions"] < 0], d[d["actions"] >= 0]
    beta_baisse = sm.OLS(baisse["r"], sm.add_constant(baisse["actions"])).fit().params["actions"]
    return {"rendement_mois_baisse_actions_%": 100 * baisse["r"].mean(),
            "rendement_mois_hausse_actions_%": 100 * hausse["r"].mean(),
            "beta_mois_baisse": float(beta_baisse), "correlation_actions": float(d["r"].corr(d["actions"]))}
