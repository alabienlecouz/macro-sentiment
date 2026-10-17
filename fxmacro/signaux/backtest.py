"""Backtest en temps réel (fenêtre croissante) du signal de valorisation.

Problème : dans le pipeline, la PPA, le BEER et le σ des z-scores ancrés sont calibrés
sur TOUT l'échantillon. À une date passée t, le z-score utilise donc des données
postérieures à t : l'historique du signal est trop flatteur (biais d'anticipation).

Ici, à chaque date t, on recalibre chaque modèle sur les seules données [.. t] :
- PPA : k_t = moyenne de ln S - ln(P_YYY / P_XXX) sur [.. t], σ_t sur la même fenêtre ;
- BEER : MCO de q sur [.. t], z(t) = dernier résidu / écart-type des résidus.
À la dernière date, la fenêtre croissante est l'échantillon entier : le signal du jour
est identique à celui du pipeline.

Évaluation : rendement de la paire sur `horizon` mois après chaque signal. Les
rendements à 12 mois de deux mois consécutifs se chevauchent sur 11 mois : on donne
aussi les statistiques sur deux sous-échantillons non chevauchants (le vrai nombre
d'observations indépendantes) :
- « calendaire » : un mois tous les `horizon` mois à partir du premier mois actif ;
- « glouton » : on garde un mois actif, on saute `horizon` mois, on recommence.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from fxmacro.fair_value.beer import construire_regresseurs, taux_change_reel


def regresseurs_beer(tableau: pd.DataFrame) -> pd.DataFrame:
    """Les trois fondamentaux relatifs du BEER (même définition que le pipeline)."""
    return construire_regresseurs(tableau)


def _dates_evaluees(index: pd.DatetimeIndex, debut: str) -> pd.DatetimeIndex:
    return index[index >= pd.Timestamp(debut)]


def z_ppa_temps_reel(tableau: pd.DataFrame, debut: str, min_obs: int = 60) -> pd.Series:
    """z ancré de la PPA recalibré à chaque date sur les seules données passées."""
    log_reel = (np.log(tableau["spot"]) - np.log(tableau["cpi_cotation"] / tableau["cpi_base"])).dropna()
    z = pd.Series(np.nan, index=tableau.index, name="z_ppa_tr")
    for t in _dates_evaluees(tableau.index, debut):
        fenetre = log_reel.loc[:t]
        if len(fenetre) <= min_obs or t not in fenetre.index:
            continue
        ecart = fenetre - fenetre.mean()
        z[t] = ecart.iloc[-1] / ecart.std()
    return z


def _residus_mco(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Résidus de la régression MCO de y sur X avec constante."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def z_beer_temps_reel(tableau: pd.DataFrame, debut: str, min_obs: int = 96,
                      pas_coint_mois: int = 12) -> pd.DataFrame:
    """z du BEER ré-estimé à chaque date sur [.. t], et p-value d'Engle-Granger (tous les 12 mois)."""
    q = taux_change_reel(tableau["spot"], tableau["cpi_base"], tableau["cpi_cotation"]).rename("q")
    donnees = pd.concat([q, regresseurs_beer(tableau)], axis=1).dropna()
    resultat = pd.DataFrame(np.nan, index=tableau.index, columns=["z", "p_coint"])
    for i, t in enumerate(_dates_evaluees(tableau.index, debut)):
        fenetre = donnees.loc[:t]
        if len(fenetre) <= min_obs or t not in fenetre.index:
            continue
        y, X = fenetre["q"].to_numpy(), fenetre.drop(columns="q").to_numpy()
        residus = _residus_mco(y, X)
        resultat.loc[t, "z"] = residus[-1] / residus.std(ddof=1)
        if i % pas_coint_mois == 0:
            resultat.loc[t, "p_coint"] = coint(y, X, trend="c")[1]
    # p-value reportée entre deux calculs (jamais vers le passé)
    resultat["p_coint"] = resultat["p_coint"].ffill().where(resultat["z"].notna())
    return resultat


def signal_depuis_z(z_val: pd.Series, seuil: float) -> pd.Series:
    """+1 si z ≤ -seuil (base sous-évaluée : hausse), -1 si z ≥ +seuil, 0 sinon (NaN compris)."""
    signal = pd.Series(0, index=z_val.index, dtype=int)
    signal[z_val <= -seuil] = 1
    signal[z_val >= seuil] = -1
    return signal


def _statistiques(r: pd.Series) -> tuple[int, float, float]:
    """n, taux de réussite (r > 0) et rendement moyen (en %) d'une série de rendements signés."""
    if r.empty:
        return 0, np.nan, np.nan
    return len(r), float((r > 0).mean()), float(100 * r.mean())


def dates_gloutonnes(echantillon: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Mois actifs non chevauchants : garder un mois actif, sauter `horizon` mois, recommencer."""
    gardees, i = [], 0
    while i < len(echantillon):
        if echantillon.iloc[i] != 0:
            gardees.append(echantillon.index[i])
            i += horizon
        else:
            i += 1
    return pd.DatetimeIndex(gardees)


def evaluer(signal: pd.Series, spot: pd.Series, horizon: int = 12) -> dict:
    """Qualité du signal : rendement signé ln(S(t+h) / S(t)) × signal sur les mois actifs.

    L'échantillon est celui des dates du signal dont le rendement futur est connu.
    """
    rendement = np.log(spot.shift(-horizon) / spot).reindex(signal.index)
    echantillon = signal[rendement.notna()]
    actifs = echantillon[echantillon != 0]
    r = actifs * rendement[actifs.index]
    n, reussite, moyenne = _statistiques(r)

    # Sous-échantillon non chevauchant : un mois tous les `horizon` mois, à partir du premier mois actif
    if actifs.empty:
        n_nc, reussite_nc, moyenne_nc = 0, np.nan, np.nan
    else:
        depart = echantillon.index.get_loc(actifs.index[0])
        dates_nc = echantillon.index[depart::horizon]
        actifs_nc = echantillon[dates_nc][echantillon[dates_nc] != 0]
        n_nc, reussite_nc, moyenne_nc = _statistiques(actifs_nc * rendement[actifs_nc.index])

    dates_g = dates_gloutonnes(echantillon, horizon)
    n_g, reussite_g, moyenne_g = _statistiques(echantillon[dates_g] * rendement[dates_g])

    return {
        "part_actifs": float(len(actifs) / len(echantillon)) if len(echantillon) else np.nan,
        "n": n, "taux_reussite": reussite, "rendement_moyen_%": moyenne,
        "n_non_chevauchant": n_nc, "taux_reussite_non_chevauchant": reussite_nc,
        "rendement_moyen_non_chevauchant_%": moyenne_nc,
        "n_glouton": n_g, "taux_reussite_glouton": reussite_g, "rendement_moyen_glouton_%": moyenne_g,
    }
