"""Backtest du sentiment des banques centrales : ce que DeBERTa et FinBERT auraient dit dans le passé.

Convention des dates : une série mensuelle est indexée au début du mois m et représente
l'information disponible à la fin du mois m (cours de clôture, dernier communiqué publié).
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


def score_mensuel(scores: pd.DataFrame, banque: str, colonne: str, anciennete_max_mois: int) -> pd.Series:
    """Score du dernier communiqué publié avant la fin de chaque mois.

    Un communiqué publié le 17 septembre compte pour septembre ; il est reporté sur les mois
    suivants jusqu'au prochain, mais pas au-delà de `anciennete_max_mois`.
    """
    banque_scores = scores[scores["banque"] == banque].sort_values("date")
    mois = pd.to_datetime(banque_scores["date"]).dt.to_period("M").dt.to_timestamp()
    serie = pd.Series(banque_scores[colonne].to_numpy(), index=mois).groupby(level=0).last()
    index = pd.date_range(serie.index[0], pd.Timestamp.today().to_period("M").to_timestamp(), freq="MS")
    return serie.reindex(index).ffill(limit=anciennete_max_mois).rename(f"{banque}_{colonne}")


def ecart_centre(ecart: pd.Series, min_obs: int) -> pd.Series:
    """Écart de sentiment moins sa moyenne sur les mois passés (exclut le mois courant).

    Le style de rédaction diffère d'une banque à l'autre (la BCE peut paraître plus « hawkish »
    que la Fed à politique égale) : on ne garde que l'écart par rapport à l'habitude.
    """
    moyenne_passee = ecart.expanding(min_periods=min_obs).mean().shift(1)
    return (ecart - moyenne_passee).rename(f"{ecart.name}_centre")


def signal_sentiment(ecart: pd.Series, seuil: float) -> pd.Series:
    """+1 (hausse de la paire) si la banque de la devise de base est plus hawkish d'au moins
    `seuil`, -1 si elle est plus dovish d'au moins `seuil`, 0 sinon (NaN compris)."""
    signal = pd.Series(0, index=ecart.index, dtype=int)
    signal[ecart >= seuil] = 1
    signal[ecart <= -seuil] = -1
    return signal


def force_surcouche(signal_valorisation: pd.Series, ecart: pd.Series, seuil: float) -> pd.Series:
    """Règle du bot : la banque de la devise sous-évaluée est-elle plus hawkish que l'autre ?

    signal +1 : la devise de base est sous-évaluée, confirmé si Δs ≥ seuil ;
    signal -1 : la devise de cotation est sous-évaluée, confirmé si Δs ≤ -seuil.
    D'où un seul test sur signal × Δs. Renvoie confirme / contredit / modere (NaN hors signal).
    """
    produit = signal_valorisation * ecart.reindex(signal_valorisation.index)
    force = pd.Series(np.nan, index=signal_valorisation.index, dtype=object)
    actif = signal_valorisation != 0
    force[actif] = "modere"
    force[actif & (produit >= seuil)] = "confirme"
    force[actif & (produit <= -seuil)] = "contredit"
    return force


def correlation_taux(score: pd.Series, taux: pd.Series, horizon_mois: int, pas: int = 1) -> dict:
    """Corrélation de rang entre le score en t et la variation du taux directeur de t à t+h.

    pas = h : un mois tous les h mois, pour des variations qui ne se chevauchent pas
    (sinon la p-value surestime fortement la significativité).
    """
    variation = (taux.shift(-horizon_mois) - taux).rename("variation")
    donnees = pd.concat([score.rename("score"), variation], axis=1).dropna().iloc[::pas]
    rho, p = spearmanr(donnees["score"], donnees["variation"])
    return {"rho": float(rho), "p_value": float(p), "n": len(donnees)}


def surprise_ton(scores: pd.DataFrame, banque: str, colonne: str) -> pd.Series:
    """Changement de ton : score d'un communiqué moins celui du communiqué précédent de la même banque."""
    serie = scores[scores["banque"] == banque].sort_values("date").set_index("date")[colonne]
    return serie.diff().dropna().rename(f"surprise_{colonne}")


def reaction_evenement(spot: pd.Series, jour: pd.Timestamp, avant: int, apres: int, decalage: int = 0) -> float:
    """ln S(fin) - ln S(début) autour d'un communiqué, en jours de cotation.

    Le jour de l'événement est le premier jour coté à partir de `jour` ; la fenêtre va de
    `avant` jours avant à `apres` jours après. `decalage` > 0 recule toute la fenêtre (placebo).
    """
    position = spot.index.searchsorted(jour)
    debut, fin = position - avant - decalage, position + apres - decalage
    if debut < 0 or fin >= len(spot) or position >= len(spot):
        return np.nan
    return float(np.log(spot.iloc[fin] / spot.iloc[debut]))


def p_binomiale(n_reussites: int, n: int) -> float:
    """p-value bilatérale du test « taux de réussite = 50 % »."""
    return float(binomtest(n_reussites, n, 0.5).pvalue) if n else np.nan
