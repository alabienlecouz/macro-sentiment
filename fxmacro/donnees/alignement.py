"""Mise en cohérence des fréquences : spot journalier vs macro mensuelle.

Convention : toutes les séries mensuelles sont indexées au 1er jour du mois
(comme FRED), ce qui permet de les joindre directement.
"""

import pandas as pd


def journalier_vers_mensuel(serie: pd.Series, methode: str = "moyenne") -> pd.Series:
    """Agrège une série journalière en mensuel, indexée au début du mois.

    - "moyenne" : moyenne des cours du mois. Cohérent avec le CPI, qui est
      lui-même un niveau de prix moyen sur le mois.
    - "fin" : dernier cours du mois (ce qu'on observerait en temps réel).
    """
    groupes = serie.dropna().resample("MS")
    if methode == "moyenne":
        return groupes.mean()
    if methode == "fin":
        return groupes.last()
    raise ValueError(f"Méthode inconnue : {methode}")


def decaler_publication(serie_mensuelle: pd.Series, mois: int) -> pd.Series:
    """Décale une série macro de `mois` mois pour refléter le délai de publication.

    Le CPI de mars est publié mi-avril : au mois t, on ne connaît que le CPI de t-1.
    Sans ce décalage, on utiliserait une information future (biais d'anticipation).
    """
    if mois == 0:
        return serie_mensuelle.copy()
    decalee = serie_mensuelle.copy()
    decalee.index = decalee.index + pd.DateOffset(months=mois)
    return decalee


def aligner_mensuel(series: dict[str, pd.Series], combler_max: int = 1,
                    reference: str | None = None) -> pd.DataFrame:
    """Joint plusieurs séries mensuelles sur un index commun.

    Les trous isolés (au plus `combler_max` mois consécutifs) sont comblés par la
    dernière valeur connue ; les lignes encore incomplètes sont supprimées.
    Avec `reference` (ex. "spot"), on ne garde que les mois où cette série est
    réellement observée (elle n'est jamais prolongée artificiellement), et les autres
    colonnes peuvent rester NaN : chaque modèle n'utilise que les données dont il a
    besoin, sans que l'historique d'un modèle soit raccourci par celui d'un autre.
    """
    tableau = pd.concat(series, axis=1).sort_index()
    tableau = tableau.ffill(limit=combler_max)
    if reference is None:
        return tableau.dropna()
    return tableau.loc[tableau.index.isin(series[reference].dropna().index)]


def trimestriel_vers_mensuel(serie_trim: pd.Series, decalage_apres_trimestre: int) -> pd.Series:
    """Série trimestrielle (datée au début du trimestre) -> mensuelle utilisable en temps réel.

    La valeur du T1 (janvier-mars) n'est connue que `decalage_apres_trimestre` mois après
    la fin de mars. On la place à cette date puis on la reporte chaque mois jusqu'à la
    publication suivante (fonction en escalier, pas d'interpolation qui utiliserait le futur).
    """
    serie = serie_trim.dropna().copy()
    serie.index = serie.index + pd.DateOffset(months=2 + decalage_apres_trimestre)
    mois = pd.date_range(serie.index[0], serie.index[-1] + pd.DateOffset(months=2), freq="MS")
    return serie.reindex(mois).ffill()


def raccorder(recente: pd.Series, ancienne: pd.Series) -> pd.Series:
    """Prolonge `recente` dans le passé avec `ancienne`, recalée sur l'écart moyen du recouvrement."""
    recouvrement = recente.index.intersection(ancienne.index)
    if recouvrement.empty:
        raise ValueError("Pas de période commune pour raccorder les séries")
    ajustement = (recente.loc[recouvrement] - ancienne.loc[recouvrement]).mean()
    avant = ancienne.loc[ancienne.index < recente.index[0]] + ajustement
    return pd.concat([avant, recente]).sort_index()


def annuel_vers_mensuel(serie_annuelle: pd.Series, decalage_apres_annee: int) -> pd.Series:
    """Série annuelle (datée au 1er janvier) -> mensuelle utilisable en temps réel.

    La valeur de l'année Y n'est connue que `decalage_apres_annee` mois après la fin de Y :
    elle est placée à cette date puis reportée chaque mois jusqu'à la publication suivante
    (fonction en escalier, jamais d'interpolation, qui utiliserait l'année suivante).
    """
    serie = serie_annuelle.dropna().copy()
    serie.index = serie.index + pd.DateOffset(months=12 + decalage_apres_annee)
    mois = pd.date_range(serie.index[0], serie.index[-1] + pd.DateOffset(months=11), freq="MS")
    return serie.reindex(mois).ffill()
