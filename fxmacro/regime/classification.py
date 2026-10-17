"""Classification de l'environnement macro en quatre régimes.

Règle (signe des dynamiques, cas nul compris dans « ≤ 0 ») :
    croissance ↑, inflation ↓ ou stable -> goldilocks
    croissance ↑, inflation ↑           -> reflation
    croissance ↓ ou stable, inflation ↑ -> stagflation
    croissance ↓ ou stable, inflation ↓ -> desinflation

Zone neutre (bruit) : une dynamique est considérée comme nulle si
|dynamique| < k × écart-type de cette dynamique, calculé en fenêtre CROISSANTE (seulement
le passé, donc point-in-time). Un mois où la croissance OU l'inflation est dans la zone
neutre ne propose aucun nouveau régime : on garde le précédent. k est fixé a priori.

Persistance (hystérésis) : un nouveau régime n'est retenu qu'après `mois_confirmation`
mois consécutifs ; sinon on garde l'ancien. Un mois neutre, ou dont la donnée n'est pas
encore publiée (valeur reprise, « reporte »), garde le régime en place et interrompt la
série de mois consécutifs. Le calcul avance dans le temps sans jamais regarder les mois suivants. Un
mois sans données (ou avant d'avoir assez d'historique pour l'écart-type) donne NaN :
aucun régime inventé, et la confirmation repart de zéro.

Outil descriptif : le régime est constaté avec retard (délai de publication +
mois de confirmation). Ce n'est pas une prévision.
"""

import numpy as np
import pandas as pd

GOLDILOCKS, REFLATION, STAGFLATION, DESINFLATION = "goldilocks", "reflation", "stagflation", "desinflation"
REGIMES = [GOLDILOCKS, REFLATION, STAGFLATION, DESINFLATION]
NEUTRE = "neutre"    # mois sans signal net : pas un régime, garde le régime précédent
REPORTE = "reporte"  # donnée du mois pas encore publiée (valeur reprise) : traité comme NEUTRE


def dans_zone_neutre(dynamique: pd.Series, ecarts_types: float, min_obs: int) -> pd.Series:
    """Vrai si |dynamique| < ecarts_types × écart-type passé (fenêtre croissante), NaN sinon connu."""
    ecart_type = dynamique.expanding(min_periods=min_obs).std()
    neutre = dynamique.abs() < ecarts_types * ecart_type
    return neutre.astype(object).where(ecart_type.notna() & dynamique.notna())


def regime_brut(croissance_dyn: float, inflation_dyn: float) -> str | float:
    """Régime d'un mois à partir du signe des deux dynamiques (NaN si l'une manque)."""
    if pd.isna(croissance_dyn) or pd.isna(inflation_dyn):
        return np.nan
    if croissance_dyn > 0:
        return REFLATION if inflation_dyn > 0 else GOLDILOCKS
    return STAGFLATION if inflation_dyn > 0 else DESINFLATION


def appliquer_persistance(bruts: pd.Series, mois_confirmation: int) -> pd.DataFrame:
    """Régime confirmé et nombre de mois passés dans ce régime, mois par mois."""
    regimes, durees = [], []
    actuel, duree, candidat, compte = None, 0, None, 0
    for brut in bruts:
        if pd.isna(brut):
            actuel, duree, candidat, compte = None, 0, None, 0
        elif brut in (NEUTRE, REPORTE):
            candidat, compte = None, 0
            duree += 1 if actuel is not None else 0
        elif brut == actuel:
            candidat, compte = None, 0
            duree += 1
        else:
            compte = compte + 1 if brut == candidat else 1
            candidat = brut
            if compte >= mois_confirmation:
                actuel, duree, candidat, compte = brut, 1, None, 0
            elif actuel is not None:
                duree += 1
        regimes.append(actuel if actuel is not None else np.nan)
        durees.append(duree if actuel is not None else np.nan)
    return pd.DataFrame({"regime": regimes, "mois_dans_regime": durees}, index=bruts.index)


def bruts_avec_zone_neutre(croissance_dyn: pd.Series, inflation_dyn: pd.Series,
                           ecarts_types: float, min_obs: int, reporte: pd.Series | None = None) -> pd.Series:
    """Régime brut de chaque mois ; NEUTRE si l'une des dynamiques est dans la zone neutre ;
    REPORTE si l'une est une valeur reprise (les valeurs reprises n'entrent pas dans l'écart-type)."""
    if reporte is None:
        reporte = pd.Series(False, index=croissance_dyn.index)
    neutre_c = dans_zone_neutre(croissance_dyn.mask(reporte), ecarts_types, min_obs)
    neutre_i = dans_zone_neutre(inflation_dyn.mask(reporte), ecarts_types, min_obs)
    bruts = []
    for c, i, nc, ni, rep in zip(croissance_dyn, inflation_dyn, neutre_c, neutre_i, reporte):
        if pd.isna(c) or pd.isna(i):
            bruts.append(np.nan)
        elif rep:
            bruts.append(REPORTE)
        elif pd.isna(nc) or pd.isna(ni):
            bruts.append(np.nan)
        elif nc or ni:
            bruts.append(NEUTRE)
        else:
            bruts.append(regime_brut(c, i))
    return pd.Series(bruts, index=croissance_dyn.index, dtype=object)


def classifier(indicateurs: pd.DataFrame, mois_confirmation: int,
               zone_neutre_ecarts_types: float = 0.0, zone_neutre_min_obs: int = 0) -> pd.DataFrame:
    """Ajoute regime_brut, regime (confirmé) et mois_dans_regime au tableau des indicateurs.

    Avec zone_neutre_ecarts_types = 0, pas de zone neutre : règle des signes seule.
    """
    bruts = bruts_avec_zone_neutre(indicateurs["croissance_dynamique"], indicateurs["inflation_dynamique"],
                                   zone_neutre_ecarts_types, zone_neutre_min_obs, indicateurs.get("reporte"))
    return pd.concat([indicateurs, bruts.rename("regime_brut"),
                      appliquer_persistance(bruts, mois_confirmation)], axis=1)
