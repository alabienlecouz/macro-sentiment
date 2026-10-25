"""Historique long (depuis 1971) des cours et des CPI, pour le banc d'essai des fair values.

Pourquoi : les écarts à la fair value mettent plusieurs années à se résorber. Depuis 2009, il
n'y a qu'une quinzaine de périodes de 12 mois indépendantes : trop peu pour conclure.

Sources gratuites :
- cours mensuels FRED (moyennes mensuelles, série « EX… ») depuis 1971 ;
- EURUSD avant 1999 : « euro synthétique » = 1,95583 / (marks par dollar), approximation
  standard de la littérature ;
- CPI : sources habituelles du projet, téléchargées depuis 1960 ; zone euro avant le HICP :
  CPI allemand de la BRI, raccordé en proportion sur la période commune.

Cache séparé (data/cache/long) : le bot en production n'est pas modifié.
"""

import copy

import numpy as np
import pandas as pd

from fxmacro.donnees.alignement import (
    aligner_mensuel, annuel_vers_mensuel, decaler_publication, journalier_vers_mensuel,
)
from fxmacro.donnees.banque_mondiale import obtenir_indicateur_banque_mondiale
from fxmacro.donnees.bri import obtenir_cpi_bri
from fxmacro.donnees.cpi import obtenir_cpi_brut
from fxmacro.donnees.fred import obtenir_serie_fred
from fxmacro.donnees.spots import obtenir_spot

DEBUT_EURO = pd.Timestamp("1999-01-01")


def config_long(config: dict) -> dict:
    """Copie de la config pour l'historique long : date de début, dossier de cache, et paires
    et CPI supplémentaires (G10) qui n'existent pas dans le bot en production."""
    long = copy.deepcopy(config)
    params = config["historique_long"]
    long["general"]["date_debut"] = params["date_debut"]
    long["general"]["dossier_cache"] = params["dossier_cache"]
    long["paires"] = {**config["paires"], **params.get("paires_supplementaires", {})}
    long["cpi"] = {**config["cpi"], **params.get("cpi_supplementaires", {})}
    return long


def paires_long(config: dict) -> list[str]:
    """Paires de l'historique long : celles du bot plus les paires supplémentaires du G10."""
    return list(config_long(config)["paires"])


def raccorder_proportion(recente: pd.Series, ancienne: pd.Series) -> pd.Series:
    """Prolonge un indice dans le passé : l'ancienne série est remise à l'échelle de la récente
    (rapport moyen sur la période commune), puis utilisée avant le début de la récente."""
    commun = recente.dropna().index.intersection(ancienne.dropna().index)
    if commun.empty:
        raise ValueError("Pas de période commune pour raccorder les séries")
    facteur = float(np.exp((np.log(recente[commun]) - np.log(ancienne[commun])).mean()))
    avant = ancienne.loc[ancienne.index < recente.dropna().index[0]] * facteur
    return pd.concat([avant, recente.dropna()]).sort_index()


def euro_synthetique(marks_par_dollar: pd.Series, dem_par_eur: float) -> pd.Series:
    """Dollars pour 1 « euro » avant 1999 : le mark converti au taux fixe de l'euro."""
    return dem_par_eur / marks_par_dollar


def cours_mensuel_long(paire: str, config: dict) -> pd.Series:
    """Cours mensuel moyen depuis 1971 (FRED), prolongé par yfinance après la dernière date FRED."""
    params = config["historique_long"]["cours"][paire]
    long = config_long(config)
    serie = obtenir_serie_fred(params["fred"], long)
    if "avant_euro" in params:
        avant = params["avant_euro"]
        synthetique = euro_synthetique(obtenir_serie_fred(avant["fred"], long), avant["dem_par_eur"])
        serie = pd.concat([synthetique[synthetique.index < DEBUT_EURO], serie[serie.index >= DEBUT_EURO]])
    serie = serie.dropna().sort_index()
    recent = journalier_vers_mensuel(obtenir_spot(paire, long), config["alignement"]["methode_mensuelle"])
    return pd.concat([serie, recent[recent.index > serie.index[-1]]]).rename("spot")


def cpi_long(devise: str, config: dict) -> pd.Series:
    """CPI publié depuis 1960 (raccordé pour la zone euro), décalé de son délai de publication."""
    long = config_long(config)
    cpi = obtenir_cpi_brut(devise, long).resample("MS").last()
    code_avant = config["historique_long"].get("cpi_avant_hicp", {}).get(devise)
    if code_avant:
        cpi = raccorder_proportion(cpi, obtenir_cpi_bri(code_avant, long).resample("MS").last())
    return decaler_publication(cpi, long["cpi"][devise]["decalage"])


def productivite_long(devise: str, config: dict) -> pd.Series:
    """PIB réel par habitant (Banque mondiale), mensualisé avec son délai de publication."""
    params = config["historique_long"]["productivite"]
    series = obtenir_indicateur_banque_mondiale(params["indicateur"], list(params["pays"].values()),
                                                config_long(config))
    return annuel_vers_mensuel(series[params["pays"][devise]], params["decalage_apres_annee_mois"])


def tableau_long(paire: str, config: dict) -> pd.DataFrame:
    """Tableau mensuel sur l'historique long : spot, CPI et productivité de la base et de la cotation."""
    infos = config_long(config)["paires"][paire]
    return aligner_mensuel({"spot": cours_mensuel_long(paire, config),
                            "cpi_base": cpi_long(infos["base"], config),
                            "cpi_cotation": cpi_long(infos["cotation"], config),
                            "prod_base": productivite_long(infos["base"], config),
                            "prod_cotation": productivite_long(infos["cotation"], config)},
                           combler_max=config["alignement"]["anciennete_max_mois"], reference="spot")
