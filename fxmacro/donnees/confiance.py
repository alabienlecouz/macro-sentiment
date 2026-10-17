"""Indicateur de confiance des entreprises de l'OCDE (BCI), proxy de la croissance.

Source : OCDE, jeu « Composite leading indicators »
    flux : OECD.SDD.STES,DSD_STES@DF_CLI,4.1
    clé  : {zones}.M.BCICP.IX._Z.AA.IX._Z.H
           (mensuel, confiance des entreprises, indice « amplitude adjusted » : 100 = moyenne
           de long terme ; > 100 = conjoncture au-dessus de sa tendance)

Pourquoi pas le CLI (indicateur avancé composite, MEASURE = LI) : l'OCDE ne le publie ni
pour la zone euro ni pour la Suisse. Le BCI existe pour les six devises (zone euro : EA20) :
on l'utilise partout pour garder une définition identique. Il suit le cycle avec moins
d'avance qu'un indicateur composite.

Limite : les séries de l'OCDE sont révisées après publication. Sans archives des versions
successives, l'historique utilise les valeurs révisées d'aujourd'hui, pas celles
réellement connues à chaque date.
"""

import pandas as pd

from fxmacro.donnees.alignement import decaler_publication
from fxmacro.donnees.ocde import obtenir_par_zone

FLUX_CONFIANCE = "OECD.SDD.STES,DSD_STES@DF_CLI,4.1"
CLE_CONFIANCE = "{zones}.M.BCICP.IX._Z.AA.IX._Z.H"


def obtenir_confiance_brute(config: dict, forcer: bool = False) -> dict[str, pd.Series]:
    """BCI mensuel par devise, tel que publié (sans décalage)."""
    params = config["regime"]["confiance"]
    zones = {devise: p["zone"] for devise, p in params.items()}
    series = obtenir_par_zone("confiance_entreprises", FLUX_CONFIANCE, CLE_CONFIANCE,
                              sorted(set(zones.values())), config, forcer)
    return {devise: series[zone] for devise, zone in zones.items()}


def obtenir_confiance(devise: str, config: dict, forcer: bool = False) -> pd.Series:
    """BCI d'une devise, décalé de son délai de publication (utilisable en temps réel)."""
    decalage = config["regime"]["confiance"][devise]["decalage"]
    return decaler_publication(obtenir_confiance_brute(config, forcer)[devise], decalage)
