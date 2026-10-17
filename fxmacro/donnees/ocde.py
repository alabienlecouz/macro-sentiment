"""Données de l'OCDE (API SDMX publique, sans clé) : balance courante et termes de l'échange.

L'API OCDE limite le nombre de requêtes par heure : on télécharge toutes les zones
en une seule requête, puis on met chaque série en cache séparément.
"""

import warnings

import pandas as pd
import requests

from fxmacro.config import chemin_projet
from fxmacro.donnees.cache import age_cache_jours, ecrire_cache, enregistrer_repli, lire_cache
from fxmacro.donnees.http import telecharger
from fxmacro.donnees.sdmx import lire_csv_sdmx

URL_OCDE = "https://sdmx.oecd.org/public/rest/data/{flux}/{cle}"

# Balance courante, solde, en % du PIB, trimestriel, désaisonnalisé
FLUX_BP = "OECD.SDD.TPS,DSD_BOP@DF_BOP,1.0"
CLE_BP = "{zones}.WXD.CA.B.T.Q.PT_B1GQ.Y"
# Déflateurs des exportations (P6) et importations (P7), comptes nationaux trimestriels
FLUX_CNT = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1"
CLE_DEFLATEURS = "Q.Y.{zones}.S1.S1.{operation}._Z._Z._Z.IX.D.N.T0102"


def telecharger_ocde(flux: str, cle: str, date_debut: str) -> dict[str, pd.Series]:
    """Télécharge une requête OCDE et renvoie une série par zone (REF_AREA)."""
    parametres = {"startPeriod": date_debut[:4], "dimensionAtObservation": "AllDimensions",
                  "format": "csvfile"}
    reponse = telecharger(URL_OCDE.format(flux=flux, cle=cle), params=parametres, timeout=180)
    return lire_csv_sdmx(reponse.texte, colonne_groupe="REF_AREA")


def obtenir_par_zone(nom: str, flux: str, cle_modele: str, zones: list[str], config: dict,
                      forcer: bool = False, **format_cle) -> dict[str, pd.Series]:
    """Séries par zone, depuis le cache si possible, sinon en une seule requête."""
    dossier = chemin_projet(config["general"]["dossier_cache"])
    duree = config["general"]["duree_cache_jours"]
    if not forcer:
        en_cache = {z: lire_cache(dossier, f"ocde_{nom}_{z}", duree) for z in zones}
        if all(s is not None for s in en_cache.values()):
            return en_cache
    cle = cle_modele.format(zones="+".join(zones), **format_cle)
    try:
        series = telecharger_ocde(flux, cle, config["general"]["date_debut"])
    except requests.RequestException:
        # Même repli que obtenir_avec_cache, mais pour une requête multi-zones (quota OCDE)
        ages = {z: age_cache_jours(dossier, f"ocde_{nom}_{z}") for z in zones}
        if any(a is None or a > config["general"]["anciennete_max_cache_jours"] for a in ages.values()):
            raise
        for zone, age in ages.items():
            enregistrer_repli(f"ocde_{nom}_{zone}", age)
        warnings.warn(f"OCDE {nom} : téléchargement impossible ; dernier cache réutilisé.")
        return {z: lire_cache(dossier, f"ocde_{nom}_{z}", None) for z in zones}
    for zone, serie in series.items():
        serie.name = f"{nom}_{zone}"
        ecrire_cache(dossier, f"ocde_{nom}_{zone}", serie)
    manquantes = set(zones) - set(series)
    if manquantes:
        raise ValueError(f"OCDE {nom} : pas de données pour {sorted(manquantes)}")
    return series


def obtenir_balance_courante_ocde(zones: list[str], config: dict, forcer: bool = False) -> dict[str, pd.Series]:
    """Balance courante en % du PIB, trimestrielle, par zone."""
    return obtenir_par_zone("balance_courante", FLUX_BP, CLE_BP, zones, config, forcer)


def obtenir_termes_echange_ocde(zones: list[str], config: dict, forcer: bool = False) -> dict[str, pd.Series]:
    """Termes de l'échange = 100 × déflateur des exportations / déflateur des importations."""
    exports = obtenir_par_zone("deflateur_exports", FLUX_CNT, CLE_DEFLATEURS, zones, config, forcer,
                                operation="P6")
    imports = obtenir_par_zone("deflateur_imports", FLUX_CNT, CLE_DEFLATEURS, zones, config, forcer,
                                operation="P7")
    return {z: (100 * exports[z] / imports[z]).dropna() for z in zones}
