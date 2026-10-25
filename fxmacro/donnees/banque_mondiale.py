"""Indicateurs annuels de la Banque mondiale (API publique, sans clé).

Utilisé pour le PIB réel par habitant (NY.GDP.PCAP.KD, dollars constants), approximation de la
productivité dans la PPA ajustée de la productivité (Balassa-Samuelson). Toutes les économies
sont demandées en une requête, puis mises en cache séparément.
Limite : données révisées, sans archives des versions successives.
"""

import pandas as pd

from fxmacro.config import chemin_projet
from fxmacro.donnees.cache import ecrire_cache, lire_cache
from fxmacro.donnees.http import telecharger

URL_BANQUE_MONDIALE = "https://api.worldbank.org/v2/country/{pays}/indicator/{indicateur}"


def lire_json_banque_mondiale(contenu: list) -> dict[str, pd.Series]:
    """Réponse JSON [métadonnées, observations] -> une série annuelle par pays (datée au 1er janvier)."""
    _, observations = contenu
    lignes = [(o["countryiso3code"], int(o["date"]), o["value"]) for o in observations or [] if o["value"] is not None]
    tableau = pd.DataFrame(lignes, columns=["pays", "annee", "valeur"])
    tableau["date"] = pd.to_datetime(tableau["annee"].astype(str) + "-01-01")
    return {pays: groupe.set_index("date")["valeur"].sort_index().rename(pays)
            for pays, groupe in tableau.groupby("pays")}


def telecharger_indicateur(indicateur: str, pays: list[str], debut: int) -> dict[str, pd.Series]:
    reponse = telecharger(URL_BANQUE_MONDIALE.format(pays=";".join(pays), indicateur=indicateur),
                          params={"format": "json", "per_page": 20000, "date": f"{debut}:2100"}, timeout=60)
    return lire_json_banque_mondiale(reponse.json())


def obtenir_indicateur_banque_mondiale(indicateur: str, pays: list[str], config: dict,
                                       forcer: bool = False) -> dict[str, pd.Series]:
    """Séries annuelles par pays, depuis le cache si possible, sinon en une seule requête."""
    dossier = chemin_projet(config["general"]["dossier_cache"])
    duree = config["general"]["duree_cache_jours"]
    noms = {p: f"banque_mondiale_{indicateur}_{p}" for p in pays}
    if not forcer:
        en_cache = {p: lire_cache(dossier, nom, duree) for p, nom in noms.items()}
        if all(s is not None for s in en_cache.values()):
            return en_cache
    series = telecharger_indicateur(indicateur, pays, int(config["general"]["date_debut"][:4]))
    manquants = set(pays) - set(series)
    if manquants:
        raise ValueError(f"Banque mondiale {indicateur} : pas de données pour {sorted(manquants)}")
    for p in pays:
        ecrire_cache(dossier, noms[p], series[p])
    return {p: series[p] for p in pays}
