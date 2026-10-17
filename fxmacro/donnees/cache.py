"""Cache local en parquet pour éviter de re-télécharger à chaque exécution.

Un seul point d'entrée, `obtenir_avec_cache`, utilisé par toutes les sources :
- cache frais -> lu tel quel ;
- cache périmé -> on re-télécharge ; si le réseau échoue, on réutilise le cache périmé
  (avec un avertissement) plutôt que de perdre toute la paire ;
- écriture atomique : un arrêt brutal (timeout systemd, manque de RAM) ne laisse
  jamais un parquet à moitié écrit.

Chaque repli sur un cache périmé est enregistré (`sources_perimees`) pour être signalé
dans maj.json et sur le tableau de bord. Au-delà de `anciennete_max_cache_jours`,
le cache n'est plus servi : l'erreur est relancée.
"""

import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests

from fxmacro.config import chemin_projet


# Sources servies depuis un cache périmé pendant l'exécution en cours
_SOURCES_PERIMEES: list[dict] = []


def reinitialiser_suivi() -> None:
    """Vide la liste des sources périmées (au début de chaque mise à jour)."""
    _SOURCES_PERIMEES.clear()


def sources_perimees() -> list[dict]:
    """Sources servies depuis un cache périmé : [{"nom": ..., "age_jours": ...}]."""
    return list(_SOURCES_PERIMEES)


def enregistrer_repli(nom: str, age_jours: float) -> None:
    _SOURCES_PERIMEES.append({"nom": nom, "age_jours": round(age_jours, 1)})


def chemin_cache(dossier: Path, nom: str) -> Path:
    """Chemin du fichier parquet pour une série donnée (ex. 'fred_CPIAUCSL')."""
    nom_propre = nom.replace("=", "_").replace("/", "_")
    return Path(dossier) / f"{nom_propre}.parquet"


def age_cache_jours(dossier: Path, nom: str) -> float | None:
    """Âge du fichier de cache en jours (None s'il n'existe pas)."""
    chemin = chemin_cache(dossier, nom)
    if not chemin.exists():
        return None
    return (time.time() - chemin.stat().st_mtime) / 86400


def lire_cache(dossier: Path, nom: str, duree_max_jours: float | None) -> pd.Series | None:
    """Série en cache si elle existe et a moins de `duree_max_jours` (None : quel que soit son âge)."""
    age_jours = age_cache_jours(dossier, nom)
    if age_jours is None:
        return None
    if duree_max_jours is not None and age_jours > duree_max_jours:
        return None
    return pd.read_parquet(chemin_cache(dossier, nom)).iloc[:, 0]


def ecrire_parquet_atomique(tableau: pd.DataFrame, chemin: Path) -> None:
    """Écrit dans un fichier temporaire du même dossier puis le renomme (opération atomique)."""
    temporaire = chemin.with_name(chemin.name + ".tmp")
    tableau.to_parquet(temporaire)
    os.replace(temporaire, chemin)


def ecrire_cache(dossier: Path, nom: str, serie: pd.Series) -> None:
    """Sauvegarde une série en parquet."""
    ecrire_parquet_atomique(serie.to_frame(name=serie.name or nom), chemin_cache(dossier, nom))


def obtenir_avec_cache(nom: str, telecharger: Callable[[], pd.Series], config: dict,
                       forcer: bool = False) -> pd.Series:
    """Série depuis le cache si elle est fraîche, sinon `telecharger()` (repli sur le cache périmé)."""
    dossier = chemin_projet(config["general"]["dossier_cache"])
    if not forcer:
        serie = lire_cache(dossier, nom, config["general"]["duree_cache_jours"])
        if serie is not None:
            return serie
    try:
        serie = telecharger()
    except requests.RequestException as erreur:
        age = age_cache_jours(dossier, nom)
        # Pas de cache, ou cache trop vieux : mieux vaut une erreur visible que des données obsolètes
        if age is None or age > config["general"]["anciennete_max_cache_jours"]:
            raise
        enregistrer_repli(nom, age)
        warnings.warn(f"{nom} : téléchargement impossible ({erreur}) ; cache de {age:.1f} jours réutilisé.")
        return lire_cache(dossier, nom, duree_max_jours=None)
    ecrire_cache(dossier, nom, serie)
    return serie
