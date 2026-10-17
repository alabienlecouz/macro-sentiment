"""Chargement de la configuration (config.yaml) et des secrets (.env)."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

RACINE_PROJET = Path(__file__).resolve().parent.parent


def fusionner(base: dict, surcharge: dict) -> dict:
    """Fusion récursive : les valeurs de `surcharge` remplacent celles de `base`."""
    resultat = dict(base)
    for cle, valeur in surcharge.items():
        if isinstance(valeur, dict) and isinstance(resultat.get(cle), dict):
            resultat[cle] = fusionner(resultat[cle], valeur)
        else:
            resultat[cle] = valeur
    return resultat


def charger_config(chemin: Path | str | None = None) -> dict:
    """Lit config.yaml, puis config.local.yaml s'il existe (réglages propres à une machine, ex. la VM)."""
    chemin = Path(chemin) if chemin else RACINE_PROJET / "config.yaml"
    with open(chemin, encoding="utf-8") as fichier:
        config = yaml.safe_load(fichier)
    chemin_local = chemin.with_name("config.local.yaml")
    if chemin_local.exists():
        with open(chemin_local, encoding="utf-8") as fichier:
            config = fusionner(config, yaml.safe_load(fichier) or {})
    return config


def cle_fred() -> str | None:
    """Renvoie la clé API FRED depuis .env, ou None si absente."""
    load_dotenv(RACINE_PROJET / ".env")
    cle = os.getenv("FRED_API_KEY")
    if not cle or cle == "ta_cle_ici":
        return None
    return cle


def chemin_projet(relatif: str) -> Path:
    """Convertit un chemin relatif de la config en chemin absolu, et crée le dossier."""
    chemin = RACINE_PROJET / relatif
    chemin.mkdir(parents=True, exist_ok=True)
    return chemin
