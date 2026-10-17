"""Chargement des modèles Hugging Face, en local et sur CPU."""

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

APPAREIL = torch.device("cpu")  # volontairement CPU : doit tourner sur n'importe quel portable


@lru_cache(maxsize=4)
def charger_modele(nom: str):
    """(tokenizer, modèle) en mode évaluation ; mis en mémoire une seule fois par session."""
    tokenizer = AutoTokenizer.from_pretrained(nom)
    # Certains modèles sont stockés en float16 : on repasse en float32, plus précis et plus rapide sur CPU
    modele = AutoModelForSequenceClassification.from_pretrained(nom).float().to(APPAREIL).eval()
    return tokenizer, modele


def par_lots(elements: list, taille: int):
    for i in range(0, len(elements), taille):
        yield elements[i:i + taille]
