"""Du score par phrase au score par banque centrale."""

import pandas as pd

ETIQUETTES = ["hawkish", "dovish", "neutre"]


def etiquette_predite(probas: pd.DataFrame) -> pd.Series:
    """Étiquette de probabilité maximale."""
    colonnes = [f"p_{e}" for e in ETIQUETTES]
    return probas[colonnes].idxmax(axis=1).str.removeprefix("p_")


def score_phrase(probas: pd.DataFrame) -> pd.Series:
    """s = P(hawkish) - P(dovish), entre -1 (très dovish) et +1 (très hawkish)."""
    return probas["p_hawkish"] - probas["p_dovish"]


def score_document(probas: pd.DataFrame) -> dict:
    """Score moyen des phrases, pondéré par leur caractère non neutre (1 - P(neutre)).

    Les phrases purement descriptives pèsent peu ; celles qui portent un signal pèsent plus.
    """
    s = score_phrase(probas)
    poids = 1 - probas["p_neutre"]
    etiquettes = etiquette_predite(probas)
    return {
        "score": float((s * poids).sum() / poids.sum()) if poids.sum() > 0 else 0.0,
        "nb_phrases": len(probas),
        "part_hawkish": float((etiquettes == "hawkish").mean()),
        "part_dovish": float((etiquettes == "dovish").mean()),
    }
