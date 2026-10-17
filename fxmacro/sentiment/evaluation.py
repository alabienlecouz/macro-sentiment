"""Évaluation d'un classifieur sur des phrases annotées à la main."""

import pandas as pd

from fxmacro.sentiment.agregation import ETIQUETTES


def matrice_confusion(vraies: pd.Series, predites: pd.Series) -> pd.DataFrame:
    """Lignes = vraie étiquette, colonnes = étiquette prédite."""
    return pd.crosstab(pd.Categorical(vraies, ETIQUETTES), pd.Categorical(predites, ETIQUETTES),
                       rownames=["vraie"], colnames=["predite"], dropna=False)


def metriques(vraies: pd.Series, predites: pd.Series) -> dict:
    """Exactitude et F1 macro (moyenne simple des F1 par étiquette).

    Précision(k) = bonnes prédictions de k / toutes les prédictions de k
    Rappel(k)    = bonnes prédictions de k / toutes les vraies k
    F1(k)        = moyenne harmonique des deux
    """
    vraies, predites = pd.Series(vraies).reset_index(drop=True), pd.Series(predites).reset_index(drop=True)
    resultat = {"exactitude": float((vraies == predites).mean())}
    f1s = []
    for k in ETIQUETTES:
        vp = ((predites == k) & (vraies == k)).sum()
        precision = vp / (predites == k).sum() if (predites == k).sum() else 0.0
        rappel = vp / (vraies == k).sum() if (vraies == k).sum() else 0.0
        f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
        resultat[f"f1_{k}"] = float(f1)
        f1s.append(f1)
    resultat["f1_macro"] = float(sum(f1s) / len(f1s))
    return resultat
