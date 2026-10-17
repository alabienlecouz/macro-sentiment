"""FinBERT (ProsusAI/finbert) : BERT affiné sur des phrases financières annotées
positive / negative / neutral (Financial PhraseBank).

FinBERT mesure le TON (bonne ou mauvaise nouvelle), pas l'orientation monétaire.
Pour le comparer, on utilise une correspondance naïve :
    positive -> hawkish (économie solide -> resserrement)
    negative -> dovish  (économie faible  -> assouplissement)
Elle échoue sur les phrases du type « l'inflation reste trop élevée » : ton négatif,
mais message hawkish. C'est précisément ce que la comparaison doit faire apparaître.
"""

import pandas as pd
import torch

from fxmacro.sentiment.modeles import APPAREIL, charger_modele, par_lots

CORRESPONDANCE = {"positive": "hawkish", "negative": "dovish", "neutral": "neutre"}


def scorer_finbert(phrases: list[str], nom_modele: str = "ProsusAI/finbert",
                   taille_lot: int = 16) -> pd.DataFrame:
    """Probabilités brutes (p_positive, ...) et converties (p_hawkish, p_dovish, p_neutre)."""
    tokenizer, modele = charger_modele(nom_modele)
    probas = []
    with torch.no_grad():
        for lot in par_lots(phrases, taille_lot):
            entrees = tokenizer(lot, padding=True, truncation=True, max_length=256,
                                return_tensors="pt").to(APPAREIL)
            probas.append(torch.softmax(modele(**entrees).logits, dim=1))
    p = torch.cat(probas).numpy()
    noms = [modele.config.id2label[i] for i in range(p.shape[1])]
    resultat = pd.DataFrame(p, columns=[f"p_{n}" for n in noms])
    for source, cible in CORRESPONDANCE.items():
        resultat[f"p_{cible}"] = resultat[f"p_{source}"]
    return resultat
