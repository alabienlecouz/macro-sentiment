"""Classification zero-shot par inférence en langage naturel (NLI), avec DeBERTa-v3.

Un modèle NLI répond à la question : la « prémisse » implique-t-elle l'« hypothèse » ?
Pour classer une phrase x sans entraînement spécifique, on la compare à une hypothèse
par étiquette, par ex. h_hawkish = « This statement suggests that interest rates will rise ».

Pour chaque étiquette k, le modèle donne un logit d'implication z_k = f(x, h_k).
Les probabilités sont un softmax sur les étiquettes :

    P(k | x) = exp(z_k) / Σ_j exp(z_j)

Le modèle n'a jamais vu le mot « hawkish » pendant un entraînement supervisé :
toute la connaissance vient de sa compréhension générale de l'implication logique.
"""

import pandas as pd
import torch

from fxmacro.sentiment.modeles import APPAREIL, charger_modele, par_lots


def scorer_zero_shot(phrases: list[str], hypotheses: dict[str, str], nom_modele: str,
                     taille_lot: int = 16) -> pd.DataFrame:
    """Probabilités p_<étiquette> pour chaque phrase (une colonne par hypothèse)."""
    tokenizer, modele = charger_modele(nom_modele)
    indice_implication = modele.config.label2id["entailment"]
    etiquettes = list(hypotheses)
    paires = [(phrase, hypotheses[e]) for phrase in phrases for e in etiquettes]

    logits_implication = []
    with torch.no_grad():
        for lot in par_lots(paires, taille_lot):
            entrees = tokenizer([p for p, _ in lot], [h for _, h in lot], padding=True,
                                truncation=True, max_length=256, return_tensors="pt").to(APPAREIL)
            logits_implication.append(modele(**entrees).logits[:, indice_implication])

    z = torch.cat(logits_implication).reshape(len(phrases), len(etiquettes))
    probas = torch.softmax(z, dim=1).numpy()
    return pd.DataFrame(probas, columns=[f"p_{e}" for e in etiquettes])
