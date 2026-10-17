"""Combinaison des modèles : valorisation (signal principal) + sentiment (confirmation).

Pour une paire XXXYYY :
- z_val = Σ w_m · z_m / Σ w_m, moyenne pondérée des z-scores ancrés des modèles de
  valorisation disponibles (PPA, BEER). z_val > 0 : XXX surévalué face à YYY.
- Δs = s_XXX - s_YYY, écart de ton entre les deux banques centrales (hawkish > 0).
  Une banque plus hawkish soutient sa devise.

Règle :
- |z_val| < seuil_z                     -> neutre
- XXX sous-évalué (z_val ≤ -seuil_z)    -> orientation hausse de XXXYYY
- XXX surévalué  (z_val ≥ +seuil_z)     -> orientation baisse de XXXYYY
- force : forte si le sentiment va dans le même sens (|Δs| ≥ seuil_sentiment),
          faible s'il va dans le sens opposé, modérée sinon.
Outil d'analyse uniquement : ce n'est ni un ordre ni un conseil d'investissement.
"""

import math

import numpy as np


def zscore_valorisation(z_par_modele: dict[str, float], poids: dict[str, float]) -> float:
    """Moyenne pondérée des z-scores disponibles (les NaN sont ignorés, poids renormalisés)."""
    paires = [(z_par_modele[m], w) for m, w in poids.items()
              if m in z_par_modele and z_par_modele[m] is not None and not math.isnan(z_par_modele[m])]
    if not paires:
        return float("nan")
    return sum(z * w for z, w in paires) / sum(w for _, w in paires)


def differentiel_sentiment(score_base: float | None, score_cotation: float | None) -> float:
    """s_base - s_cotation (NaN si l'un des deux manque)."""
    if score_base is None or score_cotation is None:
        return float("nan")
    return float(score_base) - float(score_cotation)


def zscore_ancre_direct(spot: float, fair_value: float, sigma: float) -> float:
    """z ancré recalculé avec un spot récent : ln(S / FV) / σ."""
    return float(np.log(spot / fair_value) / sigma)


def classer_signal(paire: str, base: str, cotation: str, z_val: float, delta_sentiment: float,
                   seuil_z: float, seuil_sentiment: float) -> dict:
    """Direction, force et phrase d'explication du signal d'une paire."""
    if math.isnan(z_val) or abs(z_val) < seuil_z:
        return {"orientation": "neutre", "force": "aucune",
                "explication": f"{paire} proche de sa valeur de référence (z = {z_val:+.2f})."}

    base_sous_evaluee = z_val < 0
    orientation = "hausse" if base_sous_evaluee else "baisse"
    sous, sur = (base, cotation) if base_sous_evaluee else (cotation, base)
    # Le sentiment confirme si la devise sous-évaluée a la banque centrale la plus hawkish
    confirmation = delta_sentiment if base_sous_evaluee else -delta_sentiment

    if math.isnan(confirmation):
        force, commentaire = "modérée", "sentiment indisponible"
    elif confirmation >= seuil_sentiment:
        force, commentaire = "forte", f"banque centrale {sous} plus hawkish : confirmation"
    elif confirmation <= -seuil_sentiment:
        force, commentaire = "faible", f"banque centrale {sur} plus hawkish : le sentiment contredit"
    else:
        force, commentaire = "modérée", "sentiment sans écart marqué"

    return {"orientation": orientation, "force": force,
            "explication": f"{sous} sous-évalué face à {sur} (z = {z_val:+.2f}) ; {commentaire}."}
