"""État de référence de chaque paire, calculé une fois par jour par la tâche de mise à jour.

Il contient tout ce qu'il faut pour recalculer le signal avec un spot récent, sans
relancer les modèles : dernières fair values, écart-type σ de chaque modèle, p-value
de cointégration du BEER et scores de sentiment.

Le BEER n'entre dans le signal que s'il est cointégré (p ≤ p_max_cointegration) : sans
cointégration, sa régression peut être fallacieuse. Son z reste affiché à titre indicatif.
"""

import numpy as np
import pandas as pd

from fxmacro.fair_value.beer import ResultatBEER
from fxmacro.fair_value.ecarts import ecart_log
from fxmacro.pipeline import periode_reference
from fxmacro.signaux.combinaison import (
    classer_signal, differentiel_sentiment, zscore_ancre_direct, zscore_valorisation,
)


def sigma_modele(tableau: pd.DataFrame, colonne_fv: str, periode: tuple) -> float:
    """σ de ln(S / FV) sur la période de calibration (dénominateur du z ancré)."""
    debut, fin = periode
    return float(ecart_log(tableau["spot"], tableau[colonne_fv]).loc[debut:fin].std())


def ligne_etat(paire: str, tableau: pd.DataFrame, sentiment_par_devise: dict[str, float], config: dict,
               resultat_beer: ResultatBEER | None = None) -> dict:
    """Résumé de référence d'une paire (dernier mois disponible)."""
    infos = config["paires"][paire]
    derniere = tableau.iloc[-1]
    ligne = {
        "paire": paire, "base": infos["base"], "cotation": infos["cotation"], "ticker": infos["ticker"],
        "mois_reference": tableau.index[-1].strftime("%Y-%m"), "spot_mensuel": derniere["spot"],
        "carry_base": derniere["carry_base"], "z_uip_ancre": derniere["z_uip_ancre"],
        "sentiment_base": sentiment_par_devise.get(infos["base"], np.nan),
        "sentiment_cotation": sentiment_par_devise.get(infos["cotation"], np.nan),
        "p_coint_beer": resultat_beer.p_value_cointegration if resultat_beer is not None else np.nan,
    }
    for modele in config["signal"]["poids_modeles"]:
        ligne[f"fv_{modele}"] = derniere[f"fv_{modele}"]
        ligne[f"sigma_{modele}"] = sigma_modele(tableau, f"fv_{modele}", periode_reference(modele, config))
        ligne[f"z_{modele}_ancre"] = derniere[f"z_{modele}_ancre"]
    return ligne


def beer_exclu(ligne: dict | pd.Series, config: dict) -> bool:
    """Vrai si le BEER doit être écarté du signal : non cointégré, ou p-value inconnue."""
    p = ligne.get("p_coint_beer", np.nan)
    return bool(pd.isna(p) or p > config["signal"]["p_max_cointegration"])


def evaluer_signal(ligne: dict | pd.Series, spot: float, config: dict) -> dict:
    """Recalcule z par modèle, z de valorisation et signal pour un spot donné."""
    params = config["signal"]
    z = {m: zscore_ancre_direct(spot, ligne[f"fv_{m}"], ligne[f"sigma_{m}"]) for m in params["poids_modeles"]}
    exclu = "beer" in z and beer_exclu(ligne, config)
    # Le z du BEER est mis à NaN pour le signal seulement : zscore_valorisation renormalise les poids
    z_signal = {m: (np.nan if m == "beer" and exclu else v) for m, v in z.items()}
    z_val = zscore_valorisation(z_signal, params["poids_modeles"])
    delta = differentiel_sentiment(ligne["sentiment_base"], ligne["sentiment_cotation"])
    signal = classer_signal(ligne["paire"], ligne["base"], ligne["cotation"], z_val, delta,
                            params["seuil_z"], params["seuil_sentiment"])
    ecarts = {f"ecart_{m}_%": 100 * (spot / ligne[f"fv_{m}"] - 1) for m in params["poids_modeles"]}
    return {"spot": spot, **ecarts, **{f"z_{m}": v for m, v in z.items()},
            "z_valorisation": z_val, "beer_exclu": exclu, "delta_sentiment": delta, **signal}
