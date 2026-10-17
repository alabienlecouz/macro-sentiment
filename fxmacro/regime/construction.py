"""Régimes macro d'une devise et état courant de toutes les devises."""

import numpy as np
import pandas as pd

from fxmacro.donnees.confiance import obtenir_confiance
from fxmacro.donnees.cpi import obtenir_cpi
from fxmacro.donnees.taux import obtenir_taux
from fxmacro.regime.classification import classifier
from fxmacro.regime.indicateurs import construire_indicateurs


def regimes_devise(devise: str, config: dict) -> pd.DataFrame:
    """Historique mensuel : indicateurs, régime brut, régime confirmé, mois dans le régime."""
    params = config["regime"]
    indicateurs = construire_indicateurs(obtenir_confiance(devise, config), obtenir_cpi(devise, config),
                                         obtenir_taux(devise, config), params,
                                         config["alignement"]["anciennete_max_mois"])
    return classifier(indicateurs, params["mois_confirmation"],
                      params["zone_neutre_ecarts_types"], params["zone_neutre_min_obs"])


def ligne_etat_regime(devise: str, regimes: pd.DataFrame, ton_banque: float | None) -> dict:
    """Dernier mois avec un régime confirmé, et le ton actuel de la banque centrale.

    Le ton n'a pas d'historique : il n'apparaît que dans l'état courant. Le cycle
    monétaire est le dernier connu (la BRI publie les taux directeurs avec un mois de retard).
    """
    confirmes = regimes.dropna(subset=["regime"])
    if confirmes.empty:
        return {"devise": devise, "mois": None, "regime": np.nan, "regime_brut": np.nan, "reporte": np.nan,
                "mois_dans_regime": np.nan,
                "croissance_niveau": np.nan, "croissance_dynamique": np.nan, "inflation_niveau": np.nan,
                "ecart_cible": np.nan, "cycle_monetaire": np.nan, "ton_banque_centrale": ton_banque}
    derniere = confirmes.iloc[-1]
    return {
        "devise": devise, "mois": confirmes.index[-1].strftime("%Y-%m"),
        # regime_brut ≠ regime : un changement est en attente de confirmation
        "regime": derniere["regime"], "regime_brut": derniere["regime_brut"], "reporte": bool(derniere["reporte"]),
        "mois_dans_regime": int(derniere["mois_dans_regime"]),
        "croissance_niveau": derniere["croissance_niveau"], "croissance_dynamique": derniere["croissance_dynamique"],
        "inflation_niveau": derniere["inflation_niveau"], "ecart_cible": derniere["ecart_cible"],
        # Le taux directeur (BRI) peut avoir un mois de retard : dernier cycle connu
        "cycle_monetaire": regimes["cycle_monetaire"].dropna().iloc[-1] if regimes["cycle_monetaire"].notna().any() else np.nan,
        "ton_banque_centrale": np.nan if ton_banque is None else ton_banque,
    }
