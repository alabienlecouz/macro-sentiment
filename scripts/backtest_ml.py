"""Random Forest et XGBoost contre régression logistique, walk-forward purgé, G10, 1991-2025.

Protocole fixé dans config.yaml (section apprentissage) avant tout résultat. Pour chaque horizon :
réussite hors échantillon (toutes observations et échantillon conservateur), rendement moyen,
log-loss, test de Diebold-Mariano contre la logistique, écart entraînement / hors échantillon,
importance des variables par permutation, sous-périodes.

Usage : python scripts/backtest_ml.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.donnees.historique_long import paires_long
from fxmacro.signaux.apprentissage import (
    construire_panel, diebold_mariano, evaluer, importance_permutation, variables_paire, walk_forward,
)

if __name__ == "__main__":
    config = charger_config()
    params = config["apprentissage"]
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    variables = params["variables"]
    annees = range(params["premiere_annee_test"], params["derniere_annee_test"] + 1)

    par_paire = {paire: variables_paire(paire, config) for paire in paires_long(config)}
    lignes, comparaisons, importances = [], [], []
    for horizon in params["horizons_mois"]:
        panel = construire_panel(par_paire, variables, horizon)
        predictions, internes, ajustes = walk_forward(panel, variables, params["modeles"], horizon, annees,
                                                      params["graine"])
        predictions.to_csv(sorties / f"backtest_ml_predictions_{horizon}m.csv", index=False)
        reussite_entrainement = internes.groupby("modele")["reussite_entrainement"].mean()
        periodes = {"1991-2025": (None, None), **params["sous_periodes"]}

        for nom_periode, (de, a) in periodes.items():
            masque = pd.Series(True, index=predictions.index)
            if de:
                masque = (predictions["date"] >= de) & (predictions["date"] <= a)
            p = predictions[masque]
            # Référence naïve : règle du bot (la devise sous-évaluée remonte)
            naive = pd.Series(np.where(p["z_ppa"] < 0, 1, -1), index=p.index)
            lignes.append({"horizon": horizon, "periode": nom_periode, "modele": "règle du bot (−z PPA)",
                           **evaluer(p, naive, horizon)})
            for nom in params["modeles"]:
                direction = pd.Series(np.where(p[f"proba_{nom}"] > 0.5, 1, -1), index=p.index)
                lignes.append({"horizon": horizon, "periode": nom_periode, "modele": nom,
                               **evaluer(p, direction, horizon, p[f"proba_{nom}"]),
                               "reussite_entrainement": reussite_entrainement[nom] if nom_periode == "1991-2025" else np.nan})
            for nom in [m for m in params["modeles"] if m != "logistique"]:
                comparaisons.append({"horizon": horizon, "periode": nom_periode, "modele": nom,
                                     **diebold_mariano(p, nom, "logistique", horizon)})

        for nom in params["modeles"]:
            hausses = importance_permutation(predictions, ajustes, nom, variables, params["permutations"],
                                             params["graine"])
            importances.append(hausses.rename(f"{nom}_{horizon}m"))

    resultats = pd.DataFrame(lignes)
    comparaisons = pd.DataFrame(comparaisons)
    importances = pd.concat(importances, axis=1)
    resultats.to_csv(sorties / "backtest_ml.csv", index=False)
    comparaisons.to_csv(sorties / "backtest_ml_diebold_mariano.csv", index=False)
    importances.to_csv(sorties / "backtest_ml_importance.csv")

    pd.set_option("display.width", 220)
    colonnes = ["reussite", "reussite_entrainement", "n_conservateur", "reussite_conservateur", "p_conservateur",
                "rendement_moyen_%", "t_newey_west", "log_loss", "brier"]
    print("=== Résultats hors échantillon ===")
    print(resultats.set_index(["horizon", "periode", "modele"])[colonnes].round(3).to_string())
    print("\n=== Diebold-Mariano : gain de log-loss contre la logistique (positif = mieux) ===")
    print(comparaisons.set_index(["horizon", "periode", "modele"]).round(4).to_string())
    print("\n=== Importance par permutation (hausse de log-loss hors échantillon) ===")
    print(importances.round(4).to_string())
