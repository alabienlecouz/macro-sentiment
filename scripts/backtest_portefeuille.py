"""Portefeuille G10 value + carry, rendement total, 1977/1986-2026 (protocole : config.yaml, portefeuille).

Chaque fin de mois : classement des 9 devises contre le dollar selon le carry (écart de taux), la
value (sous-évaluation PPA en temps réel) ou leur combinaison ; achat des 3 premières, vente des 3
dernières. Statistiques brutes et nettes de coûts, sous-périodes, crises, comportement les mois de
baisse des actions.

Usage : python scripts/backtest_portefeuille.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.signaux.apprentissage import taux_long
from fxmacro.signaux.portefeuille import (
    cours_fin_mois, ecart_ppa_devise, ecart_taux_portefeuille, poids_long_short, prime_de_risque, rendement_portefeuille,
    rendements_mensuels_actions, rendements_totaux, score_combine, statistiques,
)

NOMS = {"carry": "Carry", "value": "Value (PPA)", "combinaison": "Value + carry"}
COULEURS = {"carry": "#2a78d6", "value": "#eb6834", "combinaison": "#1baf7a"}

if __name__ == "__main__":
    config = charger_config()
    params = config["portefeuille"]
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    devises = list(params["devises"])
    debut_ppa = config["historique_long"]["debut_evaluation"]

    cours = pd.concat([cours_fin_mois(d, config) for d in devises], axis=1)
    taux = pd.concat([taux_long(d, config).rename(d) for d in devises], axis=1).reindex(cours.index)
    taux_usd = taux_long("USD", config).reindex(cours.index)
    rendements = rendements_totaux(cours, taux, taux_usd)

    signaux_bruts = {"carry": taux.sub(taux_usd, axis=0),
                     "value": pd.concat([ecart_ppa_devise(d, config, debut_ppa) for d in devises], axis=1)
                     .reindex(cours.index)}
    signaux_bruts["combinaison"] = score_combine(signaux_bruts["carry"], signaux_bruts["value"])
    actions = rendements_mensuels_actions(params["indice_actions"], config)

    periodes = {"1986-2026 (principal)": (params["debut_principal"], None),
                "1977-2026 (vérification)": (params["debut_verification"], None),
                **{k: tuple(v) for k, v in params["sous_periodes"].items()},
                **{k: tuple(v) for k, v in params["decennies"].items()}}
    lignes, crises, series_nettes = [], [], {}
    for nom in params["signaux"]:
        poids = poids_long_short(signaux_bruts[nom], params["nb_achat_vente"], params["min_devises"])
        r = rendement_portefeuille(poids, rendements, params["cout_pb_par_rotation"], params["cout_roll_pb_mois"])
        ecart_taux = ecart_taux_portefeuille(poids, signaux_bruts["carry"]).shift(1)
        series_nettes[nom] = r["net"]
        for nom_periode, (de, a) in periodes.items():
            bloc = r.loc[de:a]
            lignes.append({"strategie": nom, "periode": nom_periode, "rotation_moyenne": bloc["rotation"].mean(),
                           "ecart_taux_pts": ecart_taux.reindex(bloc.index).mean(),
                           **{f"{k}_brut": v for k, v in statistiques(bloc["brut"]).items()
                              if k in ("rendement_annuel_%", "sharpe")},
                           **statistiques(bloc["net"]), **prime_de_risque(bloc["net"], actions)})
        for nom_crise, (de, a) in params["crises"].items():
            crises.append({"strategie": nom, "crise": nom_crise, "rendement_cumule_%": 100 * r["net"].loc[de:a].sum()})

    resultats = pd.DataFrame(lignes)
    crises = pd.DataFrame(crises).pivot(index="crise", columns="strategie", values="rendement_cumule_%")
    nettes = pd.DataFrame(series_nettes).loc[params["debut_principal"]:]
    correlations = nettes.corr()
    resultats.to_csv(sorties / "backtest_portefeuille.csv", index=False)
    crises.to_csv(sorties / "backtest_portefeuille_crises.csv")

    # Figure : performance cumulée nette de coûts (somme des rendements logarithmiques)
    fig, axe = plt.subplots(figsize=(11, 4.8))
    for nom in params["signaux"]:
        cumul = 100 * nettes[nom].cumsum()
        axe.plot(cumul.index, cumul, color=COULEURS[nom], lw=1.6, label=NOMS[nom])
    for de, a in params["crises"].values():
        axe.axvspan(pd.Timestamp(de), pd.Timestamp(a), color="#8a8984", alpha=0.15, lw=0)
    axe.axhline(0, color="#8a8984", lw=0.6)
    axe.set_ylabel("performance cumulée nette, % (log)")
    axe.set_title("Portefeuilles G10 : 3 devises achetées, 3 vendues, rééquilibrage mensuel (zones grises : crises)",
                  loc="left", fontsize=10)
    axe.legend(frameon=False, loc="upper left")
    axe.grid(alpha=0.25)
    axe.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(sorties / "portefeuille_value_carry.png", dpi=130)

    pd.set_option("display.width", 220)
    colonnes = ["rendement_annuel_%_brut", "rendement_annuel_%", "volatilite_%", "sharpe", "t_newey_west",
                "pire_mois_%", "baisse_max_%", "asymetrie", "rotation_moyenne", "ecart_taux_pts"]
    print("=== Statistiques (nettes de coûts sauf mention « brut ») ===")
    print(resultats.set_index(["strategie", "periode"])[colonnes].round(2).to_string())
    print("\n=== Comportement selon les actions (S&P 500) ===")
    print(resultats.set_index(["strategie", "periode"])[["rendement_mois_baisse_actions_%",
          "rendement_mois_hausse_actions_%", "beta_mois_baisse", "correlation_actions"]].round(2).to_string())
    print("\n=== Rendement cumulé pendant les crises, % ===")
    print(crises.round(1).to_string())
    print("\n=== Corrélation des stratégies (1986-2026) ===")
    print(correlations.round(2).to_string())
    print(f"\nFigure : {sorties / 'portefeuille_value_carry.png'}")
