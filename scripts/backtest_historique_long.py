"""Backtest du signal de valorisation sur l'historique long (1976-2026), en temps réel.

Versions comparées, toutes avec des z-scores calculés sur les seules données passées :
  a) PPA longue ;
  b) PPA ajustée de la productivité ;
  c) combinaison : moyenne des z de a) et b) ;
  d) référence naïve : écart à la moyenne mobile du cours sur 5 ans, divisé par son
     écart-type passé (même règle de signal, pour vérifier que la PPA fait mieux qu'un
     simple retour à la moyenne du cours).
Signal : hausse si z ≤ -seuil, baisse si z ≥ +seuil ; rendement du cours sur 12 mois
(ni portage ni coûts de transaction : ce n'est pas une mesure de rentabilité).

Usage : python scripts/backtest_historique_long.py [--horizon 24]   (12 mois par défaut)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.donnees.historique_long import paires_long, tableau_long
from fxmacro.fair_value.banc_essai import ecart_moyenne_mobile, standardiser_temps_reel
from fxmacro.fair_value.ppa_productivite import ppa_productivite_temps_reel
from fxmacro.signaux.backtest import evaluer, ppa_temps_reel, signal_depuis_z

PERIODES = {"1976-2026": ("1976-01-01", None), "1976-2000": ("1976-01-01", "2000-12-01"),
            "2001-2026": ("2001-01-01", None)}
VERSIONS = ["a) PPA longue", "b) PPA productivité", "c) moyenne a + b", "d) moyenne mobile 5 ans"]


def z_par_version(tableau: pd.DataFrame, debut: str, fenetre_mois: int) -> pd.DataFrame:
    """z en temps réel de chaque version."""
    z_ppa = ppa_temps_reel(tableau, debut)["z"]
    z_prod = ppa_productivite_temps_reel(tableau, debut)["z"]
    z_mm = standardiser_temps_reel(ecart_moyenne_mobile(tableau["spot"], fenetre_mois))
    return pd.DataFrame({VERSIONS[0]: z_ppa, VERSIONS[1]: z_prod,
                         VERSIONS[2]: pd.concat([z_ppa, z_prod], axis=1).mean(axis=1), VERSIONS[3]: z_mm})


if __name__ == "__main__":
    config = charger_config()
    debut = config["historique_long"]["debut_evaluation"]
    seuil = config["signal"]["seuil_z"]
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    horizon = int(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 12

    lignes = []
    for paire in paires_long(config):
        tableau = tableau_long(paire, config)
        z = z_par_version(tableau, debut, config["banc_essai"]["fenetre_reference_mois"])
        for nom_periode, (de, a) in PERIODES.items():
            z_periode = z.loc[de:a]
            for version in VERSIONS:
                stats = evaluer(signal_depuis_z(z_periode[version], seuil), tableau["spot"], horizon=horizon)
                lignes.append({"paire": paire, "version": version, "periode": nom_periode, **stats})

    resultats = pd.DataFrame(lignes)
    fichier = sorties / ("backtest_historique_long.csv" if horizon == 12 else f"backtest_historique_long_{horizon}m.csv")
    resultats.to_csv(fichier, index=False)
    pd.set_option("display.width", 220)
    colonnes = ["part_actifs", "taux_reussite", "rendement_moyen_%", "n_glouton", "taux_reussite_glouton"]
    for nom_periode in PERIODES:
        print(f"=== {nom_periode} — moyenne des 5 paires (seuil |z| = {seuil}, horizon {horizon} mois) ===")
        bloc = resultats[resultats.periode == nom_periode]
        print(bloc.groupby("version")[colonnes].mean().round(3).to_string() + "\n")
    print("=== 1976-2026 — par paire (mois actifs / réussite) ===")
    complet = resultats[resultats.periode == "1976-2026"]
    print(complet.pivot(index="paire", columns="version", values=["part_actifs", "taux_reussite"]).round(2).to_string())
    print(f"\nRésultats : {fichier}")
