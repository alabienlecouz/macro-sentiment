"""Backtest du signal de valorisation : plein échantillon (biais d'anticipation) contre temps réel.

Quatre versions, évaluées sur les mêmes dates (≥ début) et le même horizon :
  a) plein échantillon        : moyenne de z_ppa_ancre et z_beer_ancre du pipeline ;
  b) temps réel PPA + BEER    : moyenne des deux z recalibrés à chaque date ;
  c) temps réel PPA seul ;
  d) temps réel, BEER seulement s'il est cointégré à la date t (p ≤ p_max_cointegration).
Ni coûts de transaction ni portage ne sont inclus : ce n'est pas une mesure de rentabilité.

Usage : python scripts/backtest_temps_reel.py [DEBUT]   (DEBUT = 2009-01-01 par défaut)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.pipeline import construire_tableau
from fxmacro.signaux.backtest import evaluer, signal_depuis_z, z_beer_temps_reel, z_ppa_temps_reel

VERSIONS = {
    "a": "plein échantillon",
    "b": "temps réel PPA+BEER",
    "c": "temps réel PPA seul",
    "d": "temps réel, BEER si cointégré",
}


def z_par_version(tableau: pd.DataFrame, debut: str, p_max: float) -> pd.DataFrame:
    """z de valorisation de chaque version, sur les dates ≥ début."""
    z_ppa = z_ppa_temps_reel(tableau, debut)
    beer = z_beer_temps_reel(tableau, debut)
    z_beer_cointegre = beer["z"].where(beer["p_coint"] <= p_max)
    z = pd.DataFrame({
        "a": tableau[["z_ppa_ancre", "z_beer_ancre"]].mean(axis=1),
        "b": pd.concat([z_ppa, beer["z"]], axis=1).mean(axis=1),
        "c": z_ppa,
        "d": pd.concat([z_ppa, z_beer_cointegre], axis=1).mean(axis=1),
    })
    return z.loc[z.index >= pd.Timestamp(debut)]


if __name__ == "__main__":
    debut = sys.argv[1] if len(sys.argv) > 1 else "2009-01-01"
    config = charger_config()
    seuil, p_max = config["signal"]["seuil_z"], config["signal"]["p_max_cointegration"]
    sorties = chemin_projet(config["general"]["dossier_sorties"])

    lignes, dernieres = [], []
    for paire in config["paires"]:
        tableau, _ = construire_tableau(paire, config)
        z = z_par_version(tableau, debut, p_max)
        dernieres.append({"paire": paire, "z_val_plein": z["a"].iloc[-1], "z_val_temps_reel": z["b"].iloc[-1]})
        for code, nom in VERSIONS.items():
            stats = evaluer(signal_depuis_z(z[code], seuil), tableau["spot"], horizon=12)
            lignes.append({"paire": paire, "version": f"{code}) {nom}", **stats})

    resultats = pd.DataFrame(lignes)
    resultats.to_csv(sorties / "backtest_temps_reel.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    colonnes = ["part_actifs", "taux_reussite", "rendement_moyen_%", "n",
                "n_non_chevauchant", "taux_reussite_non_chevauchant", "n_glouton", "taux_reussite_glouton"]
    print(f"Backtest depuis {debut}, seuil |z| = {seuil}, horizon 12 mois\n")
    print("=== Moyenne des paires ===")
    print(resultats.groupby("version")[colonnes].mean().round(3).to_string())
    print("\n=== Détail par paire ===")
    print(resultats.set_index(["paire", "version"])[colonnes].round(3).to_string())
    print("\n=== Dernière date : z_val plein échantillon vs temps réel (doivent être égaux) ===")
    print(pd.DataFrame(dernieres).set_index("paire").round(4).to_string())
    print(f"\nRésultats : {sorties / 'backtest_temps_reel.csv'}")
