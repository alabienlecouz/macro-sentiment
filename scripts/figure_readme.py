"""Figure du README (en anglais) : cours vs fair values et z-scores ancrés pour une paire.

Usage : python scripts/figure_readme.py [PAIRE]   (USDJPY par défaut ; lancer mettre_a_jour.py avant)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fxmacro.config import RACINE_PROJET, charger_config

COULEURS = {"spot": "#2a78d6", "fv_ppa": "#eb6834", "fv_beer": "#1baf7a"}

if __name__ == "__main__":
    paire = sys.argv[1] if len(sys.argv) > 1 else "USDJPY"
    config = charger_config()
    tableau = pd.read_csv(RACINE_PROJET / config["general"]["dossier_sorties"] / f"fair_value_{paire}.csv",
                          index_col=0, parse_dates=True).loc["2004":]
    seuil = config["signal"]["seuil_z"]

    fig, (haut, bas) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for colonne, nom in [("spot", "Spot (monthly average)"), ("fv_ppa", "PPP fair value"), ("fv_beer", "BEER fair value")]:
        haut.plot(tableau.index, tableau[colonne], color=COULEURS[colonne], lw=2 if colonne == "spot" else 1.6, label=nom)
    haut.set_title(f"{paire}: spot vs macro fair values", loc="left", fontsize=12)
    haut.legend(frameon=False, loc="upper left")
    for colonne, nom in [("z_ppa_ancre", "PPP z-score"), ("z_beer_ancre", "BEER z-score")]:
        bas.plot(tableau.index, tableau[colonne], color=COULEURS[colonne.replace("z_", "fv_").replace("_ancre", "")],
                 lw=1.6, label=nom)
    for y in (-seuil, seuil):
        bas.axhline(y, color="#8a8984", ls="--", lw=0.8)
    bas.axhline(0, color="#8a8984", lw=0.6)
    bas.set_ylabel("anchored z")
    bas.legend(frameon=False, loc="upper left", ncol=2)
    for axe in (haut, bas):
        axe.grid(alpha=0.25)
        axe.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    sortie = RACINE_PROJET / "docs" / "images" / f"fair_values_{paire.lower()}.png"
    fig.savefig(sortie, dpi=150)
    print(sortie)
