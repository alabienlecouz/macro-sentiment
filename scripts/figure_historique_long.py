"""Contrôle visuel des fair values sur 1971-2026 : le cours oscille-t-il autour d'elles ?

Pour chaque paire : à gauche, le cours et les fair values (PPA longue, PPA ajustée de la
productivité, estimées sur tout l'échantillon) en échelle log ; à droite, l'écart en %.
Statistiques : nombre de fois où le cours recroise la fair value, part du temps à moins de
10 % et de 20 %, écart maximal.

Usage : python scripts/figure_historique_long.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.donnees.historique_long import paires_long, tableau_long
from fxmacro.fair_value.ppa import fair_value_ppa
from fxmacro.fair_value.ppa_productivite import estimer_ppa_productivite

COULEURS = {"spot": "#2a78d6", "ppa": "#eb6834", "ppa_productivite": "#1baf7a"}
NOMS = {"ppa": "PPA longue", "ppa_productivite": "PPA ajustée de la productivité"}


def statistiques_ecart(spot: pd.Series, fair_value: pd.Series) -> dict:
    """Croisements du cours avec la fair value et temps passé près d'elle."""
    ecart = (spot / fair_value - 1).dropna()
    signe = np.sign(ecart)
    croisements = int((signe != signe.shift()).iloc[1:].sum())
    annees = len(ecart) / 12
    return {"croisements": croisements,
            "annees_entre_croisements": annees / croisements if croisements else np.inf,
            "part_10pct": float((ecart.abs() < 0.10).mean()), "part_20pct": float((ecart.abs() < 0.20).mean()),
            "ecart_max_pct": float(100 * ecart.abs().max())}


if __name__ == "__main__":
    config = charger_config()
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    paires = paires_long(config)
    fig, axes = plt.subplots(len(paires), 2, figsize=(14, 3.1 * len(paires)),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    lignes = []
    for (axe_cours, axe_ecart), paire in zip(axes, paires):
        tableau = tableau_long(paire, config)
        fair_values = {"ppa": fair_value_ppa(tableau["spot"], tableau["cpi_base"], tableau["cpi_cotation"]),
                       "ppa_productivite": estimer_ppa_productivite(tableau)[0]}
        axe_cours.plot(tableau.index, tableau["spot"], color=COULEURS["spot"], lw=1.6, label="Cours (moyenne mensuelle)")
        for modele, fv in fair_values.items():
            axe_cours.plot(fv.index, fv, color=COULEURS[modele], lw=1.4, label=NOMS[modele])
            axe_ecart.plot(fv.index, 100 * (tableau["spot"] / fv - 1), color=COULEURS[modele], lw=1.1)
            lignes.append({"paire": paire, "modele": modele, **statistiques_ecart(tableau["spot"], fv)})
        axe_cours.set_yscale("log")
        axe_cours.set_title(paire, loc="left", fontsize=11)
        for y in (-20, 20):
            axe_ecart.axhline(y, color="#8a8984", ls="--", lw=0.8)
        axe_ecart.axhline(0, color="#8a8984", lw=0.6)
        axe_ecart.set_ylabel("écart au cours, %")
        for axe in (axe_cours, axe_ecart):
            axe.grid(alpha=0.25)
            axe.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=8)
    axes[0][1].set_title("Écart du cours à la fair value (pointillés : ±20 %)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(sorties / "fair_values_historique_long.png", dpi=130)

    stats = pd.DataFrame(lignes)
    stats.to_csv(sorties / "fair_values_historique_long.csv", index=False)
    pd.set_option("display.width", 200)
    affichage = stats.assign(part_10pct=(100 * stats.part_10pct).round(0), part_20pct=(100 * stats.part_20pct).round(0))
    print(affichage.round(1).to_string(index=False))
    print(f"\nFigure : {sorties / 'fair_values_historique_long.png'}")
