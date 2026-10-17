"""Graphiques de sortie."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on écrit directement un fichier
import matplotlib.pyplot as plt
import pandas as pd


def tracer_spot_vs_fair_value(tableau: pd.DataFrame, paire: str, colonnes_fv: list[str],
                              panneaux_z: dict[str, list[str]], chemin: Path) -> Path:
    """Panneau du haut : spot vs fair values. Un panneau de z-scores par modèle en dessous."""
    n = 1 + len(panneaux_z)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3 + 2.2 * n), sharex=True,
                             gridspec_kw={"height_ratios": [2] + [1] * len(panneaux_z)})
    haut = axes[0]
    haut.plot(tableau.index, tableau["spot"], label="Spot (moyenne mensuelle)", color="black", lw=1.2)
    for colonne in colonnes_fv:
        haut.plot(tableau.index, tableau[colonne], label=colonne, lw=1.3)
    haut.set_title(f"{paire} : spot vs fair values")
    haut.legend()
    haut.grid(alpha=0.3)

    for axe, (titre, colonnes) in zip(axes[1:], panneaux_z.items()):
        for colonne in colonnes:
            axe.plot(tableau.index, tableau[colonne], lw=1.1, label=colonne)
        for seuil in (-2, 2):
            axe.axhline(seuil, color="grey", ls="--", lw=0.8)
        axe.axhline(0, color="grey", lw=0.6)
        axe.set_ylabel(titre)
        axe.legend(loc="upper left", fontsize=8)
        axe.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)
    return chemin
