"""Banc d'essai des fair values : chaque modèle est-il une valeur vers laquelle le cours revient ?

Modèles comparés (écarts en temps réel, fenêtre croissante) : PPA, BEER et la référence naïve
(moyenne mobile du cours sur 5 ans). Critères fixés à l'avance dans config.yaml (banc_essai).

Usage : python scripts/banc_essai_fair_value.py           (depuis 2009, PPA + BEER + référence)
        python scripts/banc_essai_fair_value.py --long    (depuis 1976 sur l'historique 1971-2026 :
                                                           PPA + référence ; le BEER n'a pas de données)
        ... --de 1976-01-01 --a 2000-12-01                (sous-période : seules les dates évaluées sont
                                                           restreintes, les fair values restent en temps réel)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.fair_value.banc_essai import (
    ecarts_temps_reel, regression_panel, regression_predictive, rendement_futur, stationnarite,
)
from fxmacro.donnees.historique_long import paires_long, tableau_long
from fxmacro.pipeline import construire_tableau

def option(nom: str) -> str | None:
    """Valeur d'une option « --nom valeur » de la ligne de commande."""
    return sys.argv[sys.argv.index(nom) + 1] if nom in sys.argv else None


if __name__ == "__main__":
    config = charger_config()
    params = config["banc_essai"]
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    historique_long = "--long" in sys.argv
    MODELES = (["ppa", "ppa_productivite", "moyenne_mobile"] if historique_long
               else ["ppa", "beer", "moyenne_mobile"])
    debut = config["historique_long"]["debut_evaluation"] if historique_long else params["debut"]

    ecarts, spots = {}, {}
    for paire in (paires_long(config) if historique_long else config["paires"]):
        tableau = tableau_long(paire, config) if historique_long else construire_tableau(paire, config)[0]
        ecarts[paire] = ecarts_temps_reel(tableau, debut, params["fenetre_reference_mois"], tuple(MODELES))
        spots[paire] = tableau["spot"]
    de, a = option("--de"), option("--a")
    if de or a:
        ecarts = {p: e.loc[de:a] for p, e in ecarts.items()}
        debut = f"{de or debut} à {a or 'aujourd’hui'}"

    lignes = []
    for modele in MODELES:
        for h in params["horizons_mois"]:
            rendements = {p: rendement_futur(spots[p], h) for p in spots}
            ecarts_modele = {p: ecarts[p][modele] for p in ecarts}
            lignes.append({"modele": modele, "horizon": h, "paire": "PANEL",
                           **regression_panel(ecarts_modele, rendements, h)})
            # Version 2, décidée après les résultats de la version 1 (voir banc_essai.regression_panel)
            lignes.append({"modele": modele, "horizon": h, "paire": "PANEL_V2",
                           **regression_panel(ecarts_modele, rendements, h, effets_fixes=True, standardiser=True)})
            for p in spots:
                lignes.append({"modele": modele, "horizon": h, "paire": p,
                               **regression_predictive(ecarts_modele[p], rendements[p], h),
                               **stationnarite(ecarts_modele[p])})
    resultats = pd.DataFrame(lignes)
    suffixe = ("_long" if historique_long else "") + (f"_{de or ''}_{a or ''}" if de or a else "")
    fichier = sorties / f"banc_essai_fair_value{suffixe}.csv"
    resultats.to_csv(fichier, index=False)

    pd.set_option("display.width", 200)
    h0 = params["horizon_principal_mois"]
    print(f"Banc d'essai depuis {debut} — r(t→t+h) = α + β·e(t)  (β < 0 attendu)\n")
    for version, titre in [("PANEL", "Panel v1 : MCO empilés (critère fixé à l'avance)"),
                           ("PANEL_V2", "Panel v2 : effets fixes + écarts standardisés (décidé après la v1)")]:
        print(f"=== {titre} — t de Driscoll-Kraay ===")
        panel = resultats[resultats.paire == version].pivot(index="modele", columns="horizon", values=["beta", "t"])
        print(panel.round(2).reindex(MODELES).to_string() + "\n")

    print(f"\n=== Par paire, horizon {h0} mois (t de Newey-West) ===")
    detail = resultats[~resultats.paire.str.startswith("PANEL") & (resultats.horizon == h0)]
    print(detail.pivot(index="paire", columns="modele", values=["beta", "t"]).round(2).to_string())

    print("\n=== Retour vers la fair value (écart en temps réel) ===")
    print(detail.pivot(index="paire", columns="modele", values=["p_adf", "demi_vie_mois"]).round(2).to_string())

    print(f"\n=== Critères fixés à l'avance (horizon {h0} mois) ===")
    for modele in MODELES:
        ligne = resultats[(resultats.modele == modele) & (resultats.horizon == h0)]
        p1, p2 = ligne[ligne.paire == "PANEL"].iloc[0], ligne[ligne.paire == "PANEL_V2"].iloc[0]
        paires = ligne[~ligne.paire.str.startswith("PANEL")]
        print(f"{modele:15} v1 β {p1['beta']:+.2f} (t {p1['t']:+.2f}) | v2 β {p2['beta']:+.3f} par écart-type"
              f" (t {p2['t']:+.2f}) | β < 0 sur {int((paires['beta'] < 0).sum())}/{len(spots)} paires"
              f" | demi-vie médiane {paires['demi_vie_mois'].median():.0f} mois | ~{p1['n_dates_indep']} dates indépendantes")
    print(f"\nRésultats : {fichier}")
