"""Démonstration : données + fair values (PPA, parité des taux) pour une ou toutes les paires.

Usage : python scripts/run_fair_value.py            (toutes les paires)
        python scripts/run_fair_value.py USDJPY     (une seule paire)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.pipeline import construire_tableau, resumer_derniere_ligne
from fxmacro.rapport.graphiques import tracer_spot_vs_fair_value

if __name__ == "__main__":
    config = charger_config()
    paires = sys.argv[1:] or list(config["paires"])
    sorties = chemin_projet(config["general"]["dossier_sorties"])

    lignes, estimations = [], []
    for paire in paires:
        tableau, beer = construire_tableau(paire, config)
        estimations.append({"paire": paire, **{f"b_{k}": v for k, v in beer.coefficients.items()},
                            **{f"t_{k}": v for k, v in beer.t_stats_hac.items() if k != "const"},
                            "R2": beer.r2, "p_coint": beer.p_value_cointegration,
                            "demi_vie_mois": beer.demi_vie_mois, "n": beer.nb_obs})
        tableau.to_csv(sorties / f"fair_value_{paire}.csv")
        tracer_spot_vs_fair_value(
            tableau, paire, ["fv_ppa", "fv_uip", "fv_beer"],
            {"z PPA": ["z_ppa_glissant", "z_ppa_ancre"], "z UIP": ["z_uip_glissant", "z_uip_ancre"],
             "z BEER": ["z_beer_glissant", "z_beer_ancre"]},
            sorties / f"fair_value_{paire}.png")
        lignes.append(resumer_derniere_ligne(paire, tableau, config))

    synthese = pd.DataFrame(lignes).set_index("paire")
    synthese.to_csv(sorties / "synthese_fair_value.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(synthese.round(2).to_string())

    estimations = pd.DataFrame(estimations).set_index("paire")
    estimations.to_csv(sorties / "estimation_beer.csv")
    print("\nEstimation BEER (MCO, t de Newey-West, p-value d'Engle-Granger) :")
    print(estimations.round(3).to_string())
    print(f"\nCSV et graphiques dans {sorties}")
