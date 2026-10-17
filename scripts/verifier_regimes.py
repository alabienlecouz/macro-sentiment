"""Vérification « à l'œil » des régimes macro sur des épisodes connus (ce n'est pas un test).

Affiche le régime confirmé de chaque devise, mois par mois, sur quatre épisodes, puis
la part du temps passée dans chaque régime et la durée moyenne d'un régime.
Les seuils de config.yaml ne doivent PAS être ajustés en fonction de ce résultat.

Usage : python scripts/verifier_regimes.py   (après scripts/mettre_a_jour.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.regime.classification import REGIMES

EPISODES = {
    "Crise financière": ("2008-10", "2009-06"),
    "Covid": ("2020-03", "2020-06"),
    "Poussée d'inflation": ("2021-06", "2022-06"),
    "Désinflation": ("2023-06", "2024-06"),
}
ABREGES = {"goldilocks": "G", "reflation": "R", "stagflation": "S", "desinflation": "D"}


def durees_des_regimes(regime: pd.Series) -> pd.Series:
    """Longueur (en mois) de chaque séquence ininterrompue d'un même régime confirmé."""
    blocs = (regime != regime.shift()).cumsum()
    sequences = regime.groupby(blocs).agg(["first", "size"])
    return sequences.loc[sequences["first"].notna(), "size"]


if __name__ == "__main__":
    config = charger_config()
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    regimes = pd.DataFrame({
        devise: pd.read_csv(sorties / f"regimes_{devise}.csv", index_col="date", parse_dates=True)["regime"]
        for devise in config["regime"]["confiance"]
    })

    print("Régime confirmé (G = goldilocks, R = reflation, S = stagflation, D = désinflation, . = indéterminé)")
    for nom, (debut, fin) in EPISODES.items():
        extrait = regimes.loc[debut:fin].apply(lambda col: col.map(ABREGES)).fillna(".")
        extrait.index = extrait.index.strftime("%Y-%m")
        print(f"\n=== {nom} ({debut} à {fin}) ===")
        print(extrait.T.to_string())

    print("\n=== Part du temps dans chaque régime (mois avec un régime confirmé) ===")
    parts = regimes.apply(lambda col: col.value_counts(normalize=True)).reindex(REGIMES).fillna(0)
    print((100 * parts).round(0).astype(int).astype(str).add(" %").to_string())

    print("\n=== Durée moyenne d'un régime (mois) ===")
    print(regimes.apply(lambda col: durees_des_regimes(col).mean()).round(1).to_string())
