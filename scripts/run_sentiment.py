"""Démonstration : comparaison DeBERTa zero-shot vs FinBERT, puis score des banques centrales.

Usage : python scripts/run_sentiment.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fxmacro.config import RACINE_PROJET, chemin_projet, charger_config
from fxmacro.sentiment.analyse import evaluer_modeles, scorer_communiques
from fxmacro.sentiment.textes import mettre_a_jour_communiques

if __name__ == "__main__":
    config = charger_config()
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 90)

    erreurs = mettre_a_jour_communiques(RACINE_PROJET / config["sentiment"]["dossier_communiques"])
    for erreur in erreurs:
        print("Téléchargement impossible (copier le texte à la main) :", erreur)

    debut = time.time()
    tableau_metriques, confusions, detail = evaluer_modeles(config)
    print(f"\n=== Phrases annotées ({len(detail)}) ===")
    print(tableau_metriques.round(2))
    for modele, matrice in confusions.items():
        print(f"\nMatrice de confusion - {modele}\n{matrice}")
    desaccords = detail[(detail["zero_shot"] == detail["etiquette"]) & (detail["finbert"] != detail["etiquette"])]
    print("\nExemples où FinBERT se trompe et pas DeBERTa :")
    print(desaccords.head(6).to_string(index=False))

    resume, phrases = scorer_communiques(config)
    print("\n=== Derniers communiqués (score de -1 dovish à +1 hawkish) ===")
    print(resume.round(2).to_string())
    print(f"\nTemps total sur CPU : {time.time() - debut:.0f} s")

    tableau_metriques.to_csv(sorties / "sentiment_evaluation.csv")
    detail.to_csv(sorties / "sentiment_phrases_annotees.csv", index=False)
    resume.to_csv(sorties / "sentiment_banques.csv")
    phrases.to_csv(sorties / "sentiment_phrases_communiques.csv", index=False)
