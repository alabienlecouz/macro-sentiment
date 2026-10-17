"""Tâche de mise à jour (à lancer une fois par jour, ex. par un timer systemd sur la VM).

1. Fair values (PPA, UIP, BEER) de chaque paire -> sorties/fair_value_<PAIRE>.csv
2. Sentiment des derniers communiqués           -> sorties/sentiment_*.csv
3. État de référence pour le tableau de bord    -> sorties/etat_paires.csv + sorties/maj.json
4. Régimes macro par devise (descriptif)        -> sorties/regimes_<DEVISE>.csv + etat_regimes.csv

Usage : python scripts/mettre_a_jour.py [--sans-sentiment]
"""

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fxmacro.config import RACINE_PROJET, chemin_projet, charger_config
from fxmacro.donnees.cache import reinitialiser_suivi, sources_perimees
from fxmacro.pipeline import construire_tableau
from fxmacro.signaux.etat import evaluer_signal, ligne_etat


def journal(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def mettre_a_jour_sentiment(config: dict, sorties: Path) -> None:
    # Import local : torch et transformers ne sont chargés que si on en a besoin
    from fxmacro.sentiment.analyse import scorer_communiques
    from fxmacro.sentiment.textes import mettre_a_jour_communiques

    for erreur in mettre_a_jour_communiques(RACINE_PROJET / config["sentiment"]["dossier_communiques"]):
        journal(f"Communiqué non téléchargé (à copier à la main) : {erreur}")
    resume, phrases = scorer_communiques(config)
    resume.to_csv(sorties / "sentiment_banques.csv")
    phrases.to_csv(sorties / "sentiment_phrases_communiques.csv", index=False)


def mettre_a_jour_regimes(config: dict, sorties: Path, sentiments: dict[str, float]) -> None:
    """Régimes macro de chaque devise (historique) et état courant, avec le ton actuel."""
    from fxmacro.regime.construction import ligne_etat_regime, regimes_devise

    lignes = []
    for devise in config["regime"]["confiance"]:
        regimes = regimes_devise(devise, config)
        regimes.to_csv(sorties / f"regimes_{devise}.csv")
        lignes.append(ligne_etat_regime(devise, regimes, sentiments.get(devise)))
    pd.DataFrame(lignes).to_csv(sorties / "etat_regimes.csv", index=False)


def sentiment_par_devise(config: dict, sorties: Path) -> dict[str, float]:
    chemin = sorties / "sentiment_banques.csv"
    if not chemin.exists():
        return {}
    resume = pd.read_csv(chemin)
    colonne = f"score_{config['signal']['modele_sentiment']}"
    return dict(zip(resume["devise"], resume[colonne]))


def main(avec_sentiment: bool = True) -> int:
    config = charger_config()
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    debut = time.time()
    erreurs = []
    reinitialiser_suivi()

    if avec_sentiment:
        journal("Sentiment des banques centrales...")
        try:
            mettre_a_jour_sentiment(config, sorties)
        except Exception:
            erreurs.append("sentiment")
            journal("ERREUR sentiment (on garde le dernier résultat) :\n" + traceback.format_exc())
    sentiments = sentiment_par_devise(config, sorties)

    lignes = []
    for paire in config["paires"]:
        journal(f"Fair values {paire}...")
        try:
            tableau, resultat_beer = construire_tableau(paire, config)
            tableau.to_csv(sorties / f"fair_value_{paire}.csv")
            ligne = ligne_etat(paire, tableau, sentiments, config, resultat_beer)
            ligne.update({f"ref_{k}": v for k, v in evaluer_signal(ligne, ligne["spot_mensuel"], config).items()})
            lignes.append(ligne)
        except Exception:
            erreurs.append(paire)
            journal(f"ERREUR {paire} :\n" + traceback.format_exc())

    if lignes:
        pd.DataFrame(lignes).to_csv(sorties / "etat_paires.csv", index=False)

    journal("Régimes macro...")
    try:
        mettre_a_jour_regimes(config, sorties, sentiments)
    except Exception:
        erreurs.append("regimes")
        journal("ERREUR régimes :\n" + traceback.format_exc())
    # Sources servies depuis un cache périmé : signalées, mais ce ne sont pas des erreurs
    perimees = sources_perimees()
    for source in perimees:
        journal(f"Source périmée (cache de {source['age_jours']} jours réutilisé) : {source['nom']}")
    (sorties / "maj.json").write_text(json.dumps({
        "horodatage_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duree_s": round(time.time() - debut), "erreurs": erreurs, "sources_perimees": perimees,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    journal(f"Terminé en {time.time() - debut:.0f} s, erreurs : {erreurs or 'aucune'}")
    return 1 if erreurs else 0


if __name__ == "__main__":
    sys.exit(main(avec_sentiment="--sans-sentiment" not in sys.argv))
