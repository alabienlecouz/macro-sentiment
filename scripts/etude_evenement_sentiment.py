"""Étude d'événement : le change réagit-il, le jour même, au changement de ton mesuré par le modèle ?

Pour chaque communiqué (Fed, BCE, BoE, depuis 2015) :
  réaction = variation du change de la veille au lendemain, signée pour que + = appréciation de la
             devise de la banque (EURUSD pour la BCE, GBPUSD pour la BoE, -moyenne des deux pour la Fed) ;
  surprise de ton = score du communiqué - score du communiqué précédent de la même banque.
Régression réaction = a + b · surprise (erreurs robustes HC1), corrélation de rang, et la même chose
avec la variation du taux directeur en contrôle (le ton apporte-t-il plus que la décision ?).
Placebo : même fenêtre, une semaine avant le communiqué (rien ne doit apparaître).

Usage : python scripts/etude_evenement_sentiment.py   (après scripts/backtest_sentiment.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from fxmacro.config import chemin_projet, charger_config
from fxmacro.donnees.spots import obtenir_spot
from fxmacro.donnees.taux import obtenir_taux
from fxmacro.signaux.backtest_sentiment import reaction_evenement, surprise_ton

# Banque -> paires et signe : + si la paire monte quand la devise de la banque s'apprécie
EXPOSITION = {"BCE": {"EURUSD": 1}, "BOE": {"GBPUSD": 1}, "FED": {"EURUSD": -1, "GBPUSD": -1}}
NOMS_MODELES = {"zero_shot": "DeBERTa (NLI)", "finbert": "FinBERT"}


def reactions(dates: pd.DatetimeIndex, banque: str, spots: dict, params: dict, decalage: int = 0) -> pd.Series:
    """Appréciation de la devise de la banque autour de chaque communiqué (moyenne des paires exposées)."""
    valeurs = {}
    for jour in dates:
        r = [signe * reaction_evenement(spots[p], jour, params["jours_avant"], params["jours_apres"], decalage)
             for p, signe in EXPOSITION[banque].items()]
        valeurs[jour] = np.nanmean(r) if not all(np.isnan(r)) else np.nan
    return pd.Series(valeurs, name="reaction")


def regression(donnees: pd.DataFrame, colonnes: list[str]) -> dict:
    """MCO de la réaction (en points de base) sur les colonnes, erreurs HC1 : pente et t de la première."""
    x = sm.add_constant(donnees[colonnes])
    res = sm.OLS(10_000 * donnees["reaction"], x).fit(cov_type="HC1")
    return {"pente_pb": float(res.params[colonnes[0]]), "t": float(res.tvalues[colonnes[0]]),
            "p": float(res.pvalues[colonnes[0]]), "r2": float(res.rsquared)}


def variation_taux_reunion(taux_mensuel: pd.Series, jour: pd.Timestamp) -> float:
    """Variation du taux directeur le mois du communiqué (approximation de la décision)."""
    mois = jour.to_period("M").to_timestamp()
    precedent = mois - pd.DateOffset(months=1)
    if mois not in taux_mensuel.index or precedent not in taux_mensuel.index:
        return np.nan
    return float(taux_mensuel[mois] - taux_mensuel[precedent])


def ligne_resultat(modele: str, banque: str, fenetre: str, d: pd.DataFrame, colonnes: list[str]) -> dict:
    rho, p_rho = spearmanr(d["surprise"], d["reaction"])
    return {"modele": modele, "banque": banque, "fenetre": fenetre, "n": len(d), **regression(d, colonnes),
            "rho": float(rho), "p_rho": float(p_rho),
            "meme_sens": float((np.sign(d["surprise"]) == np.sign(d["reaction"])).mean())}


if __name__ == "__main__":
    config = charger_config()
    params = config["backtest_sentiment"]
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    scores = pd.read_csv(sorties / "sentiment_historique.csv", parse_dates=["date"])
    spots = {p: obtenir_spot(p, config) for p in params["paires"]}
    devises = config["sentiment"]["banques"]

    evenements, lignes = [], []
    for modele in params["modeles"]:
        tout, tout_placebo = [], []
        for banque in EXPOSITION:
            surprise = surprise_ton(scores, banque, f"score_{modele}").rename("surprise")
            taux = obtenir_taux(devises[banque], config)
            donnees = pd.concat([surprise, reactions(surprise.index, banque, spots, params)], axis=1)
            donnees["variation_taux"] = [variation_taux_reunion(taux, j) for j in donnees.index]
            donnees = donnees.dropna()
            placebo = pd.concat([surprise, reactions(surprise.index, banque, spots, params,
                                                     params["decalage_placebo_jours"])], axis=1).dropna()
            evenements.append(donnees.assign(banque=banque, modele=modele))
            tout.append(donnees)
            tout_placebo.append(placebo)
            lignes.append(ligne_resultat(modele, banque, "événement", donnees, ["surprise"]))
            lignes.append(ligne_resultat(modele, banque, "événement + taux", donnees, ["surprise", "variation_taux"]))
            lignes.append(ligne_resultat(modele, banque, "placebo (1 sem. avant)", placebo, ["surprise"]))
        lignes.append(ligne_resultat(modele, "toutes", "événement", pd.concat(tout), ["surprise"]))
        lignes.append(ligne_resultat(modele, "toutes", "placebo (1 sem. avant)", pd.concat(tout_placebo), ["surprise"]))

    resultats = pd.DataFrame(lignes)
    resultats.to_csv(sorties / "etude_evenement_sentiment.csv", index=False)

    # Figure : réaction du change contre surprise de ton, un panneau par modèle
    points = pd.concat(evenements)
    fig, axes = plt.subplots(1, len(params["modeles"]), figsize=(12, 4.6), sharey=True)
    couleurs = {"FED": "#2a78d6", "BCE": "#eb6834", "BOE": "#1baf7a"}
    for axe, modele in zip(axes, params["modeles"]):
        d = points[points.modele == modele]
        for banque, g in d.groupby("banque"):
            axe.scatter(g["surprise"], 100 * g["reaction"], s=16, alpha=0.7, color=couleurs[banque], label=banque)
        ligne = resultats[(resultats.modele == modele) & (resultats.banque == "toutes")
                          & (resultats.fenetre == "événement")].iloc[0]
        axe.set_title(f"{NOMS_MODELES[modele]} : ρ = {ligne.rho:+.2f} (p = {ligne.p_rho:.2f})", loc="left", fontsize=10)
        axe.axhline(0, color="#8a8984", lw=0.6)
        axe.axvline(0, color="#8a8984", lw=0.6)
        axe.set_xlabel("surprise de ton (score - score précédent)")
        axe.grid(alpha=0.25)
        axe.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("appréciation de la devise, veille → lendemain, %")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(sorties / "etude_evenement_sentiment.png", dpi=130)

    pd.set_option("display.width", 200)
    print(resultats.round(3).to_string(index=False))
    print(f"\nFigure : {sorties / 'etude_evenement_sentiment.png'}")
