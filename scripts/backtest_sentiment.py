"""Backtest du sentiment des banques centrales (Fed, BCE, BoE), EURUSD et GBPUSD, depuis 2015.

Étapes (protocole fixé dans config.yaml, section backtest_sentiment, avant de voir les résultats) :
  0) collecte des communiqués archivés et score DeBERTa / FinBERT de chacun (mis en cache) ;
  1) validation : le score précède-t-il les variations du taux directeur ?
  2) signal sentiment seul (règle du bot, puis variante centrée), horizons 1 et 3 mois ;
  3) surcouche : signaux de valorisation PPA confirmés / contredits / modérés, réussite à 12 mois.

Usage : python scripts/backtest_sentiment.py [--collecter]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fxmacro.config import chemin_projet, charger_config
from fxmacro.donnees.alignement import journalier_vers_mensuel
from fxmacro.donnees.historique_long import tableau_long
from fxmacro.donnees.spots import obtenir_spot
from fxmacro.donnees.taux import obtenir_taux
from fxmacro.sentiment.archives import collecter, retenu
from fxmacro.sentiment.analyse import scorer_tous
from fxmacro.sentiment.textes import charger_communiques
from fxmacro.signaux.backtest import evaluer, ppa_temps_reel, signal_depuis_z
from fxmacro.signaux.backtest_sentiment import (
    correlation_taux, ecart_centre, force_surcouche, p_binomiale, score_mensuel, signal_sentiment,
)

NOMS_MODELES = {"zero_shot": "DeBERTa (NLI)", "finbert": "FinBERT"}


def scores_historiques(config: dict, collecte: bool) -> pd.DataFrame:
    """Score de chaque communiqué archivé (calcul long la première fois, puis lu dans sorties/)."""
    params = config["backtest_sentiment"]
    fichier = chemin_projet(config["general"]["dossier_sorties"]) / "sentiment_historique.csv"
    dossier = chemin_projet(params["dossier_communiques"])
    if collecte:
        debut, fin = params["annees"]
        banques = [*params["paires"].values(), params["banque_cotation"]]
        for banque in banques:
            for erreur in collecter(banque, range(debut, fin + 1), dossier):
                print("  ", erreur)
    if collecte or not fichier.exists():
        communiques = charger_communiques(dossier)
        communiques = communiques[[retenu(b, t) for b, t in zip(communiques.banque, communiques.texte)]]
        scorer_tous(communiques, params["modeles"], config).to_csv(fichier, index=False)
    return pd.read_csv(fichier, parse_dates=["date"])


def resultat_signal(signal: pd.Series, spot: pd.Series, horizon: int) -> dict:
    """Statistiques du backtest et p-value binomiale sur l'échantillon indépendant (glouton)."""
    stats = evaluer(signal, spot, horizon=horizon)
    reussites = round(stats["n_glouton"] * stats["taux_reussite_glouton"]) if stats["n_glouton"] else 0
    longs = float((signal[signal != 0] > 0).mean()) if (signal != 0).any() else float("nan")
    return {**stats, "p_glouton": p_binomiale(reussites, stats["n_glouton"]), "part_hausse": longs}


def figure_scores(mensuels: dict, taux: dict, banques: list[str], modeles: list[str], chemin: Path) -> None:
    """Score de chaque banque (un trait par modèle) et taux directeur (axe de droite)."""
    couleurs = {"zero_shot": "#2a78d6", "finbert": "#eb6834"}
    fig, axes = plt.subplots(len(banques), 1, figsize=(12, 3.2 * len(banques)), sharex=True)
    for axe, banque in zip(axes, banques):
        for modele in modeles:
            serie = mensuels[(banque, modele)]
            axe.plot(serie.index, serie, color=couleurs[modele], lw=1.5, label=NOMS_MODELES[modele])
        axe.axhline(0, color="#8a8984", lw=0.6)
        axe.set_ylabel("dovish ← score → hawkish", fontsize=8)
        axe.set_title(banque, loc="left", fontsize=11)
        droite = axe.twinx()
        droite.step(taux[banque].index, taux[banque], where="post", color="#3a3a3a", lw=1.2, ls="--",
                    label="Taux directeur (%)")
        droite.set_ylabel("taux directeur, %", fontsize=8)
        for a in (axe, droite):
            a.spines[["top"]].set_visible(False)
        axe.grid(alpha=0.25)
        if axe is axes[0]:
            traits = axe.get_legend_handles_labels()[0] + droite.get_legend_handles_labels()[0]
            axe.legend(traits, [t.get_label() for t in traits], frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)


if __name__ == "__main__":
    config = charger_config()
    params = config["backtest_sentiment"]
    seuil = config["signal"]["seuil_sentiment"]
    seuil_z = config["signal"]["seuil_z"]
    sorties = chemin_projet(config["general"]["dossier_sorties"])
    debut = pd.Timestamp(params["debut"])

    scores = scores_historiques(config, "--collecter" in sys.argv)
    banques = [params["banque_cotation"], *params["paires"].values()]
    devises = config["sentiment"]["banques"]
    mensuels = {(b, m): score_mensuel(scores, b, f"score_{m}", params["anciennete_max_mois"]).loc[debut:]
                for b in banques for m in params["modeles"]}
    taux = {b: obtenir_taux(devises[b], config).loc[debut:] for b in banques}

    # 1) Validation : le ton précède-t-il les décisions ?
    h_taux = params["horizon_taux_mois"]
    validation = []
    for (banque, modele), serie in mensuels.items():
        # Variations sur h mois qui se chevauchent : la p-value n'est calculée que sur un mois tous les h
        complet = correlation_taux(serie, taux[banque], h_taux)
        independant = correlation_taux(serie, taux[banque], h_taux, pas=h_taux)
        validation.append({"banque": banque, "modele": modele, "rho": complet["rho"], "n": complet["n"],
                           "rho_independant": independant["rho"], "p_independant": independant["p_value"],
                           "n_independant": independant["n"]})
    validation = pd.DataFrame(validation)

    # 2) Signal sentiment seul, 3) surcouche
    lignes, surcouche = [], []
    for paire, banque_base in params["paires"].items():
        spot = journalier_vers_mensuel(obtenir_spot(paire, config), "fin").loc[debut:]
        z_val = ppa_temps_reel(tableau_long(paire, config), params["debut"])["z"].loc[debut:]
        signal_val = signal_depuis_z(z_val, seuil_z)
        for modele in params["modeles"]:
            ecart = (mensuels[(banque_base, modele)] - mensuels[(params["banque_cotation"], modele)]).rename("ecart")
            variantes = {"règle du bot": ecart, "écart centré": ecart_centre(ecart, params["min_obs_moyenne"])}
            for variante, serie in variantes.items():
                for h in params["horizons_mois"]:
                    lignes.append({"paire": paire, "modele": modele, "variante": variante, "horizon": h,
                                   **resultat_signal(signal_sentiment(serie, seuil), spot, h)})
            force = force_surcouche(signal_val, ecart, seuil)
            groupes = {"valorisation seule": signal_val != 0, **{g: force == g for g in ["confirme", "modere", "contredit"]}}
            for groupe, masque in groupes.items():
                signal_groupe = signal_val.where(masque, 0)
                surcouche.append({"paire": paire, "modele": modele, "groupe": groupe,
                                  **resultat_signal(signal_groupe, spot, params["horizon_surcouche_mois"])})

    resultats = pd.DataFrame(lignes)
    resultats_surcouche = pd.DataFrame(surcouche)
    resultats.to_csv(sorties / "backtest_sentiment.csv", index=False)
    resultats_surcouche.to_csv(sorties / "backtest_sentiment_surcouche.csv", index=False)
    validation.to_csv(sorties / "backtest_sentiment_validation.csv", index=False)
    figure_scores(mensuels, taux, banques, params["modeles"], sorties / "sentiment_historique.png")

    pd.set_option("display.width", 220)
    print(f"=== 1) Le score précède-t-il le taux directeur ? (rang, variation sur {h_taux} mois) ===")
    print(validation.round(2).to_string(index=False))
    colonnes = ["part_actifs", "part_hausse", "n", "taux_reussite", "rendement_moyen_%", "n_glouton",
                "taux_reussite_glouton", "p_glouton"]
    print(f"\n=== 2) Signal sentiment seul (seuil {seuil}) ===")
    print(resultats.set_index(["paire", "modele", "variante", "horizon"])[colonnes].round(2).to_string())
    print(f"\n=== 3) Surcouche : signaux de valorisation PPA par force, {params['horizon_surcouche_mois']} mois ===")
    print(resultats_surcouche.set_index(["paire", "modele", "groupe"])[["n", "taux_reussite", "rendement_moyen_%",
                                                                       "n_glouton", "taux_reussite_glouton"]]
          .round(2).to_string())
    print(f"\nFigure : {sorties / 'sentiment_historique.png'}")
