"""Random Forest et XGBoost contre régression logistique : prédire la direction du change.

Protocole (config.yaml, section apprentissage, fixé avant les résultats) :
- panel mensuel des 9 paires du G10, variables calculées en temps réel (données passées seules) ;
- cible y = 1 si ln S(t+h) - ln S(t) > 0 ;
- walk-forward annuel : pour tester l'année Y, entraînement sur les mois t tels que t + h < janvier Y
  (purge : leur rendement futur est entièrement connu avant le test), toutes paires confondues ;
- standardisation des variables ajustée sur l'entraînement seul (pipeline scikit-learn).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fxmacro.donnees.bri import obtenir_taux_directeur_bri
from fxmacro.donnees.historique_long import config_long, tableau_long
from fxmacro.fair_value.ppa_productivite import ppa_productivite_temps_reel
from fxmacro.signaux.backtest import ppa_temps_reel

DEBUT_EURO = pd.Timestamp("1999-01-01")


# ---------- Variables ----------

def code_taux(devise: str, config: dict) -> tuple[str, float | None]:
    """Code BRI du taux directeur d'une devise et valeur des mois sans taux cible."""
    if devise in config["taux_directeurs"]:
        params = config["taux_directeurs"][devise]
        return params["id"], params.get("si_absent")
    return config["historique_long"]["cpi_supplementaires"][devise]["id"], None


def taux_long(devise: str, config: dict) -> pd.Series:
    """Taux directeur mensuel en % ; pour l'euro, taux allemand avant 1999."""
    long = config_long(config)
    code, si_absent = code_taux(devise, config)
    taux = obtenir_taux_directeur_bri(code, long).resample("MS").last()
    if devise == "EUR":
        avant = obtenir_taux_directeur_bri(config["apprentissage"]["taux_avant_euro"], long).resample("MS").last()
        taux = pd.concat([avant[avant.index < DEBUT_EURO], taux[taux.index >= DEBUT_EURO]])
    return taux.fillna(si_absent) if si_absent is not None else taux


def variables_marche(spot: pd.Series) -> pd.DataFrame:
    """Momentum sur 3 et 12 mois et volatilité des 12 derniers rendements mensuels (passé seul)."""
    log_spot = np.log(spot)
    return pd.DataFrame({"momentum_3m": log_spot - log_spot.shift(3),
                         "momentum_12m": log_spot - log_spot.shift(12),
                         "volatilite_12m": log_spot.diff().rolling(12).std()})


def rendement_futur(spot: pd.Series, horizon: int) -> pd.Series:
    """ln S(t+h) - ln S(t)."""
    return np.log(spot.shift(-horizon) / spot).rename("rendement")


def variables_paire(paire: str, config: dict) -> pd.DataFrame:
    """Variables mensuelles d'une paire, toutes calculées avec les seules données passées."""
    debut = config["apprentissage"]["debut"]
    tableau = tableau_long(paire, config)
    infos = config_long(config)["paires"][paire]
    carry = (taux_long(infos["base"], config) - taux_long(infos["cotation"], config)).reindex(tableau.index)
    variables = pd.DataFrame({"spot": tableau["spot"],
                              "z_ppa": ppa_temps_reel(tableau, debut)["z"],
                              "z_ppa_productivite": ppa_productivite_temps_reel(tableau, debut)["z"],
                              "carry": carry.ffill(limit=3)}, index=tableau.index)
    return variables.join(variables_marche(tableau["spot"])).loc[debut:]


def construire_panel(par_paire: dict[str, pd.DataFrame], variables: list[str], horizon: int) -> pd.DataFrame:
    """Panel long (date, paire) : variables, rendement futur et cible ; lignes incomplètes retirées."""
    morceaux = []
    for paire, donnees in par_paire.items():
        d = donnees[variables].copy()
        d["rendement"] = rendement_futur(donnees["spot"], horizon)
        d["paire"] = paire
        morceaux.append(d.dropna())
    panel = pd.concat(morceaux).rename_axis("date").reset_index()
    panel["y"] = (panel["rendement"] > 0).astype(int)
    return panel.sort_values(["date", "paire"]).reset_index(drop=True)


# ---------- Modèles et walk-forward ----------

def creer_modele(nom: str, params: dict, graine: int):
    """Pipeline standardisation + modèle, hyperparamètres fixés dans la config."""
    if nom == "logistique":
        modele = LogisticRegression(C=params["C"], max_iter=1000)
    elif nom == "foret_aleatoire":
        modele = RandomForestClassifier(**params, random_state=graine, n_jobs=-1)
    elif nom == "xgboost":
        modele = XGBClassifier(**params, random_state=graine, eval_metric="logloss", n_jobs=4)
    else:
        raise ValueError(f"Modèle inconnu : {nom}")
    return make_pipeline(StandardScaler(), modele)


def masque_entrainement(dates: pd.Series, debut_test: pd.Timestamp, horizon: int) -> pd.Series:
    """Mois dont le rendement futur est entièrement connu avant le début du test : t + h < début."""
    return dates + pd.DateOffset(months=horizon) < debut_test


def walk_forward(panel: pd.DataFrame, variables: list[str], modeles: dict, horizon: int,
                 annees: range, graine: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Prédictions hors échantillon année par année.

    Renvoie (prédictions : une colonne proba_<modèle> par modèle, réussite dans l'échantillon
    d'entraînement par année et modèle, modèles ajustés {année: {modèle: pipeline}}).
    """
    predictions, internes, ajustes = [], [], {}
    for annee in annees:
        debut_test = pd.Timestamp(f"{annee}-01-01")
        test = panel[(panel["date"] >= debut_test) & (panel["date"] < debut_test + pd.DateOffset(years=1))].copy()
        entrainement = panel[masque_entrainement(panel["date"], debut_test, horizon)]
        if test.empty or entrainement["y"].nunique() < 2:
            continue
        ajustes[annee] = {}
        for nom, params in modeles.items():
            modele = creer_modele(nom, params, graine).fit(entrainement[variables], entrainement["y"])
            test[f"proba_{nom}"] = modele.predict_proba(test[variables])[:, 1]
            internes.append({"annee": annee, "modele": nom,
                             "reussite_entrainement": float(modele.score(entrainement[variables], entrainement["y"]))})
            ajustes[annee][nom] = modele
        test["annee"] = annee
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(internes), ajustes


# ---------- Évaluation ----------

def log_loss_ligne(y: pd.Series, proba: pd.Series) -> pd.Series:
    p = proba.clip(1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def moyenne_newey_west(serie: pd.Series, retards: int) -> tuple[float, float]:
    """Moyenne et t de Newey-West (autocorrélation due aux horizons qui se chevauchent)."""
    res = sm.OLS(serie.to_numpy(), np.ones(len(serie))).fit(cov_type="HAC", cov_kwds={"maxlags": max(retards, 1)})
    return float(res.params[0]), float(res.tvalues[0])


def evaluer(predictions: pd.DataFrame, direction: pd.Series, horizon: int, proba: pd.Series | None = None) -> dict:
    """Qualité d'une prédiction de direction (+1 / -1) hors échantillon.

    Échantillon conservateur : rendement signé moyen des paires à chaque date (les paires partagent
    le dollar), puis un mois sur h (horizons qui ne se chevauchent pas), test binomial contre 50 %.
    """
    signe = predictions["rendement"] * direction
    par_date = signe.groupby(predictions["date"]).mean()
    conservateur = par_date.iloc[::horizon]
    reussites = int((conservateur > 0).sum())
    moyenne, t = moyenne_newey_west(par_date, horizon - 1)
    resultat = {"n": len(predictions), "reussite": float((signe > 0).mean()),
                "n_conservateur": len(conservateur), "reussite_conservateur": reussites / len(conservateur),
                "p_conservateur": float(binomtest(reussites, len(conservateur), 0.5).pvalue),
                "rendement_moyen_%": 100 * moyenne, "t_newey_west": t}
    if proba is not None:
        resultat["log_loss"] = float(log_loss_ligne(predictions["y"], proba).mean())
        resultat["brier"] = float(((proba - predictions["y"]) ** 2).mean())
    return resultat


def diebold_mariano(predictions: pd.DataFrame, modele: str, reference: str, horizon: int) -> dict:
    """Le modèle a-t-il une log-loss plus faible que la référence ? Écart moyen par date, t de Newey-West."""
    ecart = (log_loss_ligne(predictions["y"], predictions[f"proba_{reference}"])
             - log_loss_ligne(predictions["y"], predictions[f"proba_{modele}"]))
    moyenne, t = moyenne_newey_west(ecart.groupby(predictions["date"]).mean(), horizon - 1)
    return {"gain_log_loss": moyenne, "t_dm": t, "p_dm": float(2 * norm.sf(abs(t)))}


def importance_permutation(predictions: pd.DataFrame, ajustes: dict, modele: str, variables: list[str],
                           repetitions: int, graine: int) -> pd.Series:
    """Hausse de la log-loss hors échantillon quand on mélange une variable (au sein de chaque année)."""
    rng = np.random.default_rng(graine)

    def log_loss_totale(donnees: pd.DataFrame) -> float:
        probas = [ajustes[a][modele].predict_proba(g[variables])[:, 1] for a, g in donnees.groupby("annee")]
        y = np.concatenate([g["y"].to_numpy() for _, g in donnees.groupby("annee")])
        return float(log_loss_ligne(pd.Series(y), pd.Series(np.concatenate(probas))).mean())

    reference = log_loss_totale(predictions)
    hausses = {}
    for variable in variables:
        essais = []
        for _ in range(repetitions):
            melange = predictions.copy()
            melange[variable] = melange.groupby("annee")[variable].transform(lambda s: rng.permutation(s.to_numpy()))
            essais.append(log_loss_totale(melange) - reference)
        hausses[variable] = float(np.mean(essais))
    return pd.Series(hausses).sort_values(ascending=False)
