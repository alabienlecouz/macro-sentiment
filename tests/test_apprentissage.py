"""Tests du walk-forward : purge, absence d'information future, évaluation."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.signaux.apprentissage import (
    construire_panel, diebold_mariano, evaluer, masque_entrainement, variables_marche, walk_forward,
)

MODELES = {"logistique": {"C": 1.0},
           "xgboost": {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8}}


def panel_simule(graine: int = 0, n_mois: int = 240) -> pd.DataFrame:
    """Deux paires ; le rendement futur dépend (faiblement) de la variable x."""
    rng = np.random.default_rng(graine)
    index = pd.date_range("2000-01-01", periods=n_mois, freq="MS")
    par_paire = {}
    for paire in ["AAA", "BBB"]:
        x = rng.normal(size=n_mois)
        # ln S(t+1) - ln S(t) = rendements[t+1], qui dépend de x(t)
        rendements = 0.01 * np.roll(x, 1) + rng.normal(0, 0.02, n_mois)
        spot = pd.Series(np.exp(np.cumsum(rendements)), index=index)
        par_paire[paire] = pd.DataFrame({"spot": spot, "x": x}, index=index)
    return construire_panel(par_paire, ["x"], horizon=1)


def test_purge_exclut_les_rendements_qui_debordent_sur_le_test():
    dates = pd.Series(pd.date_range("2009-01-01", "2010-12-01", freq="MS"))
    garde = dates[masque_entrainement(dates, pd.Timestamp("2011-01-01"), horizon=12)]
    assert garde.max() == pd.Timestamp("2009-12-01")   # 2009-12 + 12 mois = 2010-12 < 2011-01


def test_variables_marche_n_utilisent_que_le_passe():
    spot = pd.Series(np.exp(np.linspace(0, 1, 30)), index=pd.date_range("2000-01-01", periods=30, freq="MS"))
    avant = variables_marche(spot)
    modifie = spot.copy()
    modifie.iloc[20:] *= 3
    apres = variables_marche(modifie)
    pd.testing.assert_frame_equal(avant.iloc[:20], apres.iloc[:20])


def test_walk_forward_sans_information_future():
    panel = panel_simule()
    annees = range(2010, 2019)
    pred, _, _ = walk_forward(panel, ["x"], MODELES, horizon=1, annees=annees, graine=0)
    modifie = panel.copy()
    futur = modifie["date"] >= pd.Timestamp("2015-01-01")
    modifie.loc[futur, "y"] = 1 - modifie.loc[futur, "y"]          # le futur est complètement changé
    modifie.loc[futur, "rendement"] *= -1
    pred_modifie, _, _ = walk_forward(modifie, ["x"], MODELES, horizon=1, annees=annees, graine=0)
    passe = pred["date"] < pd.Timestamp("2015-01-01")
    for nom in MODELES:
        np.testing.assert_allclose(pred.loc[passe, f"proba_{nom}"], pred_modifie.loc[passe, f"proba_{nom}"])


def test_walk_forward_trouve_une_relation_simulee():
    panel = panel_simule(n_mois=400)
    pred, internes, _ = walk_forward(panel, ["x"], {"logistique": {"C": 1.0}}, horizon=1,
                                     annees=range(2010, 2033), graine=0)
    direction = np.where(pred["proba_logistique"] > 0.5, 1, -1)
    assert evaluer(pred, pd.Series(direction, index=pred.index), 1)["reussite"] > 0.6
    assert set(internes["modele"]) == {"logistique"}


def test_echantillon_conservateur_un_mois_sur_h():
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    pred = pd.DataFrame({"date": np.repeat(dates, 2), "rendement": 0.01, "y": 1})
    resultat = evaluer(pred, pd.Series(1, index=pred.index), horizon=12)
    assert resultat["n_conservateur"] == 2 and resultat["reussite_conservateur"] == 1.0


def test_diebold_mariano_detecte_un_meilleur_modele():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-01", periods=200, freq="MS")
    y = rng.integers(0, 2, 200)
    pred = pd.DataFrame({"date": dates, "y": y, "proba_bon": np.where(y == 1, 0.7, 0.3), "proba_ref": 0.5})
    assert diebold_mariano(pred, "bon", "ref", horizon=1)["gain_log_loss"] > 0
    assert diebold_mariano(pred, "bon", "ref", horizon=1)["p_dm"] < 0.01
