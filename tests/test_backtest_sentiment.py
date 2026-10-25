"""Tests du backtest du sentiment et du nettoyage des communiqués archivés."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.sentiment.archives import nettoyer, retenu
from fxmacro.signaux.backtest_sentiment import (
    correlation_taux, ecart_centre, force_surcouche, reaction_evenement, score_mensuel, signal_sentiment,
    surprise_ton,
)


def test_score_mensuel_sans_information_future():
    scores = pd.DataFrame({"banque": ["FED", "FED"], "date": pd.to_datetime(["2020-01-29", "2020-03-18"]),
                           "score_zero_shot": [0.2, -0.4]})
    s = score_mensuel(scores, "FED", "score_zero_shot", anciennete_max_mois=4)
    assert s.loc["2020-01-01"] == 0.2          # publié le 29 janvier : connu fin janvier
    assert s.loc["2020-02-01"] == 0.2          # reporté tant qu'il n'y a pas de nouveau communiqué
    assert s.loc["2020-03-01"] == -0.4
    assert pd.Timestamp("2019-12-01") not in s.index


def test_score_mensuel_ne_reporte_pas_un_vieux_communique():
    scores = pd.DataFrame({"banque": ["BCE"], "date": pd.to_datetime(["2020-01-15"]), "score_finbert": [0.1]})
    s = score_mensuel(scores, "BCE", "score_finbert", anciennete_max_mois=2)
    assert s.loc["2020-03-01"] == 0.1 and np.isnan(s.loc["2020-04-01"])


def test_ecart_centre_utilise_seulement_le_passe():
    ecart = pd.Series([1.0, 1.0, 1.0, 5.0], index=pd.date_range("2020-01-01", periods=4, freq="MS"), name="e")
    centre = ecart_centre(ecart, min_obs=2)
    assert np.isnan(centre.iloc[1])            # pas encore 2 mois passés
    assert centre.iloc[2] == 0.0
    assert centre.iloc[3] == 4.0               # le mois courant n'entre pas dans sa propre moyenne


def test_signal_sentiment_seuils():
    s = signal_sentiment(pd.Series([0.2, -0.2, 0.1, np.nan]), seuil=0.15)
    assert s.tolist() == [1, -1, 0, 0]


def test_force_surcouche_dans_les_deux_sens():
    index = pd.date_range("2020-01-01", periods=5, freq="MS")
    signal = pd.Series([1, 1, -1, -1, 0], index=index)
    ecart = pd.Series([0.3, -0.3, -0.3, 0.05, 0.3], index=index)
    force = force_surcouche(signal, ecart, seuil=0.15)
    # +1 : base sous-évaluée, base plus hawkish = confirmé ; -1 : cotation sous-évaluée, base plus dovish = confirmé
    assert force.iloc[:4].tolist() == ["confirme", "contredit", "confirme", "modere"]
    assert pd.isna(force.iloc[4])


def test_correlation_taux_detecte_un_score_qui_precede_les_hausses():
    rng = np.random.default_rng(0)
    index = pd.date_range("2010-01-01", periods=120, freq="MS")
    score = pd.Series(rng.normal(size=120), index=index)
    taux = (score.shift(1).fillna(0) * 0.25).cumsum()   # le taux monte après un score hawkish
    assert correlation_taux(score, taux, 1)["rho"] > 0.8
    assert correlation_taux(score, taux, 6, pas=6)["n"] == pytest.approx(19, abs=1)


def test_reaction_evenement_fenetre_et_placebo():
    jours = pd.bdate_range("2020-01-01", periods=20)
    spot = pd.Series(np.exp(np.arange(20) * 0.01), index=jours)   # +1 % par jour
    jour = jours[10]
    assert reaction_evenement(spot, jour, 1, 1) == pytest.approx(0.02)
    assert reaction_evenement(spot, jour + pd.Timedelta(hours=1), 1, 1) == pytest.approx(0.02)  # jour suivant coté
    assert reaction_evenement(spot, jour, 1, 1, decalage=5) == pytest.approx(0.02)
    assert np.isnan(reaction_evenement(spot, jours[0], 1, 1))     # pas de veille


def test_surprise_ton_est_le_changement_depuis_le_communique_precedent():
    scores = pd.DataFrame({"banque": ["BCE", "BCE", "BCE", "FED"],
                           "date": pd.to_datetime(["2020-01-01", "2020-03-01", "2020-04-01", "2020-02-01"]),
                           "score_zero_shot": [0.1, 0.4, 0.2, 0.9]})
    s = surprise_ton(scores, "BCE", "score_zero_shot")
    assert s.round(2).tolist() == [0.3, -0.2]


def test_nettoyer_coupe_les_questions_de_la_bce_et_les_minutes_de_la_boe():
    bce = "Good afternoon.\nInflation remains too high.\nWhat will you do next month?\nWe will see."
    assert nettoyer("BCE", bce) == "Good afternoon.\nInflation remains too high."
    boe = "At its meeting, the MPC voted to maintain Bank Rate.\n1) Before turning to its decision, the Committee"
    assert nettoyer("BOE", boe) == "At its meeting, the MPC voted to maintain Bank Rate."


def test_retenu_ecarte_les_annonces_de_la_fed_hors_reunion():
    phrases = " ".join(f"The Committee discussed point number {i} at length today." for i in range(6))
    assert retenu("FED", phrases + " The target range for the federal funds rate is unchanged.")
    assert not retenu("FED", phrases)
