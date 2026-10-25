"""Tests du banc d'essai des fair values, sur données synthétiques dont on connaît la réponse."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.fair_value.banc_essai import (
    ecart_moyenne_mobile, regression_panel, regression_predictive, rendement_futur, stationnarite,
)

N = 300
DATES = pd.date_range("2000-01-01", periods=N, freq="MS")


def ar1(rho: float, rng: np.random.Generator, sigma: float = 0.05) -> pd.Series:
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + rng.normal(0, sigma)
    return pd.Series(x, index=DATES)


def test_rendement_futur_aligne_sur_t():
    spot = pd.Series(np.exp([0.0, 0.1, 0.3, 0.6]), index=DATES[:4])
    r = rendement_futur(spot, 2)
    assert r.iloc[0] == pytest.approx(0.3) and r.iloc[1] == pytest.approx(0.5)
    assert r.iloc[2:].isna().all()                       # t+2 pas encore connu


def test_regression_retrouve_beta():
    rng = np.random.default_rng(0)
    ecart = ar1(0.9, rng)
    rendement = -0.5 * ecart + pd.Series(rng.normal(0, 0.01, N), index=DATES)
    r = regression_predictive(ecart, rendement, horizon_mois=12)
    assert r["beta"] == pytest.approx(-0.5, abs=0.02)
    assert r["t"] < -5 and r["n_indep"] == N // 12


def test_regression_panel_retrouve_beta_commun():
    rng = np.random.default_rng(1)
    choc_commun = pd.Series(rng.normal(0, 0.02, N), index=DATES)   # ex. mouvement du dollar
    ecarts, rendements = {}, {}
    for paire in ["A", "B", "C"]:
        ecarts[paire] = ar1(0.95, rng)
        rendements[paire] = -0.3 * ecarts[paire] + choc_commun + pd.Series(rng.normal(0, 0.01, N), index=DATES)
    r = regression_panel(ecarts, rendements, horizon_mois=12)
    assert r["beta"] == pytest.approx(-0.3, abs=0.03)
    assert r["n"] == 3 * N and r["n_dates_indep"] == N // 12


def test_stationnarite_distingue_retour_a_la_moyenne_et_marche_aleatoire():
    rng = np.random.default_rng(2)
    retour = stationnarite(ar1(0.8, rng))
    assert retour["p_adf"] < 0.01
    assert retour["demi_vie_mois"] == pytest.approx(np.log(0.5) / np.log(0.8), abs=1.0)
    marche = stationnarite(pd.Series(np.cumsum(rng.normal(0, 0.05, N)), index=DATES))
    assert marche["p_adf"] > 0.10


def test_moyenne_mobile_sans_information_future():
    rng = np.random.default_rng(3)
    spot = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.02, N))), index=DATES)
    t = DATES[150]
    modifie = spot.copy()
    modifie[modifie.index > t] *= 3
    pd.testing.assert_series_equal(ecart_moyenne_mobile(spot, 60).loc[:t], ecart_moyenne_mobile(modifie, 60).loc[:t])
    assert ecart_moyenne_mobile(spot, 60).iloc[:59].isna().all()


def test_panel_v2_corrige_une_paire_dominante():
    # Paire « D » : écart décalé (+0,5) et très volatil, sans pouvoir prédictif ; paires A et B : β = -0,3
    rng = np.random.default_rng(4)
    ecarts = {"A": ar1(0.95, rng), "B": ar1(0.95, rng), "D": 0.5 + 5 * ar1(0.95, rng)}
    rendements = {p: -0.3 * ecarts[p] + pd.Series(rng.normal(0, 0.01, N), index=DATES) for p in ["A", "B"]}
    rendements["D"] = pd.Series(rng.normal(0, 0.01, N), index=DATES)
    v1 = regression_panel(ecarts, rendements, 12)
    v2 = regression_panel(ecarts, rendements, 12, effets_fixes=True, standardiser=True)
    # La paire D, très volatile et sans pouvoir prédictif, noie la relation dans la v1 (unités log) ;
    # la v2 (effets fixes, écarts par écart-type) l'extrait beaucoup plus nettement
    assert v2["beta"] < 0
    assert v2["t"] < v1["t"] - 3


def test_standardisation_sans_information_future():
    rng = np.random.default_rng(5)
    ecart = ar1(0.9, rng)
    t = DATES[120]
    modifie = ecart.copy()
    modifie[modifie.index > t] *= 10
    from fxmacro.fair_value.banc_essai import standardiser_temps_reel
    pd.testing.assert_series_equal(standardiser_temps_reel(ecart).loc[:t], standardiser_temps_reel(modifie).loc[:t])
