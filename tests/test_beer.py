"""Tests du BEER sur des données simulées dont on connaît le vrai modèle."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.fair_value.beer import (
    demi_vie, estimer_beer, fair_value_beer, inflation_glissement_annuel, taux_change_reel, taux_reel,
)

N = 300
DATES = pd.date_range("2000-01-01", periods=N, freq="MS")


def _ar1(rho: float, rng: np.random.Generator, sigma: float = 0.02) -> np.ndarray:
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + rng.normal(0, sigma)
    return x


def test_retrouve_les_coefficients_si_cointegration():
    rng = np.random.default_rng(1)
    x1 = pd.Series(np.cumsum(rng.normal(0, 0.1, N)), index=DATES)  # marche aléatoire
    x2 = pd.Series(np.cumsum(rng.normal(0, 0.1, N)), index=DATES)
    q = 0.5 + 0.3 * x1 + 0.1 * x2 + _ar1(0.8, rng)                  # résidu stationnaire
    res = estimer_beer(pd.Series(q, index=DATES), pd.DataFrame({"x1": x1, "x2": x2}))
    assert res.coefficients["x1"] == pytest.approx(0.3, abs=0.03)
    assert res.coefficients["x2"] == pytest.approx(0.1, abs=0.03)
    assert res.p_value_cointegration < 0.05


def test_regression_fallacieuse_detectee():
    # q et x sont deux marches aléatoires indépendantes : pas de cointégration
    rng = np.random.default_rng(2)
    x = pd.Series(np.cumsum(rng.normal(0, 0.1, N)), index=DATES)
    q = pd.Series(np.cumsum(rng.normal(0, 0.1, N)), index=DATES)
    res = estimer_beer(q, pd.DataFrame({"x": x}))
    assert res.p_value_cointegration > 0.10


def test_demi_vie_ar1():
    rng = np.random.default_rng(3)
    residus = pd.Series(_ar1(0.5, rng)).iloc[1:]
    # rho = 0,5 -> l'écart est divisé par 2 en 1 période
    assert demi_vie(residus) == pytest.approx(1.0, abs=0.2)


def test_fair_value_beer_retrouve_le_spot_si_q_exact():
    spot = pd.Series([1.10, 1.20], index=DATES[:2])
    cpi_base = pd.Series([100.0, 102.0], index=DATES[:2])
    cpi_cotation = pd.Series([100.0, 103.0], index=DATES[:2])
    q = taux_change_reel(spot, cpi_base, cpi_cotation)
    assert np.allclose(fair_value_beer(q, cpi_base, cpi_cotation), spot)


def test_taux_reel():
    cpi = pd.Series(np.r_[np.full(12, 100.0), 103.0], index=DATES[:13])
    assert inflation_glissement_annuel(cpi).iloc[-1] == pytest.approx(3.0)
    nominal = pd.Series(5.0, index=DATES[:13])
    assert taux_reel(nominal, cpi).iloc[-1] == pytest.approx(2.0)


def test_cpi_non_publie_n_invente_pas_d_inflation_dans_le_beer():
    """Même chemin que pipeline.charger_donnees_paire : inflation calculée sur le CPI publié,
    puis alignée (reportée) ; jamais calculée sur un CPI reporté."""
    from fxmacro.donnees.alignement import aligner_mensuel
    from fxmacro.fair_value.beer import construire_regresseurs, inflation_publiee

    index = pd.date_range("2020-01-01", periods=30, freq="MS")
    cpi = pd.Series(100 * 1.003 ** np.arange(30), index=index)
    cpi_publie = cpi.drop(index[25])                       # mois 25 : CPI pas encore publié
    serie = lambda v: pd.Series(v, index=index)
    tableau = aligner_mensuel({
        "spot": serie(1.0), "cpi_base": cpi_publie, "cpi_cotation": serie(100.0),
        "inflation_base": inflation_publiee(cpi_publie), "inflation_cotation": serie(0.0),
        "taux_base": serie(2.0), "taux_cotation": serie(0.0),
        "tde_base": serie(100.0), "tde_cotation": serie(100.0), "bc_base": serie(0.0), "bc_cotation": serie(0.0),
    }, combler_max=3, reference="spot")
    reel = construire_regresseurs(tableau)["diff_taux_reel"]

    inflation_mois_24 = 100 * (cpi[index[24]] / cpi[index[12]] - 1)
    # Mois 25 : dernière inflation calculée (mois 24), pas CPI(24) / CPI(13) - 1
    assert reel[index[25]] == pytest.approx(2.0 - inflation_mois_24)
    inventee = 100 * (cpi[index[24]] / cpi[index[13]] - 1)
    assert reel[index[25]] != pytest.approx(2.0 - inventee)
    # Le niveau de prix, lui, reste reporté (correct pour la PPA)
    assert tableau.loc[index[25], "cpi_base"] == cpi[index[24]]
