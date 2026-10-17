"""Tests de la PPA relative sur des données synthétiques dont on connaît la réponse."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.fair_value.ppa import calibrer_constante, fair_value_ppa, ratio_prix

DATES = pd.date_range("2010-01-01", periods=24, freq="MS")


def test_prix_constants_donne_fair_value_constante():
    # Sans inflation relative, la fair value est plate et égale à la moyenne géométrique du spot
    spot = pd.Series(np.linspace(1.0, 1.4, 24), index=DATES)
    cpi = pd.Series(100.0, index=DATES)
    fv = fair_value_ppa(spot, cpi, cpi, methode="moyenne")
    attendu = np.exp(np.log(spot).mean())
    assert np.allclose(fv, attendu)


def test_inflation_cotation_plus_forte_fait_monter_la_fair_value():
    # Si les prix du pays YYY doublent par rapport à XXX, la fair value XXXYYY double
    spot = pd.Series(1.0, index=DATES)
    cpi_base = pd.Series(100.0, index=DATES)
    cpi_cotation = pd.Series(np.linspace(100.0, 200.0, 24), index=DATES)
    fv = fair_value_ppa(spot, cpi_base, cpi_cotation, methode="date", date_base="2010-01-01")
    assert fv.iloc[-1] / fv.iloc[0] == pytest.approx(2.0)


def test_methode_date_egale_le_spot_a_la_date_de_base():
    spot = pd.Series(np.linspace(1.1, 1.3, 24), index=DATES)
    cpi_base = pd.Series(np.linspace(100, 110, 24), index=DATES)
    cpi_cotation = pd.Series(np.linspace(100, 120, 24), index=DATES)
    fv = fair_value_ppa(spot, cpi_base, cpi_cotation, methode="date", date_base="2010-06-01")
    assert fv.loc["2010-06-01"] == pytest.approx(spot.loc["2010-06-01"])


def test_methode_moyenne_donne_un_ecart_log_moyen_nul():
    rng = np.random.default_rng(0)
    spot = pd.Series(1.2 * np.exp(rng.normal(0, 0.05, 24)), index=DATES)
    cpi_base = pd.Series(np.linspace(100, 105, 24), index=DATES)
    cpi_cotation = pd.Series(np.linspace(100, 115, 24), index=DATES)
    fv = fair_value_ppa(spot, cpi_base, cpi_cotation, methode="moyenne")
    assert np.log(spot / fv).mean() == pytest.approx(0.0, abs=1e-12)


def test_periode_de_calibration_restreinte():
    spot = pd.Series([1.0] * 12 + [2.0] * 12, index=DATES)
    cpi = pd.Series(100.0, index=DATES)
    k = calibrer_constante(spot, ratio_prix(cpi, cpi), "moyenne", periode=("2010-01-01", "2010-12-01"))
    assert k == pytest.approx(1.0)


def test_methode_inconnue_leve_une_erreur():
    spot = pd.Series(1.0, index=DATES)
    with pytest.raises(ValueError):
        calibrer_constante(spot, spot, methode="nimporte")
