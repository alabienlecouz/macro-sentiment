"""Tests de la parité des taux sur des cas calculables à la main."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.fair_value.parite_taux import (
    decaler_mois, fair_value_uip, forward_theorique, pourcent_vers_decimal, prime_terme_annualisee,
)

DATES = pd.date_range("2020-01-01", periods=24, freq="MS")


def test_taux_egaux_forward_egal_spot():
    spot = pd.Series(1.10, index=DATES)
    i = pd.Series(0.03, index=DATES)
    assert np.allclose(forward_theorique(spot, i, i, 1.0), spot)


def test_cip_valeur_connue():
    # EURUSD = 1.10, i_EUR = 2 %, i_USD = 4 % : F = 1.10 * 1.04 / 1.02 sur 1 an
    spot = pd.Series([1.10])
    f = forward_theorique(spot, pd.Series([0.02]), pd.Series([0.04]), 1.0)
    assert f.iloc[0] == pytest.approx(1.10 * 1.04 / 1.02)
    # Pas d'arbitrage : 1 EUR placé = 1 EUR converti, placé en USD, reconverti au forward
    assert 1.02 == pytest.approx(1.10 * 1.04 / f.iloc[0])


def test_prime_terme_proche_du_differentiel():
    spot = pd.Series([1.0])
    f = forward_theorique(spot, pd.Series([0.01]), pd.Series([0.05]), 2.0)
    assert prime_terme_annualisee(spot, f, 2.0).iloc[0] == pytest.approx(1.05 / 1.01 - 1)


def test_decaler_mois():
    serie = pd.Series(np.arange(24.0), index=DATES)
    decalee = decaler_mois(serie, 12)
    assert decalee.iloc[:12].isna().all()
    assert decalee.loc["2021-01-01"] == serie.loc["2020-01-01"]


def test_fair_value_uip_taux_egaux_donne_le_spot_retarde():
    spot = pd.Series(np.linspace(1.0, 1.23, 24), index=DATES)
    i = pd.Series(0.02, index=DATES)
    fv = fair_value_uip(spot, i, i, 12)
    assert fv.loc["2021-06-01"] == pytest.approx(spot.loc["2020-06-01"])


def test_pourcent_vers_decimal():
    assert pourcent_vers_decimal(pd.Series([3.75])).iloc[0] == pytest.approx(0.0375)
