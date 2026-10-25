"""Tests de la PPA ajustée de la productivité, sur données simulées dont on connaît le vrai modèle."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.donnees.alignement import annuel_vers_mensuel
from fxmacro.fair_value.ppa_productivite import (
    estimer_ppa_productivite, ppa_productivite_temps_reel, productivite_relative,
)

N = 360
DATES = pd.date_range("1980-01-01", periods=N, freq="MS")


def tableau_simule(graine: int = 0, b: float = 0.8) -> pd.DataFrame:
    """q = 0,2 + b·Δy + AR(1) stationnaire, Δy = marche aléatoire (productivité relative)."""
    rng = np.random.default_rng(graine)
    dy = np.cumsum(rng.normal(0.001, 0.01, N))
    bruit = np.zeros(N)
    for t in range(1, N):
        bruit[t] = 0.9 * bruit[t - 1] + rng.normal(0, 0.02)
    cpi_base = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.002, N)))
    cpi_cotation = 100 * np.exp(np.cumsum(rng.normal(0.003, 0.002, N)))
    q = 0.2 + b * dy + bruit
    spot = np.exp(q - np.log(cpi_base) + np.log(cpi_cotation))
    return pd.DataFrame({"spot": spot, "cpi_base": cpi_base, "cpi_cotation": cpi_cotation,
                         "prod_base": 40000 * np.exp(dy), "prod_cotation": np.full(N, 40000.0)}, index=DATES)


def test_productivite_relative():
    assert productivite_relative(pd.Series([2.0]), pd.Series([1.0])).iloc[0] == pytest.approx(np.log(2))


def test_retrouve_le_coefficient_de_productivite():
    _, resultat = estimer_ppa_productivite(tableau_simule())
    assert resultat.coefficients["productivite_relative"] == pytest.approx(0.8, abs=0.1)
    assert resultat.p_value_cointegration < 0.05


def test_fair_value_temps_reel_sans_information_future():
    tableau = tableau_simule(1)
    t = DATES[200]
    avant = ppa_productivite_temps_reel(tableau, "1990-01-01")
    modifie = tableau.copy()
    modifie.loc[modifie.index > t, "spot"] *= 1.5
    modifie.loc[modifie.index > t, "prod_base"] *= 1.2
    apres = ppa_productivite_temps_reel(modifie, "1990-01-01")
    assert apres.loc[t, "ecart"] == pytest.approx(avant.loc[t, "ecart"])
    assert apres.iloc[-1]["ecart"] != pytest.approx(avant.iloc[-1]["ecart"])


def test_annuel_vers_mensuel_respecte_la_publication():
    annuel = pd.Series([100.0, 110.0], index=pd.to_datetime(["2020-01-01", "2021-01-01"]))
    mensuel = annuel_vers_mensuel(annuel, decalage_apres_annee=6)
    # Année 2020 utilisable 6 mois après sa fin : juillet 2021, puis reportée 12 mois
    assert mensuel.index[0] == pd.Timestamp("2021-07-01")
    assert mensuel.loc["2022-06-01"] == 100.0 and mensuel.loc["2022-07-01"] == 110.0
    assert mensuel.index[-1] == pd.Timestamp("2023-06-01")
