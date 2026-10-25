import numpy as np
import pandas as pd
import pytest

from fxmacro.fair_value.ecarts import ecart_log, ecart_pct, zscore_ancre, zscore_glissant


def test_ecart_pct():
    spot = pd.Series([1.10, 0.90])
    fv = pd.Series([1.00, 1.00])
    assert np.allclose(ecart_pct(spot, fv), [0.10, -0.10])


def test_zscore_glissant_valeurs_connues():
    serie = pd.Series([1.0, 2.0, 3.0, 4.0])
    z = zscore_glissant(serie, fenetre=3)
    # Dernière fenêtre [2, 3, 4] : moyenne 3, écart-type 1 -> z = (4 - 3) / 1
    assert z.iloc[-1] == pytest.approx(1.0)
    assert z.iloc[:2].isna().all()


def test_zscore_min_periodes():
    serie = pd.Series([1.0, 3.0, 5.0, 7.0])
    z = zscore_glissant(serie, fenetre=10, min_periodes=2)
    assert z.iloc[0] != z.iloc[0]  # NaN
    assert z.iloc[1] == pytest.approx((3 - 2) / np.std([1, 3], ddof=1))


def test_zscore_ancre_ne_recentre_pas():
    # Un écart constamment négatif reste négatif (le z glissant, lui, tomberait à 0)
    ecart = pd.Series([-0.10, -0.12, -0.08, -0.10])
    z = zscore_ancre(ecart)
    assert (z < 0).all()
    assert z.iloc[1] == pytest.approx(-0.12 / ecart.std())


def test_zscore_ancre_periode():
    ecart = pd.Series([0.1, -0.1, 0.5, -0.5], index=pd.date_range("2020-01-01", periods=4, freq="MS"))
    z = zscore_ancre(ecart, periode=("2020-01-01", "2020-02-01"))
    assert z.iloc[0] == pytest.approx(0.1 / np.std([0.1, -0.1], ddof=1))


def test_ecart_log_symetrique():
    assert ecart_log(pd.Series([2.0]), pd.Series([1.0])).iloc[0] == pytest.approx(-ecart_log(pd.Series([0.5]), pd.Series([1.0])).iloc[0])


def test_z_ancre_utilise_le_sigma_de_l_historique_long():
    from fxmacro.pipeline import ajouter_ecarts
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    tableau = pd.DataFrame({"spot": [1.0, 1.1, 1.2, 1.3], "fv_ppa": [1.0] * 4, "sigma_ppa": [0.2] * 4}, index=index)
    config = {"ecarts": {"fenetre_zscore_mois": 3, "min_periodes_zscore": 2},
              "ppa": {"periode_calibration": [None, None]}, "beer": {"periode_estimation": [None, None]}}
    resultat = ajouter_ecarts(tableau, config)
    assert resultat["z_ppa_ancre"].iloc[-1] == pytest.approx(np.log(1.3) / 0.2)   # σ long, pas celui du tableau
