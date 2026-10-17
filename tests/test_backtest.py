"""Tests du backtest en temps réel : pas d'information future, cohérence avec le pipeline."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.fair_value.ecarts import ecart_log, zscore_ancre
from fxmacro.fair_value.ppa import fair_value_ppa
from fxmacro.signaux.backtest import (
    dates_gloutonnes, evaluer, signal_depuis_z, z_beer_temps_reel, z_ppa_temps_reel,
)

N = 200
DATES = pd.date_range("2005-01-01", periods=N, freq="MS")
DEBUT = "2015-01-01"


def tableau_synthetique(graine: int = 0) -> pd.DataFrame:
    """Colonnes minimales du tableau d'une paire, avec des marches aléatoires."""
    rng = np.random.default_rng(graine)
    marche = lambda sigma: np.cumsum(rng.normal(0, sigma, N))
    return pd.DataFrame({
        "spot": 1.2 * np.exp(marche(0.02)),
        "cpi_base": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.002, N))),
        "cpi_cotation": 100 * np.exp(np.cumsum(rng.normal(0.003, 0.002, N))),
        "taux_base": 2 + marche(0.1), "taux_cotation": 3 + marche(0.1),
        "tde_base": 100 * np.exp(marche(0.01)), "tde_cotation": 100 * np.exp(marche(0.01)),
        "bc_base": marche(0.2), "bc_cotation": marche(0.2),
    }, index=DATES)


def test_pas_d_information_future():
    tableau = tableau_synthetique()
    t = pd.Timestamp("2016-06-01")
    z_ppa, z_beer = z_ppa_temps_reel(tableau, DEBUT), z_beer_temps_reel(tableau, DEBUT)

    modifie = tableau.copy()
    apres = modifie.index > t
    modifie.loc[apres, "spot"] *= 2.0
    modifie.loc[apres, "cpi_base"] *= 1.5
    modifie.loc[apres, "tde_base"] *= 0.7

    assert z_ppa_temps_reel(modifie, DEBUT)[t] == pytest.approx(z_ppa[t])
    assert z_beer_temps_reel(modifie, DEBUT).loc[t, "z"] == pytest.approx(z_beer.loc[t, "z"])
    # Contrôle : les dates postérieures, elles, changent bien
    assert z_ppa_temps_reel(modifie, DEBUT).iloc[-1] != pytest.approx(z_ppa.iloc[-1])


def test_derniere_date_egale_au_plein_echantillon():
    tableau = tableau_synthetique(1)
    fv = fair_value_ppa(tableau["spot"], tableau["cpi_base"], tableau["cpi_cotation"], methode="moyenne")
    z_plein = zscore_ancre(ecart_log(tableau["spot"], fv))
    assert z_ppa_temps_reel(tableau, DEBUT).iloc[-1] == pytest.approx(z_plein.iloc[-1])


def test_signal_depuis_z():
    z = pd.Series([-1.5, 1.5, 0.3, -1.0, 1.0, np.nan])
    assert signal_depuis_z(z, 1.0).tolist() == [1, -1, 0, 1, -1, 0]  # |z| == seuil : signal actif


def test_evaluer_cas_construit_a_la_main():
    # log du spot : 0 ; 0,1 ; 0 ; 0,2 ; 0,1 ; 0,3  -> rendements à 2 mois : 0 ; 0,1 ; 0,1 ; 0,1 ; -- ; --
    index = pd.date_range("2020-01-01", periods=6, freq="MS")
    spot = pd.Series(np.exp([0.0, 0.1, 0.0, 0.2, 0.1, 0.3]), index=index)
    signal = pd.Series([1, -1, 0, 1, 0, 0], index=index)
    r = evaluer(signal, spot, horizon=2)
    # Mois évaluables : 4 premiers ; actifs : 3 ; rendements signés : 0 ; -0,1 ; +0,1
    assert r["part_actifs"] == pytest.approx(3 / 4)
    assert r["n"] == 3
    assert r["taux_reussite"] == pytest.approx(1 / 3)
    assert r["rendement_moyen_%"] == pytest.approx(0.0, abs=1e-12)
    # Non chevauchant : janvier puis mars ; seul janvier est actif
    assert r["n_non_chevauchant"] == 1
    assert r["taux_reussite_non_chevauchant"] == 0.0


def test_sous_echantillon_glouton():
    # Actifs aux mois 1, 2, 4, 6 et 7 (horizon 3) : on garde 1, on saute à 4, on garde 4, on saute à 7, on garde 7
    index = pd.date_range("2020-01-01", periods=9, freq="MS")
    signal = pd.Series([0, 1, -1, 0, 1, 0, -1, 1, 0], index=index)
    assert list(dates_gloutonnes(signal, 3)) == [index[1], index[4], index[7]]


def test_evaluer_glouton_cas_construit_a_la_main():
    index = pd.date_range("2020-01-01", periods=6, freq="MS")
    spot = pd.Series(np.exp([0.0, 0.1, 0.0, 0.2, 0.1, 0.3]), index=index)
    signal = pd.Series([1, -1, 0, 1, 0, 0], index=index)
    r = evaluer(signal, spot, horizon=2)
    # Glouton : janvier (actif, r = 0), saut à mars (inactif), avril (actif, r = +0,1)
    assert r["n_glouton"] == 2
    assert r["taux_reussite_glouton"] == pytest.approx(0.5)
    assert r["rendement_moyen_glouton_%"] == pytest.approx(5.0)
