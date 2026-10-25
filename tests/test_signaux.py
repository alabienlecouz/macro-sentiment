"""Tests de la combinaison valorisation + sentiment."""

import math

import numpy as np
import pytest

from fxmacro.signaux.combinaison import (
    classer_signal, differentiel_sentiment, zscore_ancre_direct, zscore_valorisation,
)
from fxmacro.signaux.etat import evaluer_signal

SEUILS = {"seuil_z": 1.0, "seuil_sentiment": 0.15}


def test_zscore_valorisation_ignore_les_nan():
    poids = {"ppa": 0.5, "beer": 0.5}
    assert zscore_valorisation({"ppa": 2.0, "beer": 0.0}, poids) == pytest.approx(1.0)
    assert zscore_valorisation({"ppa": 2.0, "beer": float("nan")}, poids) == pytest.approx(2.0)
    assert math.isnan(zscore_valorisation({}, poids))


def test_signal_neutre_sous_le_seuil():
    s = classer_signal("EURUSD", "EUR", "USD", 0.5, 0.3, **SEUILS)
    assert s["orientation"] == "neutre"


def test_devise_sous_evaluee_et_banque_hawkish_donne_signal_fort():
    # EUR sous-évalué (z < 0) et BCE plus hawkish que la Fed (Δs > 0) -> hausse d'EURUSD, forte
    s = classer_signal("EURUSD", "EUR", "USD", -1.5, 0.3, **SEUILS)
    assert (s["orientation"], s["force"]) == ("hausse", "forte")


def test_sentiment_contraire_affaiblit_le_signal():
    s = classer_signal("EURUSD", "EUR", "USD", -1.5, -0.3, **SEUILS)
    assert s["force"] == "faible"


def test_devise_surevaluee():
    # USD surévalué face au JPY dans USDJPY ; BoJ plus hawkish (Δs = s_USD - s_JPY < 0) -> confirmation
    s = classer_signal("USDJPY", "USD", "JPY", 2.2, -0.2, **SEUILS)
    assert (s["orientation"], s["force"]) == ("baisse", "forte")
    assert s["explication"].startswith("JPY sous-évalué face à USD")


def test_sentiment_manquant():
    assert math.isnan(differentiel_sentiment(None, 0.2))
    s = classer_signal("EURUSD", "EUR", "USD", -1.5, float("nan"), **SEUILS)
    assert s["force"] == "modérée"


def test_zscore_ancre_direct():
    assert zscore_ancre_direct(1.1, 1.0, 0.1) == pytest.approx(np.log(1.1) / 0.1)


def test_evaluer_signal_bout_en_bout():
    config = {"signal": {"poids_modeles": {"ppa": 1.0}, **SEUILS}}
    ligne = {"paire": "EURUSD", "base": "EUR", "cotation": "USD", "fv_ppa": 1.25, "sigma_ppa": 0.08,
             "sentiment_base": 0.3, "sentiment_cotation": 0.1}
    r = evaluer_signal(ligne, 1.10, config)
    assert r["ecart_ppa_%"] == pytest.approx(100 * (1.10 / 1.25 - 1))
    assert r["orientation"] == "hausse" and r["force"] == "forte"


# ---------- BEER pris en compte seulement s'il est cointégré ----------

CONFIG_DEUX_MODELES = {"signal": {"poids_modeles": {"ppa": 0.5, "beer": 0.5}, "seuil_z": 1.0,
                                  "seuil_sentiment": 0.15, "p_max_cointegration": 0.10}}


def _ligne(p_coint):
    # Spot 1,00 : z PPA = ln(1 / 1,10) / 0,05 ≈ -1,91 ; z BEER = ln(1 / 0,95) / 0,05 ≈ +1,03
    return {"paire": "EURUSD", "base": "EUR", "cotation": "USD", "fv_ppa": 1.10, "sigma_ppa": 0.05,
            "fv_beer": 0.95, "sigma_beer": 0.05, "p_coint_beer": p_coint,
            "sentiment_base": 0.0, "sentiment_cotation": 0.0}


@pytest.mark.parametrize("p_coint", [0.23, float("nan")])
def test_beer_exclu_si_non_cointegre_ou_inconnu(p_coint):
    r = evaluer_signal(_ligne(p_coint), 1.0, CONFIG_DEUX_MODELES)
    assert r["beer_exclu"] is True
    assert r["z_valorisation"] == pytest.approx(r["z_ppa"])          # PPA seule
    assert r["z_beer"] == pytest.approx(np.log(1 / 0.95) / 0.05)      # valeur affichée conservée
    assert r["orientation"] == "hausse"


def test_beer_inclus_si_cointegre():
    r = evaluer_signal(_ligne(0.06), 1.0, CONFIG_DEUX_MODELES)
    assert r["beer_exclu"] is False
    assert r["z_valorisation"] == pytest.approx((r["z_ppa"] + r["z_beer"]) / 2)
    assert r["orientation"] == "neutre"   # (-1,91 + 1,03) / 2 ≈ -0,44


def test_modele_affiche_ne_change_pas_le_signal():
    config = {"signal": {**CONFIG_DEUX_MODELES["signal"], "modeles_affichage": ["ppa_productivite"]}}
    ligne = {**_ligne(0.06), "fv_ppa_productivite": 0.80, "sigma_ppa_productivite": 0.05}
    avec = evaluer_signal(ligne, 1.0, config)
    sans = evaluer_signal(_ligne(0.06), 1.0, CONFIG_DEUX_MODELES)
    assert avec["z_valorisation"] == pytest.approx(sans["z_valorisation"])
    assert avec["orientation"] == sans["orientation"]
    assert avec["z_ppa_productivite"] == pytest.approx(np.log(1 / 0.80) / 0.05)   # bien calculé et affiché
    assert avec["ecart_ppa_productivite_%"] == pytest.approx(100 * (1 / 0.80 - 1))
