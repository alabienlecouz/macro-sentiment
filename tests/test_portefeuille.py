"""Tests du portefeuille value + carry sur données simulées."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.signaux.portefeuille import (
    ecart_taux_portefeuille, poids_long_short, prime_de_risque, rendement_portefeuille, rendements_totaux,
    score_combine, statistiques,
)

MOIS = pd.date_range("2000-01-01", periods=4, freq="MS")


def test_rendement_total_ajoute_le_carry_du_mois_precedent():
    cours = pd.DataFrame({"AAA": [1.0, 1.0, 1.1, 1.1]}, index=MOIS)
    taux = pd.DataFrame({"AAA": [12.0, 12.0, 0.0, 0.0]}, index=MOIS)
    taux_usd = pd.Series(0.0, index=MOIS)
    rx = rendements_totaux(cours, taux, taux_usd)["AAA"]
    assert rx.iloc[1] == pytest.approx(0.01)                    # 12 % / 12, change inchangé
    assert rx.iloc[2] == pytest.approx(0.01 + np.log(1.1))      # carry fixé fin du mois précédent
    assert rx.iloc[3] == pytest.approx(0.0)


def test_poids_achete_les_meilleurs_et_vend_les_pires():
    signal = pd.DataFrame([[5, 4, 3, 2, 1, np.nan]], columns=list("ABCDEF"), index=MOIS[:1])
    poids = poids_long_short(signal, nb=2, min_devises=5).iloc[0]
    assert poids["A"] == poids["B"] == 0.5 and poids["D"] == poids["E"] == -0.5
    assert poids["C"] == 0 and poids["F"] == 0
    assert poids_long_short(signal, nb=2, min_devises=6).iloc[0].abs().sum() == 0   # pas assez de devises


def test_les_poids_s_appliquent_au_mois_suivant_et_les_couts_sont_payes():
    poids = pd.DataFrame({"A": [1.0, 1.0, -1.0, -1.0]}, index=MOIS)
    rendements = pd.DataFrame({"A": [0.50, 0.02, 0.03, -0.04]}, index=MOIS)   # 0,50 en janvier : jamais capté
    r = rendement_portefeuille(poids, rendements, cout_pb=10)
    assert r.index[0] == MOIS[1]
    assert r["brut"].tolist() == pytest.approx([0.02, 0.03, 0.04])
    assert r["net"].iloc[2] == pytest.approx(0.04 - 0.001 * 2)               # rotation de +1 à -1 = 2


def test_cout_de_renouvellement_paye_sur_tout_le_nominal():
    poids = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]}, index=MOIS[:3])
    rendements = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=MOIS[:3])
    r = rendement_portefeuille(poids, rendements, cout_pb=10, cout_roll_pb=1)
    assert r["net"].iloc[-1] == pytest.approx(-0.0001)          # aucune rotation, nominal brut = 1 → 1 pb


def test_ecart_taux_entre_devises_achetees_et_vendues():
    poids = pd.DataFrame([[0.5, 0.5, -0.5, -0.5]], columns=list("ABCD"))
    carry = pd.DataFrame([[4.0, 2.0, 1.0, -1.0]], columns=list("ABCD"))
    assert ecart_taux_portefeuille(poids, carry).iloc[0] == pytest.approx(3.0)


def test_score_combine_moyenne_de_signaux_standardises():
    carry = pd.DataFrame([[1.0, 2.0, 3.0]], columns=list("ABC"))
    value = pd.DataFrame([[3.0, 2.0, 1.0]], columns=list("ABC"))
    assert score_combine(carry, value).iloc[0].tolist() == pytest.approx([0, 0, 0])


def test_statistiques_et_prime_de_risque():
    r = pd.Series([0.01, -0.02, 0.01, 0.01] * 30, index=pd.date_range("2000-01-01", periods=120, freq="MS"))
    s = statistiques(r)
    assert s["rendement_annuel_%"] == pytest.approx(3.0)        # 0,25 % par mois × 12
    assert s["pire_mois_%"] == pytest.approx(-2.0)
    actions = pd.Series(np.where(r < 0, -0.05, 0.02), index=r.index, name="actions")
    p = prime_de_risque(r, actions)
    assert p["rendement_mois_baisse_actions_%"] == pytest.approx(-2.0)
    assert p["rendement_mois_hausse_actions_%"] == pytest.approx(1.0)
