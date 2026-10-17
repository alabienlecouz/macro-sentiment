"""Tests de la couche régimes macro, sur données synthétiques."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.regime.classification import (
    DESINFLATION, GOLDILOCKS, NEUTRE, REFLATION, REPORTE, STAGFLATION, appliquer_persistance, bruts_avec_zone_neutre,
    classifier, dans_zone_neutre, regime_brut,
)
from fxmacro.regime.indicateurs import construire_indicateurs, cycle_monetaire, reporter_indicateurs

PARAMS = {"horizon_croissance_mois": 3, "horizon_inflation_mois": 6, "cible_inflation": 2.0,
          "horizon_taux_mois": 6, "seuil_taux_pb": 25, "mois_confirmation": 2}
DATES = pd.date_range("2010-01-01", periods=60, freq="MS")
ANCIENNETE = 3
K, MIN_OBS = 0.25, 12


def regimes(confiance, cpi, taux):
    return classifier(construire_indicateurs(confiance, cpi, taux, PARAMS, ANCIENNETE),
                      PARAMS["mois_confirmation"], K, MIN_OBS)


# ---------- Règles de classification ----------

@pytest.mark.parametrize("croissance, inflation, attendu", [
    (0.5, -0.2, GOLDILOCKS), (0.5, 0.0, GOLDILOCKS),       # inflation stable : ≤ 0
    (0.5, 0.3, REFLATION),
    (-0.5, 0.3, STAGFLATION), (0.0, 0.3, STAGFLATION),     # croissance stable : ≤ 0
    (-0.5, -0.3, DESINFLATION), (0.0, 0.0, DESINFLATION),
])
def test_regles_de_classification(croissance, inflation, attendu):
    assert regime_brut(croissance, inflation) == attendu


def test_donnee_manquante_donne_nan():
    assert pd.isna(regime_brut(np.nan, 0.3)) and pd.isna(regime_brut(0.3, np.nan))


# ---------- Persistance ----------

def _persistance(bruts):
    return appliquer_persistance(pd.Series(bruts, index=DATES[:len(bruts)], dtype=object), 2)


def test_changement_d_un_seul_mois_ignore():
    r = _persistance([GOLDILOCKS, GOLDILOCKS, REFLATION, GOLDILOCKS, GOLDILOCKS])
    assert pd.isna(r["regime"].iloc[0])                 # premier régime : lui aussi à confirmer
    assert r["regime"].iloc[1:].tolist() == [GOLDILOCKS] * 4


def test_changement_de_deux_mois_retenu():
    r = _persistance([GOLDILOCKS, GOLDILOCKS, REFLATION, REFLATION, REFLATION])
    assert r["regime"].tolist()[1:] == [GOLDILOCKS, GOLDILOCKS, REFLATION, REFLATION]


def test_mois_dans_regime_repart_a_un():
    r = _persistance([GOLDILOCKS, GOLDILOCKS, GOLDILOCKS, REFLATION, REFLATION, REFLATION])
    assert r["mois_dans_regime"].tolist()[1:] == [1, 2, 3, 1, 2]


def test_trou_de_donnees_remet_a_zero():
    r = _persistance([GOLDILOCKS, GOLDILOCKS, np.nan, GOLDILOCKS, GOLDILOCKS])
    assert pd.isna(r["regime"].iloc[2]) and pd.isna(r["regime"].iloc[3])
    assert r["regime"].iloc[4] == GOLDILOCKS


# ---------- Cycle monétaire ----------

@pytest.mark.parametrize("hausse_pct, attendu", [(0.50, "resserrement"), (0.10, "pause"), (-0.50, "assouplissement")])
def test_cycle_monetaire(hausse_pct, attendu):
    taux = pd.Series(np.r_[np.full(6, 2.0), 2.0 + hausse_pct], index=DATES[:7])
    cycle = cycle_monetaire(taux, horizon_mois=6, seuil_pb=25)
    assert cycle.iloc[-1] == attendu
    assert cycle.iloc[:6].isna().all()


# ---------- Données complètes ----------

def donnees(graine=0):
    rng = np.random.default_rng(graine)
    confiance = pd.Series(100 + np.cumsum(rng.normal(0, 0.3, 60)), index=DATES)
    cpi = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.002, 0.003, 60))), index=DATES)
    taux = pd.Series(2 + np.cumsum(rng.choice([0, 0, 0, 0.25, -0.25], 60)), index=DATES)
    return confiance, cpi, taux


def test_pas_d_information_future():
    confiance, cpi, taux = donnees()
    t = DATES[40]
    avant = regimes(confiance, cpi, taux)

    apres = DATES > t
    confiance2, cpi2, taux2 = confiance.copy(), cpi.copy(), taux.copy()
    confiance2[apres] += 10
    cpi2[apres] *= 1.2
    taux2[apres] += 3
    modifie = regimes(confiance2, cpi2, taux2)

    colonnes = ["regime_brut", "regime", "mois_dans_regime", "inflation_dynamique", "cycle_monetaire"]
    pd.testing.assert_frame_equal(avant.loc[:t, colonnes], modifie.loc[:t, colonnes])


def test_donnees_manquantes_sans_exception():
    confiance, cpi, taux = donnees(1)
    confiance = confiance.iloc[:45]                      # le BCI s'arrête en cours de route
    resultat = regimes(confiance, cpi, taux)
    # Dernière valeur reprise pendant ANCIENNETE mois, puis données manquantes : régime NaN
    assert resultat["croissance_niveau"].loc[DATES[45]:DATES[44 + ANCIENNETE]].notna().all()
    assert resultat.loc[DATES[45 + ANCIENNETE]:, "regime"].isna().all()
    assert resultat.loc[:DATES[44], "regime"].notna().any()


# ---------- Report des dernières valeurs connues ----------

def test_report_limite_a_l_anciennete_maximale():
    index = pd.date_range("2020-01-01", periods=10, freq="MS")
    valeurs = [1.0, 2.0, np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    bruts = pd.DataFrame({"croissance_dynamique": valeurs, "inflation_dynamique": valeurs}, index=index)
    r = reporter_indicateurs(bruts, anciennete_max_mois=3)
    assert r["croissance_dynamique"].iloc[2:4].tolist() == [2.0, 2.0]      # trou de 2 mois : repris
    assert r["croissance_dynamique"].iloc[5:8].tolist() == [5.0, 5.0, 5.0]  # 3 mois au plus
    assert r["croissance_dynamique"].iloc[8:].isna().all()                   # au-delà : NaN
    assert r["reporte"].tolist() == [False, False, True, True, False, True, True, True, False, False]


def test_mois_reporte_n_invente_pas_d_inflation():
    # Le CPI s'arrête : l'inflation du mois suivant doit être la dernière calculée, pas
    # un ratio entre un CPI ancien et une base de 12 mois plus récente
    confiance, cpi, taux = donnees(2)
    cpi = cpi.iloc[:40]
    ind = construire_indicateurs(confiance, cpi, taux, PARAMS, ANCIENNETE)
    assert ind.loc[DATES[40], "inflation_niveau"] == ind.loc[DATES[39], "inflation_niveau"]
    assert ind.loc[DATES[40], "reporte"] and not ind.loc[DATES[39], "reporte"]
    assert ind.loc[DATES[40 + ANCIENNETE]:, "inflation_niveau"].isna().all()


def test_changement_non_confirme_par_un_mois_reporte():
    bruts = pd.Series([GOLDILOCKS, GOLDILOCKS, REFLATION, REPORTE, REPORTE], index=DATES[:5], dtype=object)
    r = appliquer_persistance(bruts, 2)
    assert r["regime"].tolist()[1:] == [GOLDILOCKS] * 4


def test_mois_reporte_classe_reporte_dans_classifier():
    index = DATES[:30]
    rng = np.random.default_rng(5)
    ind = pd.DataFrame({"croissance_dynamique": rng.normal(0, 1, 30), "inflation_dynamique": rng.normal(0, 1, 30),
                        "reporte": [False] * 29 + [True]}, index=index)
    r = classifier(ind, 2, 0.25, 12)
    assert r["regime_brut"].iloc[-1] == REPORTE
    assert r["regime"].iloc[-1] == r["regime"].iloc[-2] or pd.isna(r["regime"].iloc[-2])


# ---------- Zone neutre ----------

def test_zone_neutre_ne_declenche_pas_de_changement():
    bruts = pd.Series([GOLDILOCKS, GOLDILOCKS, NEUTRE, NEUTRE, NEUTRE], index=DATES[:5], dtype=object)
    r = appliquer_persistance(bruts, 2)
    assert r["regime"].tolist()[1:] == [GOLDILOCKS] * 4
    assert r["mois_dans_regime"].tolist()[1:] == [1, 2, 3, 4]


def test_mois_neutre_interrompt_la_confirmation():
    bruts = pd.Series([GOLDILOCKS, GOLDILOCKS, REFLATION, NEUTRE, REFLATION], index=DATES[:5], dtype=object)
    assert appliquer_persistance(bruts, 2)["regime"].iloc[-1] == GOLDILOCKS


def test_petite_dynamique_classee_neutre():
    rng = np.random.default_rng(3)
    croissance = pd.Series(rng.normal(0, 1, 40), index=DATES[:40])
    inflation = pd.Series(rng.normal(0, 1, 40), index=DATES[:40])
    croissance.iloc[-1], inflation.iloc[-1] = 0.01, 2.0          # croissance quasi nulle
    bruts = bruts_avec_zone_neutre(croissance, inflation, ecarts_types=0.25, min_obs=12)
    assert bruts.iloc[-1] == NEUTRE
    assert bruts.iloc[:11].isna().all()                          # pas assez d'historique : NaN


def test_ecart_type_de_la_zone_neutre_sans_information_future():
    rng = np.random.default_rng(4)
    dynamique = pd.Series(rng.normal(0, 1, 40), index=DATES[:40])
    t = DATES[25]
    modifiee = dynamique.copy()
    modifiee[modifiee.index > t] *= 50                           # volatilité future énorme
    pd.testing.assert_series_equal(dans_zone_neutre(dynamique, 0.25, 12).loc[:t],
                                   dans_zone_neutre(modifiee, 0.25, 12).loc[:t])
