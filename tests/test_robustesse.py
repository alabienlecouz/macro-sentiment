"""Tests des correctifs de la revue : secrets, cache, cours en direct, sentiment."""

import os
import time
import traceback

import pandas as pd
import pytest
import requests

from fxmacro.donnees import cache, http, spots_direct
from fxmacro.sentiment import analyse, textes

CLE = "CLE_SECRETE_123"


# ---------- Secrets ----------

def test_masquer_secrets():
    assert http.masquer_secrets(f"https://x.org/a?series_id=CPI&api_key={CLE}&f=json") == \
        "https://x.org/a?series_id=CPI&api_key=***&f=json"


def test_erreur_http_sans_cle_ni_trace(monkeypatch):
    def faux_get(url, params=None, **_):
        reponse = requests.Response()
        reponse.status_code, reponse.reason = 400, "Bad Request"
        reponse.url = requests.Request("GET", url, params=params).prepare().url
        reponse._content, reponse._content_consumed = b"", True
        return reponse
    monkeypatch.setattr(http.requests, "get", faux_get)
    with pytest.raises(http.ErreurTelechargement) as info:
        http.telecharger("https://api.stlouisfed.org/fred", params={"api_key": CLE})
    trace = "".join(traceback.format_exception(info.value))
    assert CLE not in trace and "api_key=***" in trace


def test_http_refuse_non_https():
    with pytest.raises(ValueError):
        http.telecharger("http://example.org")


# ---------- Cache ----------

def _config(dossier, anciennete_max=30):
    return {"general": {"dossier_cache": str(dossier), "duree_cache_jours": 0,
                        "anciennete_max_cache_jours": anciennete_max}}


def _en_panne():
    raise http.ErreurTelechargement("hors ligne")


def test_cache_perime_reutilise_si_reseau_en_panne(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "chemin_projet", lambda relatif: tmp_path)
    ancienne = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2, freq="MS"), name="s")
    cache.ecrire_cache(tmp_path, "s", ancienne)  # durée 0 jour : déjà périmé

    def en_panne():
        raise http.ErreurTelechargement("hors ligne")
    with pytest.warns(UserWarning, match="réutilisé"):
        serie = cache.obtenir_avec_cache("s", en_panne, _config(tmp_path))
    pd.testing.assert_series_equal(serie, ancienne, check_freq=False)


def test_cache_absent_et_reseau_en_panne_leve(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "chemin_projet", lambda relatif: tmp_path)
    with pytest.raises(http.ErreurTelechargement):
        cache.obtenir_avec_cache("absent", lambda: (_ for _ in ()).throw(http.ErreurTelechargement("x")),
                                 _config(tmp_path))


def test_repli_enregistre_la_source(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "chemin_projet", lambda relatif: tmp_path)
    cache.reinitialiser_suivi()
    cache.ecrire_cache(tmp_path, "fred_X", pd.Series([1.0], name="s"))
    with pytest.warns(UserWarning):
        cache.obtenir_avec_cache("fred_X", _en_panne, _config(tmp_path))
    perimees = cache.sources_perimees()
    assert [s["nom"] for s in perimees] == ["fred_X"] and perimees[0]["age_jours"] >= 0


def test_cache_trop_vieux_relance_l_erreur(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "chemin_projet", lambda relatif: tmp_path)
    cache.reinitialiser_suivi()
    cache.ecrire_cache(tmp_path, "vieux", pd.Series([1.0], name="s"))
    il_y_a_40_jours = time.time() - 40 * 86400
    os.utime(cache.chemin_cache(tmp_path, "vieux"), (il_y_a_40_jours, il_y_a_40_jours))
    with pytest.raises(http.ErreurTelechargement):
        cache.obtenir_avec_cache("vieux", _en_panne, _config(tmp_path, anciennete_max=30))
    assert cache.sources_perimees() == []


def test_reinitialiser_suivi_vide_la_liste():
    cache.enregistrer_repli("x", 3.0)
    cache.reinitialiser_suivi()
    assert cache.sources_perimees() == []


def test_ecriture_atomique_ne_laisse_pas_de_temporaire(tmp_path):
    cache.ecrire_cache(tmp_path, "s", pd.Series([1.0], name="s"))
    assert [f.name for f in tmp_path.iterdir()] == ["s.parquet"]


# ---------- Cours en direct ----------

def test_derniers_cours_sans_donnees_ne_plante_pas(monkeypatch):
    monkeypatch.setattr(spots_direct.yf, "download", lambda *a, **k: pd.DataFrame())
    resultat = spots_direct.derniers_cours(["EURUSD=X", "JPY=X"])
    assert resultat.empty and "EURUSD=X" not in resultat.index


def test_derniers_cours_une_seule_requete_par_intervalle(monkeypatch):
    appels = []
    index = pd.date_range("2026-10-16 10:00", periods=3, freq="5min", tz="UTC")
    colonnes = pd.MultiIndex.from_product([["Close"], ["EURUSD=X", "JPY=X"]])
    donnees = pd.DataFrame([[1.10, 150.0], [1.11, 150.5], [1.12, 151.0]], index=index, columns=colonnes)

    def faux_download(tickers, **_):
        appels.append(list(tickers))
        return donnees
    monkeypatch.setattr(spots_direct.yf, "download", faux_download)
    resultat = spots_direct.derniers_cours(["EURUSD=X", "JPY=X"])
    assert appels == [["EURUSD=X", "JPY=X"]]
    assert resultat.loc["JPY=X", "cours"] == 151.0


# ---------- Sentiment ----------

def _faux_moteur(appels):
    def moteur(phrases, modele, config):
        appels.append(list(phrases))
        longueurs = [len(p) for p in phrases]
        return pd.DataFrame({"p_hawkish": longueurs, "p_dovish": 0.0, "p_neutre": 0.0})
    return moteur


def test_scorer_trie_par_longueur_et_restitue_l_ordre(monkeypatch):
    appels = []
    monkeypatch.setattr(analyse, "_scorer_moteur", _faux_moteur(appels))
    phrases = ["moyenne phrase", "x", "une phrase nettement plus longue"]
    probas = analyse.scorer(phrases, "zero_shot", {})
    assert appels[0] == ["x", "moyenne phrase", "une phrase nettement plus longue"]
    assert probas["p_hawkish"].tolist() == [len(p) for p in phrases]


def test_scores_de_communique_mis_en_cache(tmp_path, monkeypatch):
    appels = []
    monkeypatch.setattr(analyse, "_scorer_moteur", _faux_moteur(appels))
    monkeypatch.setattr(analyse, "chemin_projet", lambda relatif: tmp_path)
    config = {"general": {"dossier_cache": "c"},
              "sentiment": {"modele_zero_shot": "m", "hypotheses": {"hawkish": "h"}}}
    premier = analyse.scorer_avec_cache(["a", "bb"], "zero_shot", config)
    second = analyse.scorer_avec_cache(["a", "bb"], "zero_shot", config)
    assert len(appels) == 1
    pd.testing.assert_frame_equal(premier, second)
    config["sentiment"]["hypotheses"] = {"hawkish": "autre"}  # hypothèses changées -> recalcul
    analyse.scorer_avec_cache(["a", "bb"], "zero_shot", config)
    assert len(appels) == 2


def test_sources_yaml_chemin_malveillant_refuse(tmp_path):
    (tmp_path / "sources.yaml").write_text(
        'FED:\n  - {date: "../../../etc/x", url: "https://example.org"}\n"../X":\n  - {date: "2026-01-01", url: "https://example.org"}\n',
        encoding="utf-8")
    erreurs = textes.mettre_a_jour_communiques(tmp_path)
    assert len(erreurs) == 2 and all("refusé" in e for e in erreurs)
    assert list(tmp_path.iterdir()) == [tmp_path / "sources.yaml"]
