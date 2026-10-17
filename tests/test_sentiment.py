"""Tests du module sentiment : traitement de texte et agrégation (rapides), modèles (lents)."""

import pandas as pd
import pytest

from fxmacro.sentiment.agregation import etiquette_predite, score_document
from fxmacro.sentiment.evaluation import matrice_confusion, metriques
from fxmacro.sentiment.textes import decouper_phrases, html_vers_texte, recadrer


def test_decouper_phrases_protege_les_abreviations():
    texte = "The U.S. economy grew strongly this quarter. Inflation in the U.K. remains high. Ok."
    phrases = decouper_phrases(texte, longueur_min=10)
    assert phrases == ["The U.S. economy grew strongly this quarter.", "Inflation in the U.K. remains high."]


def test_recadrer():
    texte = "Menu\nAt its meeting the MPC voted.\nText.\n1: Minutes start here"
    assert recadrer(texte, debut="^At its meeting", fin="^1: ") == "At its meeting the MPC voted.\nText."


def test_html_vers_texte_ignore_navigation():
    html = ("<html><body><nav><p>Home | About us | Contact the central bank today</p></nav>"
            "<main><p>The Committee decided to keep the policy rate unchanged at 2 percent.</p></main></body></html>")
    assert html_vers_texte(html) == "The Committee decided to keep the policy rate unchanged at 2 percent."


def test_score_document_pondere_par_le_signal():
    probas = pd.DataFrame({"p_hawkish": [0.9, 0.1], "p_dovish": [0.05, 0.1], "p_neutre": [0.05, 0.8]})
    resultat = score_document(probas)
    # La phrase neutre (poids 0,2) pèse peu face à la phrase hawkish (poids 0,95)
    assert resultat["score"] == pytest.approx((0.85 * 0.95 + 0.0 * 0.2) / 1.15)
    assert resultat["part_hawkish"] == 0.5


def test_metriques_et_confusion():
    vraies = pd.Series(["hawkish", "dovish", "neutre", "hawkish"])
    predites = pd.Series(["hawkish", "hawkish", "neutre", "hawkish"])
    m = metriques(vraies, predites)
    assert m["exactitude"] == 0.75
    assert m["f1_dovish"] == 0.0
    assert matrice_confusion(vraies, predites).loc["dovish", "hawkish"] == 1


def test_etiquette_predite():
    probas = pd.DataFrame({"p_hawkish": [0.2], "p_dovish": [0.7], "p_neutre": [0.1]})
    assert etiquette_predite(probas).iloc[0] == "dovish"


@pytest.mark.lent
def test_zero_shot_sur_phrases_evidentes():
    from fxmacro.config import charger_config
    from fxmacro.sentiment.zero_shot import scorer_zero_shot
    config = charger_config()["sentiment"]
    probas = scorer_zero_shot(["The central bank raised interest rates by 50 basis points to fight inflation.",
                               "The central bank cut interest rates sharply to support the weak economy."],
                              config["hypotheses"], config["modele_zero_shot"])
    assert list(etiquette_predite(probas)) == ["hawkish", "dovish"]
    assert probas.sum(axis=1).round(6).eq(1).all()


@pytest.mark.lent
def test_onnx_donne_les_memes_probabilites_que_torch():
    from fxmacro.config import RACINE_PROJET, charger_config
    from fxmacro.sentiment.zero_shot import scorer_zero_shot
    from fxmacro.sentiment.zero_shot_onnx import scorer_zero_shot_onnx
    config = charger_config()["sentiment"]
    dossier = RACINE_PROJET / config["dossier_onnx"]
    if not (dossier / "model.onnx").exists():
        pytest.skip("Modèle ONNX absent : lancer scripts/exporter_onnx.py")
    phrases = ["Inflation remains well above target.", "The minutes will be published in two weeks.",
               "Further rate cuts are likely."]
    p_torch = scorer_zero_shot(phrases, config["hypotheses"], config["modele_zero_shot"])
    p_onnx = scorer_zero_shot_onnx(phrases, config["hypotheses"], str(dossier))
    assert (p_torch - p_onnx).abs().to_numpy().max() < 1e-3
