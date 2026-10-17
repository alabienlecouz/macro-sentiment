"""Enchaînement : communiqués -> phrases -> probabilités -> score par banque, et comparaison des modèles."""

import hashlib
import json

import pandas as pd

from fxmacro.config import RACINE_PROJET, chemin_projet
from fxmacro.donnees.cache import ecrire_parquet_atomique
from fxmacro.sentiment.agregation import etiquette_predite, score_document, score_phrase
from fxmacro.sentiment.evaluation import matrice_confusion, metriques
from fxmacro.sentiment.textes import charger_communiques, decouper_phrases


def scorer(phrases: list[str], modele: str, config: dict) -> pd.DataFrame:
    """Probabilités p_hawkish / p_dovish / p_neutre, dans l'ordre des phrases reçues.

    Les phrases sont passées au modèle triées par longueur : chaque lot regroupe des
    phrases de taille voisine, donc presque pas de remplissage (padding) inutile.
    """
    ordre = sorted(range(len(phrases)), key=lambda i: len(phrases[i]))
    probas = _scorer_moteur([phrases[i] for i in ordre], modele, config)
    probas.index = ordre
    return probas.sort_index()


def _empreinte(phrases: list[str], modele: str, config: dict) -> str:
    """Identifie un calcul : mêmes phrases, même modèle, mêmes hypothèses -> même résultat."""
    params = config["sentiment"]
    reglages = ({"nom": params["modele_zero_shot"], "hypotheses": params["hypotheses"]} if modele == "zero_shot"
                else {"nom": params["modele_finbert"]})
    contenu = json.dumps({"modele": modele, **reglages, "phrases": phrases}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(contenu.encode("utf-8")).hexdigest()[:20]


def scorer_avec_cache(phrases: list[str], modele: str, config: dict) -> pd.DataFrame:
    """Comme `scorer`, mais un communiqué déjà analysé n'est jamais re-passé dans le modèle.

    Les communiqués changent quelques fois par trimestre alors que la mise à jour tourne
    deux fois par jour : sans ce cache, la VM refait l'inférence DeBERTa pour rien.
    """
    dossier = chemin_projet(f"{config['general']['dossier_cache']}/sentiment")
    chemin = dossier / f"{modele}_{_empreinte(phrases, modele, config)}.parquet"
    if chemin.exists():
        return pd.read_parquet(chemin)
    probas = scorer(phrases, modele, config)
    ecrire_parquet_atomique(probas, chemin)
    return probas


def _scorer_moteur(phrases: list[str], modele: str, config: dict) -> pd.DataFrame:
    """Appel du moteur choisi. Les imports sont faits ici : sur la VM, torch n'est pas
    installé et seul le moteur ONNX est utilisé.
    """
    params = config["sentiment"]
    if modele == "zero_shot" and params["moteur_zero_shot"] == "onnx":
        from fxmacro.sentiment.zero_shot_onnx import scorer_zero_shot_onnx
        return scorer_zero_shot_onnx(phrases, params["hypotheses"], str(RACINE_PROJET / params["dossier_onnx"]),
                                     params["taille_lot"])
    if modele == "zero_shot":
        from fxmacro.sentiment.zero_shot import scorer_zero_shot
        return scorer_zero_shot(phrases, params["hypotheses"], params["modele_zero_shot"], params["taille_lot"])
    if modele == "finbert":
        from fxmacro.sentiment.finbert import scorer_finbert
        return scorer_finbert(phrases, params["modele_finbert"], params["taille_lot"])
    raise ValueError(f"Modèle inconnu : {modele}")


def evaluer_modeles(config: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Compare les modèles sur les phrases annotées : métriques, matrices de confusion, détail."""
    annotees = pd.read_csv(RACINE_PROJET / config["sentiment"]["phrases_etiquetees"])
    detail = annotees.copy()
    lignes, confusions = {}, {}
    for modele in config["sentiment"]["modeles_actifs"]:
        predites = etiquette_predite(scorer(annotees["phrase"].tolist(), modele, config))
        detail[modele] = predites
        lignes[modele] = metriques(annotees["etiquette"], predites)
        confusions[modele] = matrice_confusion(annotees["etiquette"], predites)
    return pd.DataFrame(lignes).T, confusions, detail


def scorer_communiques(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score du dernier communiqué de chaque banque, par modèle, et détail par phrase."""
    params = config["sentiment"]
    communiques = charger_communiques(RACINE_PROJET / params["dossier_communiques"])
    derniers = communiques.sort_values("date").groupby("banque").tail(1)

    resume, details = [], []
    for _, doc in derniers.iterrows():
        phrases = decouper_phrases(doc["texte"])
        ligne = {"banque": doc["banque"], "devise": params["banques"].get(doc["banque"]),
                 "date": doc["date"].date(), "nb_phrases": len(phrases)}
        detail = pd.DataFrame({"banque": doc["banque"], "phrase": phrases})
        for modele in params["modeles_actifs"]:
            probas = scorer_avec_cache(phrases, modele, config)
            agrege = score_document(probas)
            ligne[f"score_{modele}"] = agrege["score"]
            ligne[f"part_hawkish_{modele}"] = agrege["part_hawkish"]
            ligne[f"part_dovish_{modele}"] = agrege["part_dovish"]
            detail[f"score_{modele}"] = score_phrase(probas).to_numpy()
            detail[f"etiquette_{modele}"] = etiquette_predite(probas).to_numpy()
        resume.append(ligne)
        details.append(detail)
    return pd.DataFrame(resume).set_index("banque"), pd.concat(details, ignore_index=True)
