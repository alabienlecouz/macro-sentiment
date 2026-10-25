"""Archives des communiqués de politique monétaire (Fed, BCE, BoE), pour le backtest du sentiment.

Pour chaque banque, on liste les réunions passées (date de publication, adresse), puis on
télécharge le texte avec les mêmes fonctions que pour les communiqués actuels.

Textes retenus, choisis pour être comparables d'une époque à l'autre :
- Fed : communiqué du FOMC (pages « monetaryAAAAMMJJa.htm ») ;
- BCE : déclaration introductive de la conférence de presse, sans les questions-réponses
  (avant 2022, le communiqué de décision ne fait que trois phrases) ;
- BoE : « Monetary Policy Summary », publié depuis août 2015.
La date retenue est celle de la publication : le texte n'est pas connu avant.
"""

import re
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup

from fxmacro.donnees.http import telecharger
from fxmacro.sentiment.textes import ENTETES, decouper_phrases, recadrer, telecharger_communique

FED = "https://www.federalreserve.gov"
BCE = "https://www.ecb.europa.eu"
BOE = "https://www.bankofengland.co.uk"

# Où couper chaque texte (mêmes conventions que sources.yaml) :
# BCE : avant les questions-réponses ; BoE : avant les minutes (paragraphes « 1: » ou « 1) »)
RECADRAGE = {"FED": {"fin": "^For media inquiries"},
             "BCE": {"fin": "^(Question:|We are now at your disposal|We are ready to take your questions)"},
             "BOE": {"fin": r"^1[:).] "}}
# Un communiqué du FOMC mentionne toujours le taux directeur (écarte les autres annonces)
MOT_OBLIGATOIRE = {"FED": "federal funds rate"}
PHRASES_MIN = 5  # en dessous : page mal extraite (texte hors des paragraphes HTML)


def nettoyer(banque: str, texte: str) -> str:
    """Recadre un texte téléchargé. BCE : depuis 2022, les questions-réponses suivent la
    déclaration sans titre dans les paragraphes ; on coupe à la première question (ligne avec « ? »)."""
    texte = recadrer(texte, **RECADRAGE[banque])
    if banque == "BCE":
        lignes = texte.split("\n")
        premiere_question = next((i for i, l in enumerate(lignes) if i > 0 and "?" in l), len(lignes))
        texte = "\n".join(lignes[:premiere_question])
    return texte


def retenu(banque: str, texte: str) -> bool:
    """Le texte est-il un communiqué de politique monétaire exploitable ?"""
    mot = MOT_OBLIGATOIRE.get(banque)
    return (mot is None or mot in texte.lower()) and len(decouper_phrases(texte)) >= PHRASES_MIN


def _page(url: str) -> str:
    return telecharger(url, headers=ENTETES, timeout=60).contenu.decode("utf-8", errors="replace")


def dates_dans_liens(html: str, motif: str, format_date: str) -> dict[pd.Timestamp, str]:
    """Liens dont l'adresse contient une date : {date: lien}. `motif` capture (lien, date)."""
    return {pd.to_datetime(d, format=format_date): lien for lien, d in re.findall(motif, html)}


def lister_fed(annees: range) -> dict[pd.Timestamp, str]:
    """Communiqués du FOMC : pages historiques par année, plus le calendrier des 5 dernières années."""
    pages = [f"{FED}/monetarypolicy/fomccalendars.htm"]
    pages += [f"{FED}/monetarypolicy/fomchistorical{a}.htm" for a in annees if a <= date.today().year - 6]
    liens = {}
    for page in pages:
        liens |= dates_dans_liens(_page(page), r'href="(/newsevents/pressreleases/monetary(\d{8})a\.htm)"', "%Y%m%d")
    return {d: FED + l for d, l in liens.items() if d.year in annees}


def lister_bce(annees: range) -> dict[pd.Timestamp, str]:
    """Déclarations de politique monétaire, une page d'index par année.

    Avant 2022 : « is….en.html » (déclaration introductive) ; depuis : « mp….en.html ».
    """
    liens = {}
    for a in annees:
        html = _page(f"{BCE}/press/press_conference/monetary-policy-statement/{a}/html/index_include.en.html")
        liens |= dates_dans_liens(html, r'href="(/press/[^"]*?(?:is|mp)(\d{6})[^"]*?\.en\.html)"', "%y%m%d")
    return {d: BCE + l for d, l in liens.items()}


def lister_boe(annees: range) -> dict[pd.Timestamp, str]:
    """Monetary Policy Summaries d'après le sitemap ; la date exacte est lue sur chaque page."""
    sitemap = _page(f"{BOE}/_api/sitemap/getsitemap")
    adresses = sorted(set(re.findall(r"<loc>(https://www\.bankofengland\.co\.uk/monetary-policy-summary-and-minutes/"
                                     r"(\d{4})/[a-z-]+\d{4})</loc>", sitemap)))
    liens = {}
    for adresse, annee in adresses:
        if int(annee) not in annees:
            continue
        publication = date_publication_boe(_page(adresse))
        if publication is not None:
            liens[publication] = adresse
    return liens


def date_publication_boe(html: str) -> pd.Timestamp | None:
    """Date de l'élément « Published on … » d'une page de la BoE."""
    element = BeautifulSoup(html, "html.parser").find(class_="published-date")
    trouve = re.search(r"\d{1,2} [A-Z][a-z]+ \d{4}", element.get_text(" ", strip=True)) if element else None
    return pd.to_datetime(trouve.group(), format="%d %B %Y") if trouve else None


LISTEURS = {"FED": lister_fed, "BCE": lister_bce, "BOE": lister_boe}


def collecter(banque: str, annees: range, dossier) -> list[str]:
    """Télécharge les communiqués absents du dossier <dossier>/<BANQUE>/<date>.txt. Renvoie les erreurs."""
    erreurs = []
    aujourd_hui = pd.Timestamp.today().normalize()
    for jour, url in sorted(LISTEURS[banque](annees).items()):
        if jour > aujourd_hui:
            continue  # pages préparées à l'avance pour des réunions futures
        chemin = dossier / banque / f"{jour.date()}.txt"
        if chemin.exists():
            continue
        try:
            texte = nettoyer(banque, telecharger_communique(url))
        except Exception as erreur:  # une page manquante ne doit pas arrêter la collecte
            erreurs.append(f"{banque} {jour.date()} : {erreur} ({url})")
            continue
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(texte, encoding="utf-8")
    return erreurs
