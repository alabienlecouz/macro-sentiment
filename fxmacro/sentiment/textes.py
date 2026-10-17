"""Récupération des communiqués de banques centrales et découpage en phrases."""

import io
import re
from pathlib import Path

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader

from fxmacro.donnees.http import telecharger

ENTETES = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Lignes de navigation de sites web à ignorer
BRUIT = re.compile(r"official website|\.gov website|Stay Connected|cookies|Share this page", re.I)

# sources.yaml sert à construire des chemins de fichiers : on n'accepte que des noms sûrs
NOM_BANQUE = re.compile(r"^[A-Z]{2,6}$")
DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Abréviations courantes qui ne terminent pas une phrase
ABREVIATIONS = ["U.S.", "U.K.", "e.g.", "i.e.", "p.a.", "Mr.", "Ms.", "No.", "approx.", "vs."]


def html_vers_texte(html: str | bytes) -> str:
    """Texte des paragraphes d'une page HTML (ignore menus, scripts, pieds de page)."""
    soupe = BeautifulSoup(html, "html.parser")
    for balise in soupe(["script", "style", "nav", "header", "footer", "aside"]):
        balise.decompose()
    zone = soupe.find("main") or soupe.find("article") or soupe.body or soupe
    paragraphes = [p.get_text(" ", strip=True) for p in zone.find_all(["p", "li"])]
    return "\n".join(p for p in paragraphes if len(p) > 40 and not BRUIT.search(p))


def pdf_vers_texte(contenu: bytes) -> str:
    lecteur = PdfReader(io.BytesIO(contenu))
    return "\n".join(page.extract_text() or "" for page in lecteur.pages)


def recadrer(texte: str, debut: str | None = None, fin: str | None = None) -> str:
    """Garde le texte entre deux motifs (expressions régulières, début de ligne avec ^).

    Ex. : la page de la BoE contient le résumé puis les minutes ; on s'arrête aux minutes.
    """
    if debut:
        trouve = re.search(debut, texte, re.M)
        if trouve:
            texte = texte[trouve.start():]
    if fin:
        trouve = re.search(fin, texte, re.M)
        if trouve:
            texte = texte[:trouve.start()]
    return texte.strip()


def telecharger_communique(url: str, debut: str | None = None, fin: str | None = None) -> str:
    """Texte brut d'un communiqué (HTML ou PDF), éventuellement recadré."""
    reponse = telecharger(url, headers=ENTETES, timeout=60, taille_max=20 * 1024 * 1024)
    if url.lower().endswith(".pdf") or "pdf" in reponse.type_contenu:
        texte = pdf_vers_texte(reponse.contenu)
    else:
        # On laisse BeautifulSoup détecter l'encodage à partir des octets (évite « â » à la place de « – »)
        texte = html_vers_texte(reponse.contenu)
    return recadrer(texte, debut, fin)


def mettre_a_jour_communiques(dossier: Path, forcer: bool = False) -> list[str]:
    """Télécharge les communiqués de sources.yaml absents du dossier. Renvoie les erreurs."""
    sources = yaml.safe_load((dossier / "sources.yaml").read_text(encoding="utf-8"))
    erreurs = []
    for banque, documents in sources.items():
        if not NOM_BANQUE.match(str(banque)):
            erreurs.append(f"nom de banque refusé dans sources.yaml : {banque!r}")
            continue
        for doc in documents:
            if not DATE_ISO.match(str(doc.get("date", ""))):
                erreurs.append(f"{banque} : date refusée dans sources.yaml : {doc.get('date')!r}")
                continue
            chemin = dossier / banque / f"{doc['date']}.txt"
            if chemin.exists() and not forcer:
                continue
            try:
                texte = telecharger_communique(doc["url"], doc.get("debut"), doc.get("fin"))
            except requests.RequestException as erreur:
                erreurs.append(f"{banque} {doc['date']} : {erreur} ({doc['url']})")
                continue
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_text(texte, encoding="utf-8")
    return erreurs


def charger_communiques(dossier: Path) -> pd.DataFrame:
    """Un communiqué par ligne : banque, date, texte (lu depuis les .txt)."""
    lignes = [{"banque": f.parent.name, "date": pd.Timestamp(f.stem), "texte": f.read_text(encoding="utf-8")}
              for f in sorted(dossier.glob("*/*.txt"))]
    return pd.DataFrame(lignes, columns=["banque", "date", "texte"])


def decouper_phrases(texte: str, longueur_min: int = 25) -> list[str]:
    """Découpe un texte en phrases (règles simples, sans modèle de langue).

    On protège les abréviations, on coupe après . ! ? suivi d'une majuscule, et on
    écarte les fragments trop courts (titres, dates, numéros de page).
    """
    texte = re.sub(r"\s+", " ", texte)
    for i, abrev in enumerate(ABREVIATIONS):
        texte = texte.replace(abrev, f"§{i}§")
    morceaux = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", texte)
    phrases = []
    for morceau in morceaux:
        for i, abrev in enumerate(ABREVIATIONS):
            morceau = morceau.replace(f"§{i}§", abrev)
        morceau = morceau.strip()
        if len(morceau) >= longueur_min:
            phrases.append(morceau)
    return phrases
