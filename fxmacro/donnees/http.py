"""Requêtes HTTP communes à toutes les sources (FRED, BRI, BCE, OCDE, communiqués).

Trois garanties, au même endroit :
- HTTPS uniquement (y compris après redirection) ;
- taille de réponse plafonnée (un PDF ou un CSV géant ne peut pas saturer la RAM de la VM) ;
- aucun secret dans les messages d'erreur : requests met l'URL complète, donc la clé
  FRED (?api_key=...), dans le texte de ses exceptions, qui finissent dans journalctl.
"""

import json
import re
from dataclasses import dataclass

import requests

TAILLE_MAX_OCTETS = 50 * 1024 * 1024  # 50 Mo : très au-delà de toute réponse légitime ici
_PARAMETRES_SECRETS = re.compile(r"(api_key|apikey|token|access_key)=[^&\s'\"]+", re.I)


class ErreurTelechargement(requests.RequestException):
    """Échec réseau ou HTTP, avec un message déjà expurgé des secrets."""


def masquer_secrets(texte: str) -> str:
    """'...?api_key=abc123&x=1' -> '...?api_key=***&x=1'."""
    return _PARAMETRES_SECRETS.sub(r"\1=***", texte)


@dataclass(frozen=True)
class Reponse:
    """Contenu déjà téléchargé et vérifié (la connexion est refermée)."""
    url: str
    contenu: bytes
    type_contenu: str
    encodage: str | None

    @property
    def texte(self) -> str:
        return self.contenu.decode(self.encodage or "utf-8", errors="replace")

    def json(self):
        return json.loads(self.contenu)


def telecharger(url: str, *, params: dict | None = None, headers: dict | None = None,
                timeout: float = 60, taille_max: int = TAILLE_MAX_OCTETS) -> Reponse:
    """GET HTTPS, plafonné en taille ; lève ErreurTelechargement (sans secret) en cas d'échec."""
    if not url.startswith("https://"):
        raise ValueError(f"URL non HTTPS refusée : {masquer_secrets(url)}")
    try:
        with requests.get(url, params=params, headers=headers, timeout=timeout, stream=True) as reponse:
            reponse.raise_for_status()
            if not reponse.url.startswith("https://"):
                raise ErreurTelechargement(f"Redirection hors HTTPS refusée : {masquer_secrets(reponse.url)}")
            morceaux, total = [], 0
            for morceau in reponse.iter_content(chunk_size=64 * 1024):
                total += len(morceau)
                if total > taille_max:
                    raise ErreurTelechargement(
                        f"Réponse de plus de {taille_max // 2**20} Mo : {masquer_secrets(reponse.url)}")
                morceaux.append(morceau)
            return Reponse(url=reponse.url, contenu=b"".join(morceaux),
                           type_contenu=reponse.headers.get("Content-Type", ""), encodage=reponse.encoding)
    except ErreurTelechargement:
        raise
    except requests.RequestException as erreur:
        # « from None » : sinon l'exception d'origine (avec la clé dans l'URL) reste attachée
        # à la trace et est imprimée par traceback.format_exc() dans le journal.
        raise ErreurTelechargement(masquer_secrets(f"{erreur.__class__.__name__}: {erreur}")) from None
