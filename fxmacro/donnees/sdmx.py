"""Lecture des réponses CSV SDMX (OCDE, BCE) et conversion des périodes en dates."""

import io

import pandas as pd


def periode_vers_date(periodes: pd.Series) -> pd.DatetimeIndex:
    """'2024-Q2' -> 2024-04-01 (début du trimestre) ; '2024-05' -> 2024-05-01."""
    texte = periodes.astype(str)
    if texte.str.contains("-Q").any():
        return pd.PeriodIndex(texte.str.replace("-Q", "Q"), freq="Q").to_timestamp(how="start")
    return pd.DatetimeIndex(pd.to_datetime(texte, format="%Y-%m"))


def lire_csv_sdmx(texte: str, colonne_groupe: str | None = None) -> dict[str, pd.Series] | pd.Series:
    """CSV SDMX -> série (ou dictionnaire de séries par valeur de `colonne_groupe`)."""
    tableau = pd.read_csv(io.StringIO(texte))
    tableau["date"] = periode_vers_date(tableau["TIME_PERIOD"])
    tableau["valeur"] = pd.to_numeric(tableau["OBS_VALUE"], errors="coerce")
    if colonne_groupe is None:
        return tableau.set_index("date")["valeur"].sort_index()
    return {cle: groupe.set_index("date")["valeur"].sort_index()
            for cle, groupe in tableau.groupby(colonne_groupe)}
