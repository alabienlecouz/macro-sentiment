"""Indicateurs macro mensuels d'une devise, calculés uniquement avec des données publiées.

Les entrées (confiance des entreprises, CPI) sont déjà décalées de leur délai de
publication ; tous les calculs ci-dessous ne regardent que le passé (différences et
décalages vers l'arrière). Le taux directeur est connu le jour même.

Un mois pas encore publié reprend la dernière valeur connue des INDICATEURS (pas des
séries brutes), pendant au plus `anciennete_max_mois` mois. Reporter le niveau du CPI
avant de calculer l'inflation sur 12 mois diviserait un CPI ancien par une base plus
récente : une inflation inventée. Les mois reportés sont signalés (colonne `reporte`)
et la classification ne s'en sert pas pour changer de régime. Au-delà de
`anciennete_max_mois`, la donnée reste manquante.
"""

import numpy as np
import pandas as pd

from fxmacro.fair_value.beer import inflation_glissement_annuel

RESSERREMENT, ASSOUPLISSEMENT, PAUSE = "resserrement", "assouplissement", "pause"


def mensualiser(serie: pd.Series) -> pd.Series:
    """Index mensuel régulier (début de mois) : un mois manquant devient NaN, jamais comblé."""
    return serie.resample("MS").last()


def croissance_niveau(confiance: pd.Series) -> pd.Series:
    """Écart du BCI à sa moyenne de long terme (100) : > 0 = conjoncture au-dessus de la tendance."""
    return confiance - 100


def croissance_dynamique(confiance: pd.Series, horizon_mois: int) -> pd.Series:
    """Variation du BCI sur `horizon_mois` mois : > 0 = la conjoncture s'améliore."""
    return confiance.diff(horizon_mois)


def inflation_niveau(cpi: pd.Series) -> pd.Series:
    """Inflation sur 12 mois, en %."""
    return inflation_glissement_annuel(cpi)


def inflation_dynamique(inflation: pd.Series, horizon_mois: int) -> pd.Series:
    """Variation de l'inflation sur 12 mois, sur `horizon_mois` mois (en points)."""
    return inflation.diff(horizon_mois)


def ecart_cible(inflation: pd.Series, cible: float) -> pd.Series:
    """Inflation moins la cible de la banque centrale (en points)."""
    return inflation - cible


def cycle_monetaire(taux_pct: pd.Series, horizon_mois: int, seuil_pb: float) -> pd.Series:
    """'resserrement' / 'assouplissement' / 'pause' selon la variation du taux sur `horizon_mois`."""
    variation_pb = 100 * taux_pct.diff(horizon_mois)
    cycle = pd.Series(np.where(variation_pb > seuil_pb, RESSERREMENT,
                               np.where(variation_pb < -seuil_pb, ASSOUPLISSEMENT, PAUSE)),
                      index=taux_pct.index, dtype=object)
    return cycle.where(variation_pb.notna())


def reporter_indicateurs(bruts: pd.DataFrame, anciennete_max_mois: int) -> pd.DataFrame:
    """Reprend la dernière valeur calculée de chaque indicateur pendant au plus
    `anciennete_max_mois` mois, et marque `reporte` = vrai si l'une des deux dynamiques
    (croissance, inflation) du mois est une valeur reprise et non calculée."""
    reportes = bruts.ffill(limit=anciennete_max_mois)
    dynamiques = ["croissance_dynamique", "inflation_dynamique"]
    reportes["reporte"] = (bruts[dynamiques].isna() & reportes[dynamiques].notna()).any(axis=1)
    return reportes


def construire_indicateurs(confiance: pd.Series, cpi: pd.Series, taux_pct: pd.Series,
                           params: dict, anciennete_max_mois: int) -> pd.DataFrame:
    """Tableau mensuel des indicateurs d'une devise (séries d'entrée déjà point-in-time).

    Les indicateurs sont calculés sur les séries brutes (mois non publiés = NaN), puis
    reportés : jamais de calcul sur une valeur reprise.
    """
    base = pd.concat({"confiance": mensualiser(confiance), "cpi": mensualiser(cpi),
                      "taux": mensualiser(taux_pct)}, axis=1)
    confiance, cpi, taux_pct = base["confiance"], base["cpi"], base["taux"]
    inflation = inflation_niveau(cpi)
    tableau = pd.DataFrame({
        "croissance_niveau": croissance_niveau(confiance),
        "croissance_dynamique": croissance_dynamique(confiance, params["horizon_croissance_mois"]),
        "inflation_niveau": inflation,
        "inflation_dynamique": inflation_dynamique(inflation, params["horizon_inflation_mois"]),
        "ecart_cible": ecart_cible(inflation, params["cible_inflation"]),
        "cycle_monetaire": cycle_monetaire(taux_pct, params["horizon_taux_mois"], params["seuil_taux_pb"]),
    })
    tableau.index.name = "date"
    return reporter_indicateurs(tableau, anciennete_max_mois)
