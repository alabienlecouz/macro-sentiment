"""Assemblage des briques : données -> fair values -> écarts, pour une paire."""

import pandas as pd

from fxmacro.donnees.alignement import aligner_mensuel, journalier_vers_mensuel
from fxmacro.donnees.cpi import obtenir_cpi
from fxmacro.donnees.fondamentaux import obtenir_balance_courante, obtenir_termes_echange
from fxmacro.donnees.spots import obtenir_spot
from fxmacro.donnees.taux import obtenir_taux
from fxmacro.fair_value.beer import (
    ResultatBEER, construire_regresseurs, estimer_beer, fair_value_beer, inflation_publiee, taux_change_reel,
)
from fxmacro.fair_value.ecarts import ecart_log, ecart_pct, zscore_ancre, zscore_glissant
from fxmacro.fair_value.parite_taux import (
    fair_value_uip, forward_theorique, pourcent_vers_decimal, prime_terme_annualisee,
)
from fxmacro.fair_value.ppa import fair_value_ppa

# Modèles dont on calcule écart et z-scores (suffixe de colonne -> colonne de fair value)
MODELES = {"ppa": "fv_ppa", "uip": "fv_uip", "beer": "fv_beer"}


def periode_reference(modele: str, config: dict) -> tuple[str | None, str | None]:
    """Période sur laquelle on mesure σ du z-score ancré : celle où le modèle est calibré.

    PPA (et UIP, qui n'a pas de calibrage) : periode_calibration ; BEER : periode_estimation.
    Seule source de vérité, utilisée ici et par l'état quotidien des signaux.
    """
    if modele == "beer":
        return tuple(config["beer"]["periode_estimation"])
    return tuple(config["ppa"]["periode_calibration"])


def charger_donnees_paire(paire: str, config: dict) -> pd.DataFrame:
    """Spot mensuel, CPI, inflation, taux directeurs et fondamentaux BEER, alignés et sans
    information future.

    aligner_mensuel reporte la dernière valeur connue de chaque colonne. C'est juste pour un
    niveau de prix (PPA), mais pas pour une inflation calculée sur un CPI reporté : on calcule
    donc l'inflation AVANT l'alignement, sur le CPI publié, et c'est elle qui est reportée.
    Elle part du premier mois de spot, comme avant : l'échantillon du BEER est inchangé.
    """
    infos = config["paires"][paire]
    params_align = config["alignement"]
    spot = journalier_vers_mensuel(obtenir_spot(paire, config), params_align["methode_mensuelle"])
    cpi_base, cpi_cotation = obtenir_cpi(infos["base"], config), obtenir_cpi(infos["cotation"], config)
    debut = spot.index[0]
    return aligner_mensuel(
        {"spot": spot,
         "cpi_base": cpi_base,
         "cpi_cotation": cpi_cotation,
         "inflation_base": inflation_publiee(cpi_base.loc[debut:]),
         "inflation_cotation": inflation_publiee(cpi_cotation.loc[debut:]),
         "taux_base": obtenir_taux(infos["base"], config),
         "taux_cotation": obtenir_taux(infos["cotation"], config),
         "tde_base": obtenir_termes_echange(infos["base"], config),
         "tde_cotation": obtenir_termes_echange(infos["cotation"], config),
         "bc_base": obtenir_balance_courante(infos["base"], config),
         "bc_cotation": obtenir_balance_courante(infos["cotation"], config)},
        combler_max=params_align["anciennete_max_mois"], reference="spot",
    )


def ajouter_ppa(tableau: pd.DataFrame, config: dict) -> pd.DataFrame:
    params = config["ppa"]
    tableau["fv_ppa"] = fair_value_ppa(
        tableau["spot"], tableau["cpi_base"], tableau["cpi_cotation"],
        methode=params["methode"], periode=tuple(params["periode_calibration"]),
        date_base=params["date_base"],
    )
    return tableau


def ajouter_parite_taux(tableau: pd.DataFrame, config: dict) -> pd.DataFrame:
    horizon_mois = config["parite_taux"]["horizon_mois"]
    i_base = pourcent_vers_decimal(tableau["taux_base"])
    i_cotation = pourcent_vers_decimal(tableau["taux_cotation"])
    tableau["forward_cip"] = forward_theorique(tableau["spot"], i_base, i_cotation, horizon_mois / 12)
    # Portage annualisé d'une position longue sur la devise de base : ≈ i_XXX - i_YYY
    tableau["carry_base"] = -prime_terme_annualisee(tableau["spot"], tableau["forward_cip"], horizon_mois / 12)
    tableau["fv_uip"] = fair_value_uip(tableau["spot"], i_base, i_cotation, horizon_mois)
    return tableau


def ajouter_beer(tableau: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, ResultatBEER]:
    params = config["beer"]
    q = taux_change_reel(tableau["spot"], tableau["cpi_base"], tableau["cpi_cotation"])
    regresseurs = construire_regresseurs(tableau)[params["variables"]]
    resultat = estimer_beer(q, regresseurs, tuple(params["periode_estimation"]))
    tableau["fv_beer"] = fair_value_beer(resultat.q_estime, tableau["cpi_base"], tableau["cpi_cotation"])
    return tableau, resultat


def ajouter_ecarts(tableau: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Pour chaque modèle : écart en %, z-score glissant et z-score ancré."""
    params = config["ecarts"]
    for suffixe, colonne_fv in MODELES.items():
        tableau[f"ecart_{suffixe}"] = ecart_pct(tableau["spot"], tableau[colonne_fv])
        tableau[f"z_{suffixe}_glissant"] = zscore_glissant(
            tableau[f"ecart_{suffixe}"], params["fenetre_zscore_mois"], params["min_periodes_zscore"])
        tableau[f"z_{suffixe}_ancre"] = zscore_ancre(ecart_log(tableau["spot"], tableau[colonne_fv]),
                                                    periode_reference(suffixe, config))
    return tableau


def construire_tableau(paire: str, config: dict) -> tuple[pd.DataFrame, ResultatBEER]:
    """Tableau mensuel complet d'une paire (tous modèles) et résultat de l'estimation BEER."""
    tableau = charger_donnees_paire(paire, config)
    tableau = ajouter_ppa(tableau, config)
    tableau = ajouter_parite_taux(tableau, config)
    tableau, resultat_beer = ajouter_beer(tableau, config)
    return ajouter_ecarts(tableau, config), resultat_beer


def resumer_derniere_ligne(paire: str, tableau: pd.DataFrame, config: dict) -> dict:
    """Ligne de synthèse pour le dernier mois disponible."""
    infos = config["paires"][paire]
    derniere = tableau.iloc[-1]
    ligne = {"paire": paire, "mois": tableau.index[-1].strftime("%Y-%m"), "spot": derniere["spot"]}
    for suffixe, colonne_fv in MODELES.items():
        ligne[colonne_fv] = derniere[colonne_fv]
        ligne[f"ecart_{suffixe}_%"] = 100 * derniere[f"ecart_{suffixe}"]
        ligne[f"z_{suffixe}_gliss"] = derniere[f"z_{suffixe}_glissant"]
        ligne[f"z_{suffixe}_ancre"] = derniere[f"z_{suffixe}_ancre"]
    ligne["carry_base_%"] = 100 * derniere["carry_base"]
    ligne["sous_eval_ppa"] = infos["base"] if derniere["ecart_ppa"] < 0 else infos["cotation"]
    return ligne
