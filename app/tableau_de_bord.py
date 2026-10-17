"""Tableau de bord du bot macro FX.

Lancement : streamlit run app/tableau_de_bord.py
Les modèles sont recalculés par scripts/mettre_a_jour.py (une fois par jour) ; cette
page relit leurs résultats et recalcule le signal avec le dernier cours disponible.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from fxmacro.config import RACINE_PROJET, charger_config
from fxmacro.donnees.spots_direct import derniers_cours
from fxmacro.signaux.etat import evaluer_signal

# Palette catégorielle validée (ordre fixe) et pôles divergents dovish / hawkish
COULEURS_SERIES = {"Spot": "#2a78d6", "Fair value PPA": "#eb6834", "Fair value BEER": "#1baf7a"}
COULEURS_Z = {"z PPA": "#eb6834", "z BEER": "#1baf7a"}
DOVISH, HAWKISH = "#2a78d6", "#e34948"
SYMBOLES = {"hausse": "▲", "baisse": "▼", "neutre": "■"}
# Quatre couleurs fixes pour les régimes (palette catégorielle validée ; info-bulle et
# tableau en clair comme repère supplémentaire pour les daltoniens)
COULEURS_REGIMES = {"goldilocks": "#1baf7a", "reflation": "#eda100",
                    "stagflation": "#e34948", "desinflation": "#2a78d6"}

config = charger_config()
SORTIES = RACINE_PROJET / config["general"]["dossier_sorties"]
RAFRAICHISSEMENT = config["tableau_de_bord"]["rafraichissement_secondes"]

st.set_page_config(page_title="Bot macro FX", page_icon="💱", layout="wide")


# ---------- Lecture des fichiers produits par la tâche de mise à jour ----------
# maj.json est écrit en dernier par scripts/mettre_a_jour.py : sa date de modification sert de
# numéro de version. Passée en argument, elle fait partie de la clé du cache de Streamlit, qui
# relit donc les fichiers dès qu'une mise à jour se termine, sans délai.

def version_maj() -> float:
    """Date de modification de maj.json (0 s'il n'existe pas encore)."""
    chemin = SORTIES / "maj.json"
    return chemin.stat().st_mtime if chemin.exists() else 0.0


@st.cache_data(max_entries=2)
def lire_etat(version: float) -> tuple[pd.DataFrame, dict]:
    etat = pd.read_csv(SORTIES / "etat_paires.csv")
    chemin_maj = SORTIES / "maj.json"
    maj = json.loads(chemin_maj.read_text(encoding="utf-8")) if chemin_maj.exists() else {}
    return etat, maj


@st.cache_data(max_entries=10)
def lire_historique(paire: str, version: float) -> pd.DataFrame:
    return pd.read_csv(SORTIES / f"fair_value_{paire}.csv", index_col=0, parse_dates=True)


@st.cache_data(max_entries=2)
def lire_sentiment(version: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    banques = pd.read_csv(SORTIES / "sentiment_banques.csv")
    phrases = pd.read_csv(SORTIES / "sentiment_phrases_communiques.csv")
    return banques, phrases


@st.cache_data(max_entries=2)
def lire_regimes(version: float) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """État courant des régimes et historique (une ligne par devise et par mois), s'ils existent."""
    chemin_etat = SORTIES / "etat_regimes.csv"
    if not chemin_etat.exists():
        return None
    etat = pd.read_csv(chemin_etat)
    historiques = [pd.read_csv(SORTIES / f"regimes_{devise}.csv", parse_dates=["date"]).assign(devise=devise)
                   for devise in etat["devise"] if (SORTIES / f"regimes_{devise}.csv").exists()]
    return etat, pd.concat(historiques, ignore_index=True)


@st.cache_data(ttl=RAFRAICHISSEMENT, show_spinner="Récupération des derniers cours...")
def cours_recents(tickers: tuple[str, ...]) -> pd.DataFrame:
    return derniers_cours(list(tickers))


# ---------- Mise en forme ----------

def libelle_signal(orientation: str, force: str, paire: str) -> str:
    if orientation == "neutre":
        return f"{SYMBOLES['neutre']} Neutre"
    return f"{SYMBOLES[orientation]} {orientation.capitalize()} de {paire} · {force}"


def graphique_fair_values(historique: pd.DataFrame, annees: int) -> alt.Chart:
    debut = historique.index.max() - pd.DateOffset(years=annees)
    donnees = (historique.loc[debut:, ["spot", "fv_ppa", "fv_beer"]]
               .rename(columns={"spot": "Spot", "fv_ppa": "Fair value PPA", "fv_beer": "Fair value BEER"})
               .reset_index(names="date").melt("date", var_name="série", value_name="cours").dropna())
    survol = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    couleur = alt.Color("série:N", scale=alt.Scale(domain=list(COULEURS_SERIES), range=list(COULEURS_SERIES.values())),
                        legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(donnees).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
                                     y=alt.Y("cours:Q", title=None, scale=alt.Scale(zero=False)))
    lignes = base.mark_line(strokeWidth=2).encode(color=couleur)
    points = base.mark_point(size=60, filled=True).encode(
        color=couleur, opacity=alt.condition(survol, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", format="%b %Y"), "série:N", alt.Tooltip("cours:Q", format=".4f")],
    ).add_params(survol)
    regle = base.mark_rule(color="gray", strokeWidth=1).encode(
        opacity=alt.condition(survol, alt.value(0.5), alt.value(0))).transform_filter(survol)
    return (lignes + points + regle).properties(height=320)


def graphique_z(historique: pd.DataFrame, annees: int, seuil: float) -> alt.Chart:
    debut = historique.index.max() - pd.DateOffset(years=annees)
    donnees = (historique.loc[debut:, ["z_ppa_ancre", "z_beer_ancre"]]
               .rename(columns={"z_ppa_ancre": "z PPA", "z_beer_ancre": "z BEER"})
               .reset_index(names="date").melt("date", var_name="modèle", value_name="z").dropna())
    couleur = alt.Color("modèle:N", scale=alt.Scale(domain=list(COULEURS_Z), range=list(COULEURS_Z.values())),
                        legend=alt.Legend(orient="top", title=None))
    lignes = alt.Chart(donnees).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")), y=alt.Y("z:Q", title="z-score ancré"), color=couleur,
        tooltip=[alt.Tooltip("date:T", format="%b %Y"), "modèle:N", alt.Tooltip("z:Q", format="+.2f")])
    seuils = alt.Chart(pd.DataFrame({"y": [-seuil, 0, seuil]})).mark_rule(
        color="gray", strokeDash=[4, 4], strokeWidth=1).encode(y="y:Q")
    return (seuils + lignes).properties(height=200)


def graphique_regimes(historique: pd.DataFrame, annees: int = 10) -> alt.Chart:
    """Frise : une ligne par devise, un rectangle par mois, couleur = régime confirmé."""
    debut = historique["date"].max() - pd.DateOffset(years=annees)
    donnees = historique.loc[(historique["date"] >= debut) & historique["regime"].notna(),
                             ["devise", "date", "regime", "mois_dans_regime"]]
    return alt.Chart(donnees).mark_rect().encode(
        x=alt.X("yearmonth(date):T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
        y=alt.Y("devise:N", title=None, sort=list(donnees["devise"].unique()),
                scale=alt.Scale(paddingInner=0.25)),
        color=alt.Color("regime:N", title=None,
                        scale=alt.Scale(domain=list(COULEURS_REGIMES), range=list(COULEURS_REGIMES.values())),
                        legend=alt.Legend(orient="top")),
        tooltip=["devise:N", alt.Tooltip("yearmonth(date):T", title="mois", format="%b %Y"),
                 alt.Tooltip("regime:N", title="régime"), alt.Tooltip("mois_dans_regime:Q", title="mois dans le régime")],
    ).properties(height=alt.Step(28))


def graphique_sentiment(banques: pd.DataFrame, modele: str) -> alt.Chart:
    donnees = banques.assign(score=banques[f"score_{modele}"],
                             ton=lambda d: d["score"].map(lambda s: "hawkish" if s >= 0 else "dovish"))
    return alt.Chart(donnees).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("banque:N", title=None, sort="-x"),
        x=alt.X("score:Q", title="dovish  ←  score  →  hawkish", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("ton:N", scale=alt.Scale(domain=["dovish", "hawkish"], range=[DOVISH, HAWKISH]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["banque:N", "devise:N", "date:N", alt.Tooltip("score:Q", format="+.2f"), "nb_phrases:Q"],
    ).properties(height=alt.Step(30))


# ---------- Page ----------

st.title("Bot macro FX")
st.caption("Outil d'analyse : écarts aux fair values (PPA, BEER) confirmés par le ton des banques centrales. "
           "Ce n'est ni un ordre ni un conseil d'investissement.")

if not (SORTIES / "etat_paires.csv").exists():
    st.error("Aucun résultat : lancer d'abord `python scripts/mettre_a_jour.py`.")
    st.stop()

etat, maj = lire_etat(version_maj())
params_signal = config["signal"]

with st.sidebar:
    st.subheader("Mise à jour des modèles")
    if maj:
        horodatage = pd.Timestamp(maj["horodatage_utc"]).tz_convert("Europe/Paris")
        st.write(f"{horodatage:%d/%m/%Y %H:%M} (Paris)")
        if maj.get("erreurs"):
            st.warning("Erreurs : " + ", ".join(maj["erreurs"]))
        perimees = maj.get("sources_perimees", [])  # absent d'un ancien maj.json
        if perimees:
            st.warning("Données non rafraîchies (source en panne, cache réutilisé) :\n"
                       + "\n".join(f"- {s['nom']} : {s['age_jours']:.1f} jours" for s in perimees))
    st.write(f"Cours rafraîchis toutes les {RAFRAICHISSEMENT // 60} min.")
    st.subheader("Paramètres du signal")
    st.write(f"Poids : {', '.join(f'{m.upper()} {w:.0%}' for m, w in params_signal['poids_modeles'].items())}")
    st.write(f"Seuil |z| : {params_signal['seuil_z']} · seuil sentiment : {params_signal['seuil_sentiment']}")
    st.caption("Modifiables dans config.yaml.")


@st.fragment(run_every=RAFRAICHISSEMENT)
def section_signaux() -> None:
    cours = cours_recents(tuple(etat["ticker"]))
    lignes = []
    for _, ligne in etat.iterrows():
        spot = cours.loc[ligne["ticker"], "cours"] if ligne["ticker"] in cours.index else ligne["spot_mensuel"]
        resultat = evaluer_signal(ligne, spot, config)
        heure = cours.loc[ligne["ticker"], "horodatage"] if ligne["ticker"] in cours.index else None
        lignes.append({"paire": ligne["paire"], "heure": heure,
                       "p_coint_beer": ligne.get("p_coint_beer", float("nan")), **resultat})
    signaux = pd.DataFrame(lignes)

    colonnes = st.columns(len(signaux))
    for colonne, (_, s) in zip(colonnes, signaux.iterrows()):
        with colonne.container(border=True):
            st.markdown(f"**{s['paire']}**")
            st.markdown(f"### {s['spot']:.4f}" if s["spot"] < 50 else f"### {s['spot']:.2f}")
            st.markdown(libelle_signal(s["orientation"], s["force"], s["paire"]))
            st.caption(f"z valorisation {s['z_valorisation']:+.2f}")
            if s.get("beer_exclu"):
                p = s["p_coint_beer"]
                detail = f"p = {p:.2f}".replace(".", ",") if pd.notna(p) else "p inconnue"
                st.caption(f"BEER non cointégré ({detail}) : exclu du signal")
            if s["orientation"] != "neutre":
                st.caption(s["explication"])

    tableau = signaux[["paire", "spot", "ecart_ppa_%", "ecart_beer_%", "z_ppa", "z_beer", "p_coint_beer",
                       "z_valorisation", "delta_sentiment"]]
    st.dataframe(tableau, hide_index=True, use_container_width=True, column_config={
        "spot": st.column_config.NumberColumn("Spot", format="%.4f"),
        "ecart_ppa_%": st.column_config.NumberColumn("Écart PPA", format="%+.1f %%"),
        "ecart_beer_%": st.column_config.NumberColumn("Écart BEER", format="%+.1f %%"),
        "z_ppa": st.column_config.NumberColumn("z PPA", format="%+.2f"),
        "z_beer": st.column_config.NumberColumn("z BEER", format="%+.2f"),
        "p_coint_beer": st.column_config.NumberColumn(
            "p coint. BEER", format="%.2f",
            help=f"p-value d'Engle-Granger ; au-delà de {params_signal['p_max_cointegration']}, "
                 "le BEER n'entre pas dans le signal"),
        "z_valorisation": st.column_config.NumberColumn("z valorisation", format="%+.2f"),
        "delta_sentiment": st.column_config.NumberColumn("Δ sentiment", format="%+.2f",
                                                         help="Score de la banque de base moins celui de la banque de cotation"),
    })
    heures = signaux["heure"].dropna()
    if not heures.empty:
        st.caption(f"Dernier cours : {pd.Timestamp(heures.max()):%d/%m/%Y %H:%M} (heure de la source, différé possible). "
                   "Écart positif : devise de base surévaluée.")


section_signaux()

st.divider()
st.subheader("Détail par paire")
choix_col, duree_col = st.columns([1, 1])
paire = choix_col.selectbox("Paire", etat["paire"].tolist())
annees = duree_col.select_slider("Historique (années)", options=[3, 5, 10, 20], value=10)
historique = lire_historique(paire, version_maj())
st.altair_chart(graphique_fair_values(historique, annees), use_container_width=True)
st.altair_chart(graphique_z(historique, annees, params_signal["seuil_z"]), use_container_width=True)
st.caption("Fair values et z-scores mensuels (spot en moyenne mensuelle). Pointillés : seuils du signal.")

st.divider()
st.subheader("Ton des banques centrales")
banques, phrases = lire_sentiment(version_maj())
modele = params_signal["modele_sentiment"]
graphe_col, phrases_col = st.columns([1, 1])
graphe_col.altair_chart(graphique_sentiment(banques, modele), use_container_width=True)
graphe_col.caption(f"Modèle : {modele}. Dernier communiqué de chaque banque.")
banque = phrases_col.selectbox("Phrases du communiqué", banques["banque"].tolist())
detail = phrases[phrases["banque"] == banque][["phrase", f"score_{modele}", f"etiquette_{modele}"]]
phrases_col.dataframe(detail.sort_values(f"score_{modele}"), hide_index=True, use_container_width=True, height=300,
                      column_config={f"score_{modele}": st.column_config.NumberColumn("Score", format="%+.2f"),
                                     f"etiquette_{modele}": "Étiquette",
                                     "phrase": st.column_config.TextColumn("Phrase", width="large")})

st.divider()
st.subheader("Régimes macro")
st.caption("Outil descriptif : environnement macro de chaque économie, d'après la dynamique de la confiance "
           "des entreprises (OCDE) et de l'inflation. Les régimes sont constatés avec retard (délai de "
           "publication + 2 mois de confirmation) : ce n'est pas une prévision, et ils n'entrent pas dans le signal.")
regimes = lire_regimes(version_maj())
if regimes is None:
    st.info("Pas encore de régimes : lancer `python scripts/mettre_a_jour.py`.")
else:
    etat_regimes, historique_regimes = regimes
    st.dataframe(etat_regimes, hide_index=True, use_container_width=True, column_config={
        "mois": "Mois", "regime": "Régime", "regime_brut": st.column_config.TextColumn(
            "Régime du mois", help="Différent du régime : changement en attente de confirmation"),
        "reporte": st.column_config.CheckboxColumn(
            "Non publié", help="Donnée du mois pas encore publiée : dernière valeur reprise, "
                               "le régime en place est conservé"),
        "mois_dans_regime": st.column_config.NumberColumn("Mois dans le régime", format="%d"),
        "croissance_niveau": st.column_config.NumberColumn("Confiance − 100", format="%+.2f"),
        "croissance_dynamique": st.column_config.NumberColumn("Δ confiance (3 m)", format="%+.2f"),
        "inflation_niveau": st.column_config.NumberColumn("Inflation", format="%.1f %%"),
        "ecart_cible": st.column_config.NumberColumn("Écart à la cible", format="%+.1f"),
        "cycle_monetaire": "Cycle monétaire",
        "ton_banque_centrale": st.column_config.NumberColumn("Ton banque centrale", format="%+.2f"),
    })
    st.altair_chart(graphique_regimes(historique_regimes), use_container_width=True)
    st.caption("Régime confirmé sur les 10 dernières années ; cases vides : données manquantes ou régime "
               "pas encore confirmé. Le ton de la banque centrale n'a pas d'historique : état actuel seulement.")
