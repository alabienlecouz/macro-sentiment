import pandas as pd

from fxmacro.donnees.alignement import aligner_mensuel, decaler_publication, journalier_vers_mensuel


def test_journalier_vers_mensuel_moyenne_et_fin():
    dates = pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03"])
    serie = pd.Series([1.0, 3.0, 5.0], index=dates)
    moyenne = journalier_vers_mensuel(serie, "moyenne")
    fin = journalier_vers_mensuel(serie, "fin")
    assert moyenne.loc["2020-01-01"] == 2.0
    assert fin.loc["2020-01-01"] == 3.0
    assert moyenne.index[0] == pd.Timestamp("2020-01-01")


def test_decaler_publication():
    serie = pd.Series([100.0, 101.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    decalee = decaler_publication(serie, 1)
    # Le CPI de janvier n'est utilisable qu'en février
    assert decalee.loc["2020-02-01"] == 100.0


def test_aligner_mensuel_comble_un_trou_isole():
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    a = pd.Series([1.0, None, 3.0, 4.0], index=index)
    b = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    tableau = aligner_mensuel({"a": a, "b": b})
    assert len(tableau) == 4
    assert tableau.loc["2020-02-01", "a"] == 1.0


def test_aligner_mensuel_ne_prolonge_pas_la_reference():
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    spot = pd.Series([1.0, 1.1, 1.2], index=index[:3])
    macro = pd.Series([5.0, 5.0, 5.0, 5.0], index=index)
    tableau = aligner_mensuel({"spot": spot, "macro": macro}, combler_max=3, reference="spot")
    assert tableau.index[-1] == index[2]


def test_fusion_config_locale():
    from fxmacro.config import fusionner
    base = {"sentiment": {"moteur": "torch", "taille_lot": 16}, "autre": 1}
    assert fusionner(base, {"sentiment": {"moteur": "onnx"}}) == {"sentiment": {"moteur": "onnx", "taille_lot": 16}, "autre": 1}
