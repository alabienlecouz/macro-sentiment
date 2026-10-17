import pandas as pd
import pytest

from fxmacro.donnees.alignement import raccorder, trimestriel_vers_mensuel


def test_trimestriel_vers_mensuel_respecte_la_publication():
    trim = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-04-01"]))
    mensuel = trimestriel_vers_mensuel(trim, decalage_apres_trimestre=3)
    # T1 (fin mars) publié 3 mois plus tard : utilisable en juin, pas avant
    assert mensuel.index[0] == pd.Timestamp("2020-06-01")
    assert mensuel.loc["2020-08-01"] == 1.0
    assert mensuel.loc["2020-09-01"] == 2.0
    assert mensuel.index[-1] == pd.Timestamp("2020-11-01")


def test_raccorder_recale_le_niveau():
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    ancienne = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    recente = pd.Series([3.5, 4.5], index=index[2:])
    serie = raccorder(recente, ancienne)
    assert len(serie) == 4
    assert serie.iloc[0] == pytest.approx(1.5)   # décalage moyen de +0,5
    assert serie.iloc[-1] == 4.5
