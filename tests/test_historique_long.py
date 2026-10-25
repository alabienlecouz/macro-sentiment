"""Tests hors ligne de l'historique long : raccord des indices et euro synthétique."""

import numpy as np
import pandas as pd
import pytest

from fxmacro.donnees.historique_long import euro_synthetique, raccorder_proportion


def test_raccord_en_proportion_garde_la_serie_recente():
    index = pd.date_range("1990-01-01", periods=10, freq="MS")
    ancienne = pd.Series(np.arange(50.0, 60.0), index=index)              # base différente
    recente = pd.Series(2 * np.arange(55.0, 60.0), index=index[5:])        # même dynamique, x2
    serie = raccorder_proportion(recente, ancienne)
    assert len(serie) == 10
    pd.testing.assert_series_equal(serie.loc[index[5]:], recente, check_names=False)
    assert serie.iloc[0] == pytest.approx(100.0)                           # 50 remis à l'échelle (x2)
    # Le taux de variation de l'ancienne série est conservé
    assert serie.iloc[1] / serie.iloc[0] == pytest.approx(51 / 50)


def test_raccord_sans_periode_commune_leve_une_erreur():
    a = pd.Series([1.0], index=pd.to_datetime(["1990-01-01"]))
    b = pd.Series([1.0], index=pd.to_datetime(["2000-01-01"]))
    with pytest.raises(ValueError):
        raccorder_proportion(b, a)


def test_euro_synthetique():
    # 1,95583 mark pour 1 dollar -> 1 euro = 1 dollar ; 1,5 mark pour 1 dollar -> 1 euro = 1,3039 dollar
    marks = pd.Series([1.95583, 1.5])
    assert euro_synthetique(marks, 1.95583).tolist() == pytest.approx([1.0, 1.95583 / 1.5])
