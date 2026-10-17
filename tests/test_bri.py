"""Test hors ligne du parseur CSV de la BRI."""

import pandas as pd

from fxmacro.donnees.bri import lire_csv_bri

EXEMPLE = """DATAFLOW,FREQ,REF_AREA,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE
BIS:WS_LONG_CPI(1.0),M,JP,628,2024-02,106.5
BIS:WS_LONG_CPI(1.0),M,JP,628,2024-01,106.2
BIS:WS_LONG_CPI(1.0),M,JP,628,2024-03,NaN
"""


def test_lire_csv_bri():
    serie = lire_csv_bri(EXEMPLE)
    assert serie.index[0] == pd.Timestamp("2024-01-01")  # trié, début de mois
    assert serie.iloc[1] == 106.5
    assert serie.isna().iloc[2]
