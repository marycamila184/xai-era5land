"""Gera o dataset ERA5-Land amostrado no Brasil: âncoras e vizinhas, um parquet por região.

Uma linha por célula e por dia UTC, 9 linhas por âncora (center e as 8 vizinhas do anel),
com as mesmas variáveis e a mesma agregação do dataset da RMC, todas do mesmo dia. As
células vêm de celulas.csv (escolher_ancoras.py) na pasta do anel. O vento diário fica
guardado por ano em <pasta do anel>/cache/, então uma corrida interrompida recomeça de onde parou.

Uso (na raiz do projeto, depois de escolher_ancoras.py):
    uv run python av1/amostra_brasil/gerar_dataset_amostra.py               # anel de 30 km
    uv run python av1/amostra_brasil/gerar_dataset_amostra.py --anel 150km
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz
from av1.amostra_brasil.ancoras import COL0, REGIOES, ROW0, anel_da_linha_de_comando  # noqa: E402
from av1.amostra_brasil.era5land_diario import (ler_diario, ler_vento, montar_features,  # noqa: E402
                                                 t0)

ANEL, CONFIG = anel_da_linha_de_comando()
POSICOES = CONFIG["posicoes"]
DADOS = CONFIG["dados"]
CACHE = DADOS / "cache"
ARQUIVO = {"Norte": "norte", "Nordeste": "nordeste", "Sudeste": "sudeste", "Sul": "sul",
           "Centro-Oeste": "centro_oeste"}
ROTULOS = ["region", "state", "municipality", "municipality_code", "municipality_area_km2"]

anc = pd.read_csv(DADOS / "ancoras.csv")
cel = pd.read_csv(DADOS / "celulas.csv").merge(anc[["anchor_id"] + ROTULOS], on="anchor_id", how="left")
if cel.duplicated(["row_era5", "col_era5"]).any():
    raise ValueError("há célula em duas âncoras: refaça escolher_ancoras.py")
iy, ix = (cel.row_era5 - ROW0).to_numpy(), (cel.col_era5 - COL0).to_numpy()
print(f"anel {ANEL}: {len(anc)} âncoras, {len(cel)} células")

tp = ler_diario("tp", ["tp"], iy, ix)["tp"]
ssrd = ler_diario("ssrd", ["ssrd"], iy, ix)["ssrd"]
pev = ler_diario("pev", ["pev"], iy, ix)["pev"]
temp = ler_diario("temp", ["t2m", "d2m", "t2m_min", "t2m_max"], iy, ix)
vento = ler_vento(iy, ix, CACHE)

dias = tp.index
for df in (ssrd, pev, temp["t2m"], vento["u10"]):
    dias = dias.intersection(df.index)
passos = dias.to_series().diff().dropna()
if not (passos == pd.Timedelta("1D")).all():
    raise ValueError("o eixo diário tem buracos")
print(f"período comum: {dias.min():%Y-%m-%d} a {dias.max():%Y-%m-%d} ({len(dias)} dias)")

n_ano = np.where(dias.is_leap_year, 366, 365)
ang = 2 * np.pi * (dias.dayofyear - 1) / n_ano
posicao = pd.CategoricalDtype(list(POSICOES))

for regiao in REGIOES.values():
    colunas = np.flatnonzero(cel.region.to_numpy() == regiao)
    c = cel.iloc[colunas]
    feat = montar_features(tp, ssrd, pev, temp, vento, dias, colunas)
    n_d, n_c = len(dias), len(c)
    df = pd.DataFrame({
        "date": np.repeat(dias.values, n_c),
        "anchor_id": np.tile(c.anchor_id.to_numpy("int32"), n_d),
        "position": pd.Categorical(np.tile(c.position.to_numpy(), n_d), dtype=posicao),
        "dist_km": np.tile(c.dist_km.to_numpy("float32"), n_d),
        "latitude": np.tile(c.latitude.to_numpy("float32"), n_d),
        "longitude": np.tile(c.longitude.to_numpy("float32"), n_d),
        "row_era5": np.tile(c.row_era5.to_numpy("int16"), n_d),
        "col_era5": np.tile(c.col_era5.to_numpy("int16"), n_d),
        "altitude": np.tile(c.altitude.to_numpy("float32"), n_d),
        "region": pd.Categorical(np.tile(c.region.to_numpy(), n_d)),
        "state": pd.Categorical(np.tile(c.state.to_numpy(), n_d)),
        "municipality": pd.Categorical(np.tile(c.municipality.to_numpy(), n_d)),
        "municipality_code": np.tile(c.municipality_code.to_numpy("int32"), n_d),
        "municipality_area_km2": np.tile(c.municipality_area_km2.to_numpy("float32"), n_d),
    })
    for nome, tab in feat.items():
        df[nome] = np.asarray(tab).ravel().astype("float32")
    df["day_sin"] = np.repeat(np.sin(ang), n_c).astype("float32")
    df["day_cos"] = np.repeat(np.cos(ang), n_c).astype("float32")

    saida = DADOS / f"era5land_amostra_{ARQUIVO[regiao]}.parquet"
    df.to_parquet(saida, index=False)
    nan = df.isna().sum()
    print(f"{saida}: {c.anchor_id.nunique()} âncoras, {len(df):,} linhas x {df.shape[1]} colunas; "
          f"NaN: {nan[nan > 0].to_dict() or 'nenhum'} ({time.time() - t0:.0f}s)", flush=True)
    del df, feat

print("pronto")
