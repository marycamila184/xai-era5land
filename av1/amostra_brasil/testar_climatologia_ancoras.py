"""Testes do CSV de climatologia das âncoras (ancoras_brasil_<anel>.csv).

Recalcula tudo sem passar pelos parquets nem pelos índices row/col do gerador: a célula é
escolhida pela latitude e longitude do CSV (vizinho mais próximo na grade do NetCDF).
1. Metadados: uma linha por âncora, as mesmas de ancoras.csv, sem NaN; região = 1º dígito
   do código IBGE; célula = sede arredondada a 0,1°.
2. Altitude igual à do arquivo estático, e a célula é terra (lsm > 0,5).
3. Médias de t2m, d2m e da chuva anual somadas em float64 direto do processed_daily, de
   1980-01-01 a 2025-12-30. O CSV não é arredondado; a diferença que sobra é a do float32
   dos parquets, de 0,001 °C e 0,01 mm no máximo.

Uso (na raiz do projeto, depois de climatologia_ancoras.py; ~30 min, lê o disco externo):
    uv run python av1/amostra_brasil/testar_climatologia_ancoras.py --anel 150km
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz
from av1.amostra_brasil.ancoras import REGIOES, anel_da_linha_de_comando  # noqa: E402

ANEL, CONFIG = anel_da_linha_de_comando()
DADOS = CONFIG["dados"]
DAILY = Path("/media/mary-camila/Expansion/era5land/processed_daily")
ESTATICO = Path("commons/data/static_era5land.nc")
FIM = np.datetime64("2025-12-30")
N_DIAS = 16801
t0 = time.time()
falhas = []


def checar(nome, ok, detalhe=""):
    print(f"{'ok   ' if ok else 'FALHA'} {nome} {detalhe}", flush=True)
    if not ok:
        falhas.append(nome)


def somar(grupo, nomes, lat, lon):
    """Média diária de cada variável e soma por ano, em float64, nas células (lat, lon)."""
    soma = {n: 0.0 for n in nomes}
    por_ano = {n: {} for n in nomes}
    n_dias = 0
    for f in sorted((DAILY / grupo).glob(f"{grupo}_*.nc")):
        with xr.open_dataset(f) as ds:
            ds = ds.sel(valid_time=ds.valid_time <= FIM)
            if ds.sizes["valid_time"] == 0:
                continue
            p = ds.sel(latitude=lat, longitude=lon, method="nearest")
            ano = pd.Timestamp(ds.valid_time.values[0]).year
            for n in nomes:
                v = p[n].values.astype(np.float64).sum(0)
                soma[n] = soma[n] + v
                por_ano[n][ano] = por_ano[n].get(ano, 0) + v
            n_dias += ds.sizes["valid_time"]
    print(f"   {grupo}: {n_dias} dias ({time.time() - t0:.0f}s)", flush=True)
    return {n: s / n_dias for n, s in soma.items()}, por_ano, n_dias


def main():
    csv = pd.read_csv(DADOS / f"ancoras_brasil_{ANEL}.csv")
    ancoras = pd.read_csv(DADOS / "ancoras.csv")

    checar("uma linha por âncora", csv.codigo_ibge.is_unique and len(csv) == len(ancoras),
           f"({len(csv)})")
    checar("mesmas âncoras de ancoras.csv", set(csv.codigo_ibge) == set(ancoras.anchor_id))
    checar("sem NaN", not csv.isna().any().any())
    checar("região = 1º dígito do código IBGE",
           (csv.regiao == (csv.codigo_ibge // 1_000_000).map(REGIOES)).all())
    m = csv.merge(ancoras, left_on="codigo_ibge", right_on="anchor_id", suffixes=("", "_a"))
    checar("célula = sede arredondada a 0,1°",
           np.allclose(m.latitude, np.round(m.lat_sede, 1))
           and np.allclose(m.longitude, np.round(m.lon_sede, 1)))

    lat = xr.DataArray(csv.latitude.values, dims="a")
    lon = xr.DataArray(csv.longitude.values, dims="a")
    with xr.open_dataset(ESTATICO) as st:
        p = st.sel(latitude=lat, longitude=lon, method="nearest")
        dif = np.abs(p.altitude.values - csv.altitude_m).max()
        checar("altitude igual à do arquivo estático", dif <= 1e-3, f"(máx |dif| {dif:.6f} m)")
        checar("célula em terra", (p.lsm.values > 0.5).all(), f"(lsm mín {p.lsm.values.min():.3f})")

    tp, tp_ano, n_tp = somar("tp", ["tp"], lat, lon)
    temp, _, n_temp = somar("temp", ["t2m", "d2m"], lat, lon)
    checar("16.801 dias de 1980 a 2025", n_tp == n_temp == N_DIAS and len(tp_ano["tp"]) == 46,
           f"(tp {n_tp}, temp {n_temp}, {len(tp_ano['tp'])} anos)")
    recalculado = {   # coluna -> (valor direto do NetCDF, tolerância do float32)
        "temperatura_media_c": (temp["t2m"] - 273.15, 1e-3),
        "ponto_orvalho_medio_c": (temp["d2m"] - 273.15, 1e-3),
        "chuva_media_anual_mm": (np.mean(list(tp_ano["tp"].values()), axis=0) * 1000, 1e-2),
    }
    for coluna, (valor, tolerancia) in recalculado.items():
        dif = np.abs(valor - csv[coluna]).max()
        checar(f"{coluna} igual ao NetCDF", dif <= tolerancia, f"(máx |dif| {dif:.6f})")

    print(f"\n{len(falhas)} falha(s)" if falhas else "\ntudo ok")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
