"""Testes do dataset ERA5-Land amostrado no Brasil (era5land_amostra_<regiao>.parquet).

1. Estrutura, por arquivo: 9 linhas por âncora e dia (uma de cada posição), dias contínuos
   de 1980-01-01 a 2025-12-30, sem NaN fora de wind_const, rótulos iguais nas 9 linhas.
2. Geometria: deslocamento de cada vizinha = posições do anel; nenhuma célula em duas
   âncoras; âncoras a >= 7 células (Chebyshev); dist_km na faixa do anel (27 a 34 no de
   30 km, 120 a 148 no de 150 km) nas vizinhas e 0 na center.
3. Recálculo direto do raw horário em células e dias sorteados, por índice inteiro.
4. Faixas físicas (Brasil todo).
5. Contagem de âncoras por região igual à de ancoras.csv.

Uso (na raiz do projeto, depois de gerar_dataset_amostra.py):
    uv run python av1/amostra_brasil/testar_dataset_amostra.py               # anel de 30 km
    uv run python av1/amostra_brasil/testar_dataset_amostra.py --anel 150km
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz
from av1.amostra_brasil.ancoras import (COL0, ESPACAMENTO, REGIOES, ROW0,  # noqa: E402
                                         anel_da_linha_de_comando)

ANEL, CONFIG = anel_da_linha_de_comando()
POSICOES = CONFIG["posicoes"]
DADOS = CONFIG["dados"]
DIST_MIN, DIST_MAX = CONFIG["dist_km"]
ARQUIVO = {"Norte": "norte", "Nordeste": "nordeste", "Sudeste": "sudeste", "Sul": "sul",
           "Centro-Oeste": "centro_oeste"}
RAW = Path("/media/mary-camila/Expansion/era5land/raw")
ROTULOS = ["region", "state", "municipality", "municipality_code", "municipality_area_km2"]
INICIO, FIM = pd.Timestamp("1980-01-01"), pd.Timestamp("2025-12-30")
N_SORTEIO_POR_ARQUIVO = 3
TOL = 1e-3
FAIXAS = {"tp_mm": (0, 600), "altitude": (-100, 3000), "ssrd_wm2": (0, 450), "t2m": (-10, 45),
          "t2m_min": (-20, 40), "t2m_max": (-5, 48), "dpd": (-0.01, 40), "wind_speed": (0, 25),
          "wind_max": (0, 40), "wind_const": (0, 1.0001)}
falhas = []


def confere(nome, ok, detalhe=""):
    print(f"   [{'ok' if ok else 'FALHOU'}] {nome}{(': ' + detalhe) if detalhe else ''}")
    if not ok:
        falhas.append(nome)


print(f"anel {ANEL}, {DADOS}")
anc = pd.read_csv(DADOS / "ancoras.csv")
rng = np.random.default_rng(18)
geometria, faixas = [], {c: [np.inf, -np.inf] for c in FAIXAS}

for regiao, nome in ARQUIVO.items():
    print(f"\n== {regiao} ==")
    df = pd.read_parquet(DADOS / f"era5land_amostra_{nome}.parquet")
    n_anc, n_dias = df.anchor_id.nunique(), df.date.nunique()

    print("1. Estrutura")
    confere("9 linhas por âncora e dia", len(df) == n_anc * 9 * n_dias
            and not df.duplicated(["date", "anchor_id", "position"]).any(),
            f"{len(df):,} = {n_anc} âncoras x 9 x {n_dias} dias")
    dias = pd.DatetimeIndex(sorted(df.date.unique()))
    confere("dias contínuos, 1980-01-01 a 2025-12-30",
            dias.min() == INICIO and dias.max() == FIM and (np.diff(dias.values) == np.timedelta64(1, "D")).all())
    nan = df.drop(columns="wind_const").isna().sum()
    confere("sem NaN (fora wind_const)", nan.sum() == 0, str(nan[nan > 0].to_dict()) if nan.sum() else "")
    print(f"   wind_const NaN em {int(df.wind_const.isna().sum())} linhas (calmaria total)")
    confere("rótulos iguais nas 9 linhas", (df.groupby("anchor_id", observed=True)[ROTULOS].nunique() == 1).all().all())
    confere("só a região do arquivo", set(df.region.unique()) == {regiao})

    um_dia = df[df.date == INICIO]
    geometria.append(um_dia[["anchor_id", "position", "row_era5", "col_era5", "dist_km"]])

    print("3. Recálculo direto do raw horário")
    for r in df.iloc[rng.choice(len(df), N_SORTEIO_POR_ARQUIVO, replace=False)].itertuples():
        d = pd.Timestamp(r.date)
        prox = d + pd.Timedelta("1D")
        iy, ix = int(r.row_era5) - ROW0, int(r.col_era5) - COL0

        def ponto(grupo, dia):
            with xr.open_dataset(RAW / grupo / f"{grupo}_{dia.year}_{dia.month:02d}.nc") as ds:
                return ds.isel(latitude=iy, longitude=ix).load()

        acum = {g: float(ponto(g, prox)[g].sel(valid_time=prox)) for g in ("tp", "ssrd", "pev")}
        t = ponto("temp", d).sel(valid_time=slice(d, d + pd.Timedelta("23h")))
        w = ponto("wind", d).sel(valid_time=slice(d, d + pd.Timedelta("23h")))
        vel = np.sqrt(w.u10 ** 2 + w.v10 ** 2)
        esperado = {"tp_mm": acum["tp"] * 1000, "ssrd_wm2": acum["ssrd"] / 86400, "pev_mm": acum["pev"] * 1000,
                    "t2m": float(t.t2m.mean()) - 273.15, "t2m_min": float(t.t2m.min()) - 273.15,
                    "t2m_max": float(t.t2m.max()) - 273.15, "d2m": float(t.d2m.mean()) - 273.15,
                    "u10": float(w.u10.mean()), "v10": float(w.v10.mean()),
                    "wind_speed": float(vel.mean()), "wind_max": float(vel.max())}
        n_horas = (t.sizes["valid_time"], w.sizes["valid_time"])
        difs = {k: abs(getattr(r, k) - v) for k, v in esperado.items()}
        pior = max(difs, key=difs.get)
        confere(f"{d:%Y-%m-%d} {r.position} de {r.anchor_id} ({r.latitude:.1f}, {r.longitude:.1f})",
                max(difs.values()) < 0.01 and n_horas == (24, 24),
                f"tp {r.tp_mm:.2f} = {esperado['tp_mm']:.2f} mm; pior diferença em {pior}: {difs[pior]:.2g}")

    print("4. Faixas físicas")
    for col, (a, b) in FAIXAS.items():
        s = df[col].dropna()
        faixas[col] = [min(faixas[col][0], s.min()), max(faixas[col][1], s.max())]
        confere(f"{col} em [{a:g}, {b:g}]", s.between(a, b).all(), f"min {s.min():.2f}, max {s.max():.2f}")
    confere("t2m_min <= t2m <= t2m_max", ((df.t2m_min <= df.t2m + TOL) & (df.t2m <= df.t2m_max + TOL)).all())
    confere("wind_res <= wind_speed", (df.wind_res <= df.wind_speed + TOL).all())

    print("5. Contagem")
    esperadas = int((anc.region == regiao).sum())
    confere("âncoras = ancoras.csv", n_anc == esperadas, f"{n_anc} de {esperadas}")
    del df

print("\n== Brasil ==")
print("2. Geometria")
g = pd.concat(geometria)
centro = g[g.position == "center"].set_index("anchor_id")[["row_era5", "col_era5"]]
g = g.join(centro, on="anchor_id", rsuffix="_anc")
desloc = list(zip(g.row_era5 - g.row_era5_anc, g.col_era5 - g.col_era5_anc))
confere(f"deslocamento de cada vizinha = posições do anel {ANEL}", all(d == POSICOES[p] for d, p in zip(desloc, g.position)))
confere("nenhuma célula em duas âncoras", not g.duplicated(["row_era5", "col_era5"]).any(), f"{len(g)} células")
r, c = centro.row_era5.to_numpy(int), centro.col_era5.to_numpy(int)
cheb = np.maximum(abs(r[:, None] - r[None]), abs(c[:, None] - c[None]))
np.fill_diagonal(cheb, 10**6)
confere(f"âncoras a >= {ESPACAMENTO} células", cheb.min() >= ESPACAMENTO, f"menor distância {cheb.min()} células")
viz = g[g.position != "center"].dist_km
confere(f"dist_km de {DIST_MIN} a {DIST_MAX} nas vizinhas e 0 na center", viz.between(DIST_MIN, DIST_MAX).all()
        and (g[g.position == "center"].dist_km == 0).all(), f"{viz.min():.1f} a {viz.max():.1f} km")
confere("total de âncoras = ancoras.csv", len(centro) == len(anc), f"{len(centro)}")
print("faixas no Brasil:", {k: (round(float(a), 2), round(float(b), 2)) for k, (a, b) in faixas.items()})
print("âncoras por região:", {k: int((anc.region == k).sum()) for k in REGIOES.values()})

print(f"\n{'TUDO OK' if not falhas else str(len(falhas)) + ' FALHA(S): ' + ', '.join(falhas)}")
