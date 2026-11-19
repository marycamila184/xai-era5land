"""Leitura diária do ERA5-Land em células escolhidas por índice.

As células vêm por índice inteiro na janela do disco (iy, ix), em vez de uma caixa
recortada. Cada arquivo é carregado inteiro (mais rápido que pedir ponto a ponto no
netCDF) e só depois as células são separadas.

- tp, ssrd e pev: o total do dia D é o passo das 00 UTC de D+1, já no processed_daily;
- t2m, d2m, t2m_min, t2m_max: do processed_daily;
- vento: do raw horário, com scripts/utils/wind_stats.daily_wind.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz, para scripts.utils
from scripts.utils.wind_stats import daily_wind, derive_wind  # noqa: E402

RAW = Path("/media/mary-camila/Expansion/era5land/raw")
DAILY = Path("/media/mary-camila/Expansion/era5land/processed_daily")
ANOS = range(1980, 2026)
DIA_S = 86400
t0 = time.time()


def ler_diario(grupo, nomes, iy, ix):
    """processed_daily do grupo: {variável: DataFrame dias x células}, em float32 como no disco."""
    blocos = {n: [] for n in nomes}
    tempos = []
    for f in sorted((DAILY / grupo).glob(f"{grupo}_*.nc")):
        with xr.open_dataset(f) as ds:
            for n in nomes:
                blocos[n].append(ds[n].values[:, iy, ix])
            tempos.append(pd.DatetimeIndex(ds.valid_time.values).normalize())
    idx = tempos[0].append(tempos[1:])
    print(f"   {grupo}: {len(idx)} dias, {idx.min():%Y-%m-%d} a {idx.max():%Y-%m-%d} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return {n: pd.DataFrame(np.vstack(blocos[n]), index=idx) for n in nomes}


def conferir_cache(cache, iy, ix):
    """O cache vale só para as células com que foi gravado (guardadas em celulas.npz)."""
    cache.mkdir(parents=True, exist_ok=True)
    arq = cache / "celulas.npz"
    if not arq.exists():
        np.savez(arq, iy=iy, ix=ix)
        return
    z = np.load(arq)
    if not (np.array_equal(z["iy"], iy) and np.array_equal(z["ix"], ix)):
        raise ValueError(f"o cache em {cache} foi gravado para outras células; apague a pasta")


def ler_vento(iy, ix, cache):
    """Vento diário do raw horário; um .npz por ano em cache, reaproveitado se existir."""
    conferir_cache(cache, iy, ix)
    nomes = ("u10", "v10", "wind_speed", "wind_max")
    partes, tempos = {n: [] for n in nomes}, []
    for ano in ANOS:
        arq = cache / f"vento_{ano}.npz"
        if not arq.exists():
            meses = []
            for mes in range(1, 13):
                with xr.open_dataset(RAW / "wind" / f"wind_{ano}_{mes:02d}.nc") as ds:
                    u, v = ds.u10.values[:, iy, ix], ds.v10.values[:, iy, ix]
                    hora = ds.valid_time.values
                pts = xr.Dataset({"u10": (("valid_time", "cel"), u), "v10": (("valid_time", "cel"), v)},
                                 coords={"valid_time": hora})
                meses.append(daily_wind(pts))
            d = xr.concat(meses, dim="valid_time")
            np.savez(arq, tempo=d.valid_time.values, **{n: d[n].values for n in nomes})
            print(f"   vento {ano} ({time.time() - t0:.0f}s)", flush=True)
        z = np.load(arq)
        tempos.append(pd.DatetimeIndex(z["tempo"]).normalize())
        for n in nomes:
            partes[n].append(z[n])
    idx = tempos[0].append(tempos[1:])
    return {n: pd.DataFrame(np.vstack(partes[n]), index=idx) for n in nomes}


def montar_features(tp, ssrd, pev, temp, vento, dias, colunas=slice(None)):
    """As colunas da RMC, com as mesmas conversões, nos dias e nas colunas (células) pedidas."""
    def pega(df):
        return df.loc[dias].iloc[:, colunas].astype("float64")

    feat = {
        "tp_mm": pega(tp) * 1000,
        "ssrd_wm2": pega(ssrd) / DIA_S,
        "pev_mm": pega(pev) * 1000,
        "t2m": pega(temp["t2m"]) - 273.15,
        "t2m_min": pega(temp["t2m_min"]) - 273.15,
        "t2m_max": pega(temp["t2m_max"]) - 273.15,
        "d2m": pega(temp["d2m"]) - 273.15,
    }
    feat["dpd"] = feat["t2m"] - feat["d2m"]
    for n in ("u10", "v10", "wind_speed", "wind_max"):
        feat[n] = pega(vento[n])
    feat.update({k: pd.DataFrame(np.asarray(v), index=dias) for k, v in
                 derive_wind(feat["u10"].values, feat["v10"].values, feat["wind_speed"].values).items()})
    return feat
