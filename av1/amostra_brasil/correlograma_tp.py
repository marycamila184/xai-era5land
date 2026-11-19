"""Correlograma do tp diário do ERA5-Land: correlação entre a âncora e células cada vez mais longe.

Para cada âncora de dados/ancoras.csv, nas 8 direções do anel (n/s/e/w andando em linha ou
coluna, diagonais andando nas duas), pega a célula a k passos para cada k de PASSOS e calcula
a correlação de Pearson do tp diário com a âncora. Célula fora da janela ou no mar fica sem
valor. As somas são acumuladas por mês do calendário, arquivo a arquivo, então dá para tirar
a correlação do ano inteiro ou de uma estação sem reler o disco.

Saída: dados/correlograma_tp.parquet, uma linha por âncora, direção e passo.

Uso (na raiz do projeto):
    uv run python av1/amostra_brasil/correlograma_tp.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz
from av1.amostra_brasil.ancoras import COL0, POSICOES, ROW0, centro_da_celula, distancia_km  # noqa: E402
from av1.amostra_brasil.era5land_diario import DAILY  # noqa: E402

DADOS = Path("av1/amostra_brasil/dados")
PASSOS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 27, 32, 40, 50])
DIRECOES = {k: np.sign(v) for k, v in POSICOES.items() if k != "center"}
t0 = time.time()

anc = pd.read_csv(DADOS / "ancoras.csv")
n_a, n_d, n_k = len(anc), len(DIRECOES), len(PASSOS)
dirs = np.array(list(DIRECOES.values()))                       # (8, 2)
r = anc.row_era5.to_numpy()[:, None, None] + dirs[None, :, None, 0] * PASSOS[None, None]
c = anc.col_era5.to_numpy()[:, None, None] + dirs[None, :, None, 1] * PASSOS[None, None]
iy, ix = r - ROW0, c - COL0                                     # (âncora, direção, passo)
iy0, ix0 = anc.row_era5.to_numpy() - ROW0, anc.col_era5.to_numpy() - COL0

# soma de n, x, y, x², y², xy por mês do calendário; x = âncora, y = célula afastada
S = np.zeros((6, 12, n_a, n_d, n_k))
dentro = None
for f in sorted((DAILY / "tp").glob("tp_*.nc")):
    with xr.open_dataset(f) as ds:
        tp = ds.tp.values * 1000                                # mm/dia
        mes = pd.DatetimeIndex(ds.valid_time.values).month[0]
    if dentro is None:
        dentro = (iy >= 0) & (iy < tp.shape[1]) & (ix >= 0) & (ix < tp.shape[2])
    x = tp[:, iy0, ix0][:, :, None, None]
    y = np.full((len(tp), n_a, n_d, n_k), np.nan)
    y[:, dentro] = tp[:, iy[dentro], ix[dentro]]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    S[:, mes - 1] += np.stack([ok.sum(0), x.sum(0), y.sum(0), (x * x).sum(0), (y * y).sum(0),
                               (x * y).sum(0)])
    if mes == 12:
        print(f"   {f.stem[3:7]} ({time.time() - t0:.0f}s)", flush=True)


def pearson(s):
    n, sx, sy, sxx, syy, sxy = s
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        return cov / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))


estacoes = {"ano": range(1, 13), "DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
lat0, lon0 = centro_da_celula(anc.row_era5.to_numpy(), anc.col_era5.to_numpy())
lat, lon = centro_da_celula(r, c)
saida = pd.DataFrame({
    "anchor_id": np.repeat(anc.anchor_id.to_numpy(), n_d * n_k),
    "region": np.repeat(anc.region.to_numpy(), n_d * n_k),
    "direction": np.tile(np.repeat(list(DIRECOES), n_k), n_a),
    "step": np.tile(PASSOS, n_a * n_d),
    "dist_km": distancia_km(lat0[:, None, None], lon0[:, None, None], lat, lon).ravel(),
    "n_days": S[0].sum(0).ravel().astype(int),
})
for nome, meses in estacoes.items():
    saida[f"corr_{nome}"] = pearson(S[:, [m - 1 for m in meses]].sum(1)).ravel()
saida.loc[saida.n_days == 0, [f"corr_{e}" for e in estacoes]] = np.nan
saida.to_parquet(DADOS / "correlograma_tp.parquet", index=False)
print(f"gravado: {len(saida)} linhas, {saida.corr_ano.notna().sum()} com valor ({time.time() - t0:.0f}s)")
