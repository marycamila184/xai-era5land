"""Monta a tabela do experimento com e sem grafo a partir do dataset de 150 km.

Uma linha por âncora e dia t. O alvo é o tp da âncora no dia t (`tp_mm`); todas as features
são da véspera (t−1), menos o seno e o cosseno do dia do ano (do dia t, que se sabe de antemão)
e a altitude (fixa).

Dois grupos de features:

- **locais** (braço sem grafo): as variáveis da própria âncora na véspera, `<var>_d1`.
- **grafo** (prefixo `g_`, somadas às locais no braço com grafo), calculadas nas 8 vizinhas
  do anel de ~150 km na véspera:
    1. agregados do anel: média, desvio e média menos âncora de cada variável; máximo e
       fração de vizinhas com mais de 1 mm para o tp;
    2. gradientes de t2m e d2m (°C/100 km), de um plano ajustado por mínimos quadrados às
       8 vizinhas (diferença para a âncora contra a posição em km);
    3. advecção de t2m e d2m pelo vento da âncora, −(u·∂/∂x + v·∂/∂y), em °C/dia;
    4. convergência e vorticidade do vento a 10 m, do mesmo ajuste de plano em u e v
       (10⁻⁵ s⁻¹; vorticidade negativa é giro ciclônico no hemisfério sul).

Saída: av1/experimentos/dados/features_<regiao>.parquet.

Uso (na raiz do projeto):
    uv run python av1/experimentos/montar_features.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENTRADA = Path("av1/amostra_brasil/dados_150")
SAIDA = Path("av1/experimentos/dados")
REGIOES = ["norte", "nordeste", "centro_oeste", "sudeste", "sul"]
VIZINHAS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
POSICOES = ["center"] + VIZINHAS
LOCAIS = ["tp_mm", "t2m", "t2m_min", "t2m_max", "d2m", "dpd", "ssrd_wm2", "pev_mm", "u10", "v10",
          "wind_speed", "wind_max", "wind_res", "wind_dir_sin", "wind_dir_cos"]
AGREGADAS = ["tp_mm", "t2m", "d2m", "dpd", "ssrd_wm2", "pev_mm", "wind_speed"]
GRADIENTES = ["t2m", "d2m"]
KM_POR_GRAU = 111.32


def cubo(d, var, n_a, n_t):
    """Valores de `var` em (âncora, dia, posição)."""
    x = np.full((n_a, n_t, len(POSICOES)), np.nan, np.float32)
    x[d["_a"].to_numpy(), d["_t"].to_numpy(), d["_p"].to_numpy()] = d[var].to_numpy()
    return x


def ajuste_plano(xy):
    """(âncora, 2, 8): leva a diferença vizinha − âncora ao gradiente (∂/∂x, ∂/∂y) por km."""
    xt = np.transpose(xy, (0, 2, 1))                                  # (a, 2, 8)
    return np.linalg.inv(xt @ xy) @ xt


def gradiente(m, v):
    """Gradiente por km em (âncora, dia, 2)."""
    return np.einsum("akp,atp->atk", m, v[:, :, 1:] - v[:, :, :1])


def montar(regiao):
    d = pd.read_parquet(ENTRADA / f"era5land_amostra_{regiao}.parquet",
                        columns=["date", "anchor_id", "position", "latitude", "longitude",
                                 "altitude", "region", "day_sin", "day_cos"] + LOCAIS)
    d["position"] = d["position"].astype(str)
    ancoras = np.sort(d.anchor_id.unique())
    datas = np.sort(d.date.unique())
    assert (np.diff(datas) == np.timedelta64(1, "D")).all(), "datas com buraco"
    n_a, n_t = len(ancoras), len(datas)
    d["_a"] = np.searchsorted(ancoras, d.anchor_id)
    d["_t"] = np.searchsorted(datas, d.date)
    d["_p"] = d.position.map({p: i for i, p in enumerate(POSICOES)})
    assert len(d) == n_a * n_t * len(POSICOES)

    # posição das vizinhas em km (x para leste, y para norte) em relação à âncora
    geo = d[d._t == 0].sort_values(["_a", "_p"])
    lat = geo.latitude.to_numpy().reshape(n_a, 9)
    lon = geo.longitude.to_numpy().reshape(n_a, 9)
    x_km = (lon[:, 1:] - lon[:, :1]) * np.cos(np.radians(lat[:, :1])) * KM_POR_GRAU
    y_km = (lat[:, 1:] - lat[:, :1]) * KM_POR_GRAU
    plano = ajuste_plano(np.stack([x_km, y_km], axis=2))

    v = {var: cubo(d, var, n_a, n_t) for var in LOCAIS}
    f = {}
    for var in LOCAIS:                                                 # 0. locais
        f[f"{var}_d1"] = v[var][:, :, 0]
    for var in AGREGADAS:                                              # 1. agregados do anel
        anel = v[var][:, :, 1:]
        f[f"g_mean_{var}"] = anel.mean(2)
        f[f"g_std_{var}"] = anel.std(2)
        f[f"g_diff_{var}"] = anel.mean(2) - v[var][:, :, 0]
    f["g_max_tp_mm"] = v["tp_mm"][:, :, 1:].max(2)
    f["g_frac1mm_tp_mm"] = (v["tp_mm"][:, :, 1:] > 1).mean(2)
    u, w = v["u10"][:, :, 0], v["v10"][:, :, 0]
    for var in GRADIENTES:                                             # 2. gradientes, 3. advecção
        g = gradiente(plano, v[var])                                   # °C/km
        f[f"g_gradx_{var}"] = 100 * g[..., 0]
        f[f"g_grady_{var}"] = 100 * g[..., 1]
        f[f"g_gradmag_{var}"] = 100 * np.hypot(g[..., 0], g[..., 1])
        f[f"g_adv_{var}"] = -86.4 * (u * g[..., 0] + w * g[..., 1])   # m/s × °C/km → °C/dia
    gu, gv = gradiente(plano, v["u10"]), gradiente(plano, v["v10"])  # 4. cinemática, (m/s)/km
    f["g_conv"] = -100 * (gu[..., 0] + gv[..., 1])                     # (m/s)/km = 100 × 10⁻⁵ s⁻¹
    f["g_vort"] = 100 * (gv[..., 0] - gu[..., 1])

    # alvo no dia t, features em t−1
    centro = d[d._p == 0].sort_values(["_a", "_t"])
    alvo = centro.tp_mm.to_numpy().reshape(n_a, n_t)
    out = pd.DataFrame({
        "date": np.tile(datas[1:], n_a),
        "anchor_id": np.repeat(ancoras, n_t - 1),
        "region": centro.region.iloc[0],
        "tp_mm": alvo[:, 1:].ravel(),
        "day_sin": centro.day_sin.to_numpy().reshape(n_a, n_t)[:, 1:].ravel(),
        "day_cos": centro.day_cos.to_numpy().reshape(n_a, n_t)[:, 1:].ravel(),
        "altitude": np.repeat(centro.altitude.to_numpy().reshape(n_a, n_t)[:, 0], n_t - 1),
    })
    for nome, x in f.items():
        out[nome] = x[:, :-1].ravel().astype(np.float32)
    return out


if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for regiao in REGIOES:
        out = montar(regiao)
        out.to_parquet(SAIDA / f"features_{regiao}.parquet", index=False)
        print(f"{regiao}: {len(out):,} linhas, {out.shape[1]} colunas ({time.time() - t0:.0f}s)",
              flush=True)
