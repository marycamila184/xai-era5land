"""Figura de referência da grade do ERA5-Land e da amostragem das âncoras (seção de Dados).

O Brasil na grade de 0,1° do ERA5-Land, com a altitude de cada célula de terra dentro do país
(contorno das UFs do IBGE) e as 512 âncoras do anel de 150 km por região. O retângulo marca o
recorte da figura do grafo em estrela (figura_grafo.py), em Itaqui (RS).

A altitude e a máscara terra-mar vêm de commons/data/static_era5land.nc.

Saída: av1/experimentos/figuras/grade_brasil.png.

Uso (na raiz do projeto):
    uv run python av1/experimentos/figura_grade.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path as Caminho

ESTATICO = Path("commons/data/static_era5land.nc")
ANCORAS = Path("av1/amostra_brasil/dados_150/ancoras.csv")
CELULAS = Path("av1/amostra_brasil/dados_150/celulas.csv")
UFS = Path("av1/amostra_brasil/dados/ibge/ufs.json")
FIGURAS = Path("av1/experimentos/figuras")
ANCORA = 4310603   # Itaqui (RS), a mesma da figura do grafo
TINTA, TINTA_2, CINZA = "#1f1f1e", "#5f5e5a", "#b4b2a9"
COR_REGIAO = {"Norte": "#2a78d6", "Nordeste": "#eb6834", "Sudeste": "#1baf7a", "Sul": "#eda100",
              "Centro-Oeste": "#e87ba4"}
RELEVO = LinearSegmentedColormap.from_list(
    "relevo", ["#f4f3ef", "#e3e0d6", "#cbc6b8", "#aaa493", "#857f6e", "#5f5a4d"])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": CINZA, "axes.labelcolor": TINTA_2,
                     "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlecolor": TINTA})


def contorno_ufs(ax, lw=0.6):
    for f in json.loads(UFS.read_text())["features"]:
        g = f["geometry"]
        for pol in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            anel = np.asarray(pol[0])
            ax.plot(anel[:, 0], anel[:, 1], color=TINTA_2, lw=lw, zorder=2)


def dentro_do_brasil(lat, lon):
    """Máscara (lat, lon) das células cujo centro cai em alguma UF."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    pts = np.c_[lo.ravel(), la.ravel()]
    dentro = np.zeros(len(pts), bool)
    for f in json.loads(UFS.read_text())["features"]:
        g = f["geometry"]
        for pol in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            dentro |= Caminho(np.asarray(pol[0])).contains_points(pts)
    return dentro.reshape(la.shape)


def bordas(c):
    """Bordas das células a partir dos centros (passo de 0,1°)."""
    passo = np.diff(c).mean()
    return np.r_[c - passo / 2, c[-1] + passo / 2]


if __name__ == "__main__":
    st = xr.open_dataset(ESTATICO)
    lat, lon = st.latitude.values, st.longitude.values
    alt = np.where(st.lsm.values > 0.5, st.altitude.values, np.nan)
    alt_br = np.where(dentro_do_brasil(lat, lon), alt, np.nan)
    anc = pd.read_csv(ANCORAS)
    cel = pd.read_csv(CELULAS)
    cel = cel[cel.anchor_id == ANCORA]
    c = anc[anc.anchor_id == ANCORA].iloc[0]

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    m = ax.pcolormesh(bordas(lon), bordas(lat), alt_br, cmap=RELEVO, vmin=0, vmax=1500, rasterized=True,
                      zorder=1)
    contorno_ufs(ax, lw=0.3)
    for regiao, cor in COR_REGIAO.items():
        a = anc[anc.region == regiao]
        ax.scatter(a.longitude, a.latitude, s=4, color=cor, edgecolor="white", lw=0.2, zorder=3)
    # o recorte da figura do grafo em estrela (Itaqui)
    ax.add_patch(Rectangle((c.longitude - 2.6, c.latitude - 2.9), 6.0, 5.2, fill=False, ec=TINTA, lw=0.9,
                           zorder=4))
    ax.set(xlim=(-75, -33), ylim=(-34.5, 6), aspect="equal")
    ax.set_xlabel("longitude", fontsize=7)
    ax.set_ylabel("latitude", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(handles=[Patch(color=cor, label=f"{r} ({(anc.region == r).sum()})")
                       for r, cor in COR_REGIAO.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=6.5,
              labelcolor=TINTA, handlelength=1.2, columnspacing=1.2)
    cb = fig.colorbar(m, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("altitude da célula (m)", color=TINTA_2, fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / "grade_brasil.png", dpi=250, bbox_inches="tight")
    print(f"figura em {FIGURAS}/")
