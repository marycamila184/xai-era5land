"""Figura do grafo de uma âncora e da extração das variáveis de grafo num dia real.

- grafo_estrela.png: a âncora, as 8 vizinhas do anel de ~150 km e as arestas da estrela sobre a
  grade de 0,1° do ERA5-Land, entre as outras âncoras do Sul.
- grafo_extracao.png: a véspera de um dia de chuva forte, com o campo de chuva do ERA5-Land em
  todas as células ao fundo, o tp e o vento a 10 m em cada nó, o gradiente de t2m na âncora e os
  valores das variáveis de grafo daquele dia, com o alvo do dia seguinte.

O campo de chuva vem do processed_daily no disco externo (/media/mary-camila/Expansion), que
precisa estar montado.

O caso (Itaqui, RS, alvo em 29/07/1994) está no período de treino, fora do teste.

Saída: av1/experimentos/figuras/grafo_estrela.png e grafo_extracao.png.

Uso (na raiz do projeto, depois de montar_features.py):
    uv run python av1/experimentos/figura_grafo.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize

ANCORA, DIA = 4310603, pd.Timestamp("1994-07-29")
VESPERA = DIA - pd.Timedelta("1D")
BRUTO = Path("av1/amostra_brasil/dados_150/era5land_amostra_sul.parquet")
FEATURES = Path("av1/experimentos/dados/features_sul.parquet")
ANCORAS = Path("av1/amostra_brasil/dados_150/ancoras.csv")
UFS = Path("av1/amostra_brasil/dados/ibge/ufs.json")
FIGURAS = Path("av1/experimentos/figuras")
TP_DIARIO = Path("/media/mary-camila/Expansion/era5land/processed_daily/tp")
TINTA, TINTA_2, CINZA = "#1f1f1e", "#5f5e5a", "#b4b2a9"
ACENTO = "#eb6834"
AZUL = LinearSegmentedColormap.from_list(
    "azul", ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": CINZA, "axes.labelcolor": TINTA_2,
                     "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlecolor": TINTA})


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


def contorno_ufs(ax):
    for f in json.loads(UFS.read_text())["features"]:
        g = f["geometry"]
        for pol in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            anel = np.asarray(pol[0])
            ax.plot(anel[:, 0], anel[:, 1], color=CINZA, lw=0.6, zorder=1)


def grade(ax, x0, x1, y0, y1):
    """Linhas da grade de 0,1° do ERA5-Land (bordas das células, a 0,05° dos centros)."""
    xs = np.arange(np.floor(x0 * 10) / 10 - 0.05, x1 + 0.1, 0.1)
    ys = np.arange(np.floor(y0 * 10) / 10 - 0.05, y1 + 0.1, 0.1)
    ax.vlines(xs, y0, y1, color=CINZA, lw=0.25, alpha=0.6, zorder=0)
    ax.hlines(ys, x0, x1, color=CINZA, lw=0.25, alpha=0.6, zorder=0)


def campo_tp(dia):
    """tp do dia (mm) em toda a janela do disco, com latitude e longitude."""
    with xr.open_dataset(TP_DIARIO / f"tp_{dia:%Y_%m}.nc") as ds:
        k = list(pd.DatetimeIndex(ds.valid_time.values).normalize()).index(dia)
        return ds.latitude.values, ds.longitude.values, ds.tp.values[k] * 1000


def estrela(ax, nos, lw=1.2):
    c = nos[nos.position == "center"].iloc[0]
    for _, v in nos[nos.position != "center"].iterrows():
        ax.plot([c.longitude, v.longitude], [c.latitude, v.latitude], color=TINTA_2, lw=lw, zorder=2)


if __name__ == "__main__":
    bruto = pd.read_parquet(BRUTO, filters=[("anchor_id", "==", ANCORA),
                                            ("date", "in", [VESPERA, DIA])])
    bruto["position"] = bruto.position.astype(str)
    nos = bruto[bruto.date == VESPERA].reset_index(drop=True)
    alvo = float(bruto[(bruto.date == DIA) & (bruto.position == "center")].tp_mm.iloc[0])
    f = pd.read_parquet(FEATURES, filters=[("anchor_id", "==", ANCORA), ("date", "==", DIA)]).iloc[0]
    anc = pd.read_csv(ANCORAS)
    nome = anc.set_index("anchor_id").loc[ANCORA, "municipality"]
    c = nos[nos.position == "center"].iloc[0]

    FIGURAS.mkdir(parents=True, exist_ok=True)

    # (a) o grafo em estrela sobre a grade, entre as outras âncoras do Sul
    fig, ax_a = plt.subplots(figsize=(4.4, 3.9))
    contorno_ufs(ax_a)
    sul = anc[anc.region == "Sul"]
    ax_a.scatter(sul.longitude, sul.latitude, s=6, color=CINZA, lw=0, zorder=2, label="outras âncoras")
    estrela(ax_a, nos, lw=1)
    viz = nos[nos.position != "center"]
    ax_a.scatter(viz.longitude, viz.latitude, s=22, color="white", edgecolor=TINTA, lw=1, zorder=3,
                 label="vizinhas (123 a 145 km)")
    ax_a.scatter([c.longitude], [c.latitude], s=50, color=ACENTO, edgecolor="white", lw=0.8, zorder=4,
                 label=f"âncora ({nome}, RS)")
    for _, v in viz.iterrows():
        dx, dy = v.longitude - c.longitude, v.latitude - c.latitude
        r = np.hypot(dx, dy)
        ax_a.text(v.longitude + 0.4 * dx / r, v.latitude + 0.4 * dy / r, v.position.upper(),
                  fontsize=6.5, color=TINTA_2, ha="center", va="center", zorder=5)
    lim_a = (c.longitude - 2.6, c.longitude + 3.4, c.latitude - 2.9, c.latitude + 2.3)
    grade(ax_a, *lim_a)
    ax_a.set(xlim=lim_a[:2], ylim=lim_a[2:], aspect="equal", xticks=[], yticks=[])
    for s_ in ax_a.spines.values():
        s_.set_visible(False)
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=6.5,
                labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(FIGURAS / "grafo_estrela.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # (b) as variáveis de grafo na véspera, sobre o campo de chuva
    fig = plt.figure(figsize=(4.4, 5.0))
    sub = fig.add_gridspec(2, 1, height_ratios=[4.7, 1.5], hspace=0.45)
    ax_b = fig.add_subplot(sub[0])
    norm = Normalize(0, max(nos.tp_mm.max(), alvo))
    lat, lon, tp = campo_tp(VESPERA)
    i = (lat >= c.latitude - 1.8) & (lat <= c.latitude + 1.9)
    j = (lon >= c.longitude - 1.9) & (lon <= c.longitude + 2.1)
    ax_b.pcolormesh(lon[j], lat[i], tp[np.ix_(i, j)], cmap=AZUL, norm=norm, shading="nearest",
                    edgecolor="white", lw=0.1, alpha=0.85, rasterized=True, zorder=0)
    estrela(ax_b, nos, lw=0.7)
    ax_b.scatter(nos.longitude, nos.latitude, s=230, c=nos.tp_mm, cmap=AZUL, norm=norm,
                 edgecolor=TINTA_2, lw=0.7, zorder=3)
    for _, v in nos.iterrows():
        escuro = norm(v.tp_mm) > 0.6
        ax_b.text(v.longitude, v.latitude, num(v.tp_mm), fontsize=5.8, ha="center", va="center",
                  color="white" if escuro else TINTA, zorder=4)
    # vento a 10 m ao lado de cada nó, centrado num ponto a nordeste do círculo
    ax_b.quiver(nos.longitude + 0.36, nos.latitude + 0.3, nos.u10, nos.v10, color=TINTA, scale=60,
                width=0.006, pivot="middle", zorder=5)
    # gradiente de t2m na âncora (aponta para o ar mais quente), saindo da borda do círculo
    gx, gy = f.g_gradx_t2m, f.g_grady_t2m
    ux, uy = gx / np.hypot(gx, gy), gy / np.hypot(gx, gy)
    ini = (c.longitude - 0.22 * ux, c.latitude - 0.22 * uy)
    fim = (c.longitude - 0.95 * ux, c.latitude - 0.95 * uy)
    ax_b.annotate("", xy=ini, xytext=fim, arrowprops=dict(arrowstyle="-|>", color=ACENTO, lw=1.6),
                  zorder=6)
    ax_b.text(fim[0] - 0.05, fim[1] - 0.06, "∇t2m (para o ar\nmais quente)", color=ACENTO, fontsize=6,
              ha="right", va="top", zorder=7,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    ax_b.set(xlim=(c.longitude - 1.75, c.longitude + 2.0), ylim=(c.latitude - 1.7, c.latitude + 1.75),
             aspect="equal", xticks=[], yticks=[])
    for s_ in ax_b.spines.values():
        s_.set_visible(False)
    ax_cb = ax_b.inset_axes([0.0, -0.075, 1.0, 0.03])   # barra de cores colada embaixo do mapa
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=AZUL), cax=ax_cb, orientation="horizontal")
    cb.set_label("chuva na véspera (mm). Setas pretas, vento a 10 m", color=TINTA_2, fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)

    # os valores calculados, em duas colunas
    ax_t = fig.add_subplot(sub[1])
    ax_t.axis("off")
    linhas = [
        ("chuva média no anel", f"{num(f.g_mean_tp_mm)} mm"),
        ("chuva máxima no anel", f"{num(f.g_max_tp_mm)} mm"),
        ("fração de vizinhas > 1 mm", num(f.g_frac1mm_tp_mm, 2)),
        ("convergência", f"{num(f.g_conv, 2)} × 10⁻⁵ s⁻¹"),
        ("vorticidade", f"{num(f.g_vort, 2)} × 10⁻⁵ s⁻¹"),
        ("|∇t2m|", f"{num(f.g_gradmag_t2m, 2)} °C/100 km"),
    ]
    ax_t.text(0.0, 1.0, f"Variáveis de grafo em {VESPERA:%d/%m/%Y}", fontsize=7, color=TINTA,
              fontweight="bold", va="top")
    for k, (rotulo, valor) in enumerate(linhas):
        col, lin = divmod(k, 3)
        x0 = 0.0 if col == 0 else 0.53
        y = 0.78 - lin * 0.2
        ax_t.text(x0, y, rotulo, fontsize=6.5, color=TINTA_2, va="top")
        ax_t.text(x0 + 0.46, y, valor, fontsize=6.5, color=TINTA, va="top", ha="right")
    ax_t.text(0.0, 0.12, f"Alvo, chuva da âncora em {DIA:%d/%m/%Y}", fontsize=7, color=TINTA,
              fontweight="bold", va="top")
    ax_t.text(0.99, 0.12, f"{num(alvo)} mm", fontsize=7, color=TINTA, fontweight="bold", va="top",
              ha="right")
    fig.savefig(FIGURAS / "grafo_extracao.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(nos[["position", "tp_mm", "u10", "v10", "t2m"]].round(2).to_string(index=False))
    print("alvo", alvo, "| gx gy", round(gx, 2), round(gy, 2))
