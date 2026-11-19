"""Figuras que explicam a validação espacial-temporal (validacao_espacial.py).

- estrategia_sul.png: no Sul, bloco a bloco, quais âncoras e quais anos treinam e quais validam.
- temporal_vs_espacial.png: o primeiro bloco nas duas validações, lado a lado.
- grupos_espaciais.png: os 5 grupos contíguos de cada região no mapa do Brasil.
- estrategia_brasil.png: a mesma validação no modelo completo do Brasil, em que o bloco i valida o
  grupo i de todas as regiões; as regiões aparecem afastadas umas das outras.
- estrategia_global.png: o teste espacial-temporal do modelo global, em que o modelo g treina
  sem o grupo g de cada região (1980–2014) e é testado nesse grupo (2015–2025).

Cores da paleta categórica validada (dataviz): treino em azul, validação em laranja. No mapa dos
grupos, a cor é a da região (como no mapa de âncoras) e cada grupo é identificado por número e
contorno.

Saída: av1/experimentos/figuras/.

Uso (na raiz do projeto):
    uv run python av1/experimentos/figuras_validacao.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treinar_p90 import BLOCOS  # noqa: E402
from validacao_espacial import NOMES_REGIAO, grupos_espaciais  # noqa: E402

FIGURAS = Path("av1/experimentos/figuras")
UFS = Path("av1/amostra_brasil/dados/ibge/ufs.json")
TREINO, VALID = "#2a78d6", "#eb6834"
TINTA, TINTA_2, CINZA, FORA = "#1f1f1e", "#5f5e5a", "#b4b2a9", "#e8e7e2"
ANO_INI, ANO_TESTE, ANO_FIM = 1980, 2015, 2025
EXTENSAO_SUL = dict(xlim=(-58.2, -47.6), ylim=(-34.0, -22.4))
# as cores por região do mapa de âncoras (amostra_brasil/figuras_amostra.py); laranja e amarelo
# ficam no Nordeste e no Sul, que não se tocam
COR_REGIAO = {"norte": "#2a78d6", "nordeste": "#eb6834", "sudeste": "#1baf7a", "sul": "#eda100",
              "centro_oeste": "#e87ba4"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": CINZA, "axes.labelcolor": TINTA_2,
                     "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlecolor": TINTA})
UF = json.loads(UFS.read_text())["features"]


def contorno_ufs(ax):
    for f in UF:
        g = f["geometry"]
        for pol in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            anel = np.asarray(pol[0])
            ax.plot(anel[:, 0], anel[:, 1], color=CINZA, lw=0.6, zorder=1)


CODIGO_REGIAO = {"1": "norte", "2": "nordeste", "3": "sudeste", "4": "sul", "5": "centro_oeste"}
SIGLA = {"norte": "N", "nordeste": "NE", "centro_oeste": "CO", "sudeste": "SE", "sul": "S"}
LADO_SIGLA = {"norte": "esquerda", "nordeste": "direita", "centro_oeste": "esquerda",
              "sudeste": "direita", "sul": "esquerda"}
CENTRO_BRASIL = np.array([-52.0, -14.0])
AFASTAMENTO = 0.4   # fração da distância centro do país → centro da região


def aneis(g):
    for pol in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
        yield np.asarray(pol[0])


def deslocamentos():
    """Quanto mover cada região para fora do centro do país, no mapa explodido (centro pelas UFs)."""
    pts = {}
    for f in UF:
        r = CODIGO_REGIAO[f["properties"]["codarea"][0]]
        pts.setdefault(r, []).extend(aneis(f["geometry"]))
    return {r: AFASTAMENTO * (np.vstack(a).mean(0) - CENTRO_BRASIL) for r, a in pts.items()}


def mapa_explodido(ax, grupos, valida):
    """Brasil com as 5 regiões afastadas umas das outras; as UFs e as âncoras vão junto."""
    desl = deslocamentos()
    centros = {}
    for f in UF:
        regiao = CODIGO_REGIAO[f["properties"]["codarea"][0]]
        d = desl[regiao]
        for anel in aneis(f["geometry"]):
            ax.fill(anel[:, 0] + d[0], anel[:, 1] + d[1], color=FORA, lw=0, zorder=0)
            ax.plot(anel[:, 0] + d[0], anel[:, 1] + d[1], color=CINZA, lw=0.4, zorder=1)
            centros.setdefault(regiao, []).append(anel + d)
    for regiao, a in centros.items():   # sigla do lado de fora, no lado oposto ao centro do país
        pts = np.vstack(a)
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        lado = LADO_SIGLA[regiao]
        x, ha = (x0 - 1.0, "right") if lado == "esquerda" else (x1 + 1.0, "left")
        ax.text(x, (y0 + y1) / 2, SIGLA[regiao], fontsize=7, fontweight="bold", color=TINTA, ha=ha,
                va="center", zorder=5)
    dx = grupos.regiao.map(lambda r: desl[r][0]).to_numpy()
    dy = grupos.regiao.map(lambda r: desl[r][1]).to_numpy()
    x, y = grupos.longitude.to_numpy() + dx, grupos.latitude.to_numpy() + dy
    ax.scatter(x[~valida], y[~valida], s=4, color=TREINO, lw=0, zorder=3)
    ax.scatter(x[valida], y[valida], s=4, color=VALID, lw=0, zorder=4)
    ax.set(aspect="equal", xticks=[], yticks=[], xlim=(-86, -21), ylim=(-41, 11))
    for s_ in ax.spines.values():
        s_.set_visible(False)


def mapa(ax, g, valida, extensao, ambos=False):
    """Âncoras de treino em azul e validadas em laranja; `ambos`: todas fazem os dois papéis."""
    contorno_ufs(ax)
    if ambos:
        ax.scatter(g.longitude, g.latitude, s=30, color=TREINO, edgecolor=VALID, lw=1.6, zorder=3)
    else:
        t, v = g[~valida], g[valida]
        ax.scatter(t.longitude, t.latitude, s=24, color=TREINO, edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(v.longitude, v.latitude, s=34, color=VALID, edgecolor="white", lw=0.6, zorder=4)
    ax.set(aspect="equal", xticks=[], yticks=[], **extensao)
    for s in ax.spines.values():
        s.set_visible(False)


def linha_do_tempo(ax, ini, fim):
    """Anos de treino, o bloco validado, os anos não usados e o teste reservado."""
    partes = [(ANO_INI, ini, TREINO, None), (ini, fim + 1, VALID, None),
              (fim + 1, ANO_TESTE, FORA, None), (ANO_TESTE, ANO_FIM + 1, "white", "////")]
    for a, b, cor, hachura in partes:
        if b > a:
            ax.barh(0, b - a, left=a, height=0.6, color=cor, hatch=hachura,
                    edgecolor=CINZA if hachura else "white", lw=0.8 if hachura else 2)
    marcas = [ANO_INI, ini]   # o fim do bloco e o início do teste ficam na legenda
    ax.set(xlim=(ANO_INI, ANO_FIM + 1), ylim=(-0.5, 0.5), yticks=[], xticks=marcas)
    ax.set_xticklabels([str(m) for m in marcas], fontsize=7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


LEGENDA = [Patch(color=TREINO, label="treino"), Patch(color=VALID, label="validação"),
           Patch(color=FORA, label="anos não usados no bloco"),
           Patch(facecolor="white", edgecolor=CINZA, hatch="////", label="teste 2015–2025 (reservado)")]


def estrategia_sul(grupos):
    g = grupos[grupos.regiao == "sul"]
    fig = plt.figure(figsize=(9.5, 3.6))
    sub = fig.add_gridspec(2, len(BLOCOS), height_ratios=[6, 0.8], hspace=0.05, wspace=0.05)
    for i, (ini, fim) in enumerate(BLOCOS):
        ax = fig.add_subplot(sub[0, i])
        mapa(ax, g, (g.grupo == i).to_numpy(), EXTENSAO_SUL)
        n_v = int((g.grupo == i).sum())
        ax.set_title(f"Bloco {i + 1}, {ini} a {fim}\ngrupo {i} ({n_v} âncoras) validado",
                     loc="left", fontsize=8)
        linha_do_tempo(fig.add_subplot(sub[1, i]), ini, fim)
    fig.legend(handles=LEGENDA, loc="lower center", ncol=4, frameon=False, labelcolor=TINTA,
               bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12)
    fig.savefig(FIGURAS / "estrategia_sul.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def temporal_vs_espacial(grupos):
    g = grupos[grupos.regiao == "sul"]
    ini, fim = BLOCOS[0]
    fig = plt.figure(figsize=(4.6, 3.6))
    sub = fig.add_gridspec(2, 2, height_ratios=[6, 0.8], hspace=0.05, wspace=0.08)
    textos = [
        ("Temporal", f"as {len(g)} âncoras treinam\ne as mesmas são validadas", True),
        ("Espaço-temporal", f"{int((g.grupo != 0).sum())} âncoras treinam\ne só o grupo 0 é validado", False),
    ]
    for j, (nome, texto, ambos) in enumerate(textos):
        ax = fig.add_subplot(sub[0, j])
        mapa(ax, g, (g.grupo == 0).to_numpy(), EXTENSAO_SUL, ambos=ambos)
        ax.set_title(f"{nome}\n{texto}", loc="left", fontsize=7)
        linha_do_tempo(fig.add_subplot(sub[1, j]), ini, fim)
    leg = [Patch(facecolor=TREINO, edgecolor=VALID, lw=1.6, label="treino e validação (mesma âncora)")]
    fig.legend(handles=leg + LEGENDA, loc="lower center", ncol=2, frameon=False, labelcolor=TINTA,
               bbox_to_anchor=(0.5, -0.12), fontsize=6.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12)
    fig.savefig(FIGURAS / "temporal_vs_espacial.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def mapa_grupos(grupos):
    fig, ax = plt.subplots(figsize=(4.6, 4.7))
    contorno_ufs(ax)
    for (regiao, grupo), g in grupos.groupby(["regiao", "grupo"]):
        cor = COR_REGIAO[regiao]
        xy = g[["longitude", "latitude"]].to_numpy()
        if len(xy) >= 3:
            h = ConvexHull(xy)
            anel = np.vstack([xy[h.vertices], xy[h.vertices[:1]]])
            ax.fill(anel[:, 0], anel[:, 1], color=cor, alpha=0.15, lw=0, zorder=2)
            ax.plot(anel[:, 0], anel[:, 1], color=cor, lw=1, zorder=2)
        ax.scatter(xy[:, 0], xy[:, 1], s=4, color=cor, lw=0, zorder=3)
        cx, cy = xy.mean(0)
        ax.text(cx, cy, str(grupo), ha="center", va="center", fontsize=6.5, fontweight="bold",
                color=TINTA, zorder=4,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=TINTA_2, lw=0.6))
    ax.set(xlim=(-75, -33), ylim=(-34.5, 6), aspect="equal", xlabel="longitude", ylabel="latitude")
    ax.legend(handles=[Patch(facecolor=cor, edgecolor=cor, alpha=0.6, label=NOMES_REGIAO[r])
                       for r, cor in COR_REGIAO.items()],
              loc="lower left", frameon=False, labelcolor=TINTA, title="região", title_fontsize=7,
              fontsize=7)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURAS / "grupos_espaciais.png", dpi=200)
    plt.close(fig)


def estrategia_global(grupos):
    """Teste espacial-temporal do modelo global: o modelo g treina sem o grupo g de cada região."""
    fig = plt.figure(figsize=(10, 3.5))
    sub = fig.add_gridspec(2, 5, height_ratios=[6, 0.7], hspace=0.05, wspace=0.05)
    for g in range(5):
        ax = fig.add_subplot(sub[0, g])
        mapa(ax, grupos, (grupos.grupo == g).to_numpy(), dict(xlim=(-75, -33), ylim=(-34.5, 6)))
        for col in ax.collections:   # pontos menores no Brasil inteiro
            col.set_sizes([5])
        n = int((grupos.grupo == g).sum())
        ax.set_title(f"Modelo {g + 1}\ngrupo {g} testado, {n} âncoras", loc="left", fontsize=8)
        ax_t = fig.add_subplot(sub[1, g])
        ax_t.barh(0, ANO_TESTE - ANO_INI, left=ANO_INI, height=0.6, color=TREINO, edgecolor="white", lw=2)
        ax_t.barh(0, ANO_FIM + 1 - ANO_TESTE, left=ANO_TESTE, height=0.6, color=VALID, edgecolor="white",
                  lw=2)
        ax_t.set(xlim=(ANO_INI, ANO_FIM + 1), ylim=(-0.5, 0.5), yticks=[], xticks=[ANO_INI, ANO_TESTE])
        ax_t.set_xticklabels([str(ANO_INI), str(ANO_TESTE)], fontsize=7)
        for s_ in ("top", "right", "left"):
            ax_t.spines[s_].set_visible(False)
    fig.legend(handles=[Patch(color=TREINO, label="treino, outras âncoras de 1980 a 2014"),
                        Patch(color=VALID, label="teste, âncoras do grupo de 2015 a 2025")],
               loc="lower center", ncol=2, frameon=False, labelcolor=TINTA, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12)
    fig.savefig(FIGURAS / "estrategia_global.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def estrategia_brasil(grupos):
    """Validação espacial-temporal do modelo completo: o bloco i valida o grupo i de todas as regiões.

    Os blocos de anos são os mesmos da figura do Sul (estrategia_sul.png), então a linha do tempo
    não se repete aqui.
    """
    fig, eixos = plt.subplots(1, len(BLOCOS), figsize=(9.5, 2.1), gridspec_kw={"wspace": 0.03})
    for i, ((ini, fim), ax) in enumerate(zip(BLOCOS, eixos)):
        mapa_explodido(ax, grupos, (grupos.grupo == i).to_numpy())
        for col in ax.collections:
            if col.get_sizes().size:
                col.set_sizes([2.5])
        n = int((grupos.grupo == i).sum())
        ax.set_title(f"Bloco {i + 1}, {ini} a {fim}\ngrupo {i} validado ({n} âncoras)", loc="left", fontsize=8)
    fig.legend(handles=LEGENDA[:2], loc="lower center", ncol=2, frameon=False, labelcolor=TINTA,
               bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.0)
    fig.savefig(FIGURAS / "estrategia_brasil.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    FIGURAS.mkdir(parents=True, exist_ok=True)
    grupos = grupos_espaciais()
    estrategia_sul(grupos)
    temporal_vs_espacial(grupos)
    mapa_grupos(grupos)
    estrategia_global(grupos)
    estrategia_brasil(grupos)
    print(f"figuras em {FIGURAS}/")
