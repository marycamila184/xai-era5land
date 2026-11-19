"""Figuras de conferência da amostragem.

- mapa_ancoras.png: as âncoras por região e as vizinhas, com as UFs (IBGE).
- altitude_ancoras_vs_sedes.png: por região, a altitude da célula das âncoras contra a de
  todas as sedes. Imprime também os quartis das duas.

Vão para a pasta de figuras do anel (figuras/ no de 30 km, figuras_150/ no de 150 km).

Uso (na raiz do projeto, depois de escolher_ancoras.py):
    uv run python av1/amostra_brasil/figuras_amostra.py               # anel de 30 km
    uv run python av1/amostra_brasil/figuras_amostra.py --anel 150km
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz, para av1.amostra_brasil
from av1.amostra_brasil.ancoras import (ESPACAMENTO, anel_da_linha_de_comando, baixar_json,  # noqa: E402
                                         distancia_km)

ANEL, CONFIG = anel_da_linha_de_comando()
DADOS = CONFIG["dados"]
FIGURAS = CONFIG["figuras"]
UFS = Path("av1/amostra_brasil/dados/ibge/ufs.json")
URL_UFS = ("https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
           "?formato=application/vnd.geo+json&intrarregiao=UF&qualidade=minima")
# paleta categórica validada (dataviz), em ordem fixa
COR = {"Norte": "#2a78d6", "Nordeste": "#eb6834", "Sudeste": "#1baf7a", "Sul": "#eda100",
       "Centro-Oeste": "#e87ba4"}
TINTA, TINTA_2, CINZA = "#1f1f1e", "#5f5e5a", "#b4b2a9"

anc = pd.read_csv(DADOS / "ancoras.csv")
cel = pd.read_csv(DADOS / "celulas.csv")
sedes = pd.read_csv(DADOS / "sedes.csv")
FIGURAS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": CINZA, "axes.labelcolor": TINTA_2,
                     "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlecolor": TINTA})

# Mapa
if not UFS.exists():
    UFS.parent.mkdir(parents=True, exist_ok=True)
    UFS.write_text(json.dumps(baixar_json(URL_UFS)))
fig, ax = plt.subplots(figsize=(8, 8.4))
for f in json.loads(UFS.read_text())["features"]:
    g = f["geometry"]
    for pol in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
        anel = np.asarray(pol[0])
        ax.plot(anel[:, 0], anel[:, 1], color=CINZA, lw=0.6, zorder=1)
viz = cel[cel.position != "center"]
ax.scatter(viz.longitude, viz.latitude, s=2, color=TINTA_2, alpha=0.5, lw=0, zorder=2,
           label=f"vizinhas ({viz.dist_km.min():.0f} a {viz.dist_km.max():.0f} km)")
for regiao, cor in COR.items():
    a = anc[anc.region == regiao]
    ax.scatter(a.longitude, a.latitude, s=22, color=cor, edgecolor="white", lw=0.6, zorder=3,
               label=f"{regiao} ({len(a)})")
ax.set(xlim=(-75, -33), ylim=(-34.5, 6), aspect="equal", xlabel="longitude", ylabel="latitude")
# menor distância real entre âncoras: a regra é em células (Chebyshev), e 0,1° de longitude encolhe
# com a latitude
d = distancia_km(anc.latitude.values[:, None], anc.longitude.values[:, None],
                 anc.latitude.values[None], anc.longitude.values[None])
np.fill_diagonal(d, np.inf)
ax.set_title(f"{len(anc)} âncoras (sedes municipais) a pelo menos {ESPACAMENTO} células uma da outra "
             f"({d.min():.0f} km ou mais)", loc="left")
ax.legend(loc="lower left", frameon=False, labelcolor=TINTA)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURAS / "mapa_ancoras.png", dpi=150)
plt.close(fig)

# Altitude: âncoras contra todas as sedes, por região
fig, eixos = plt.subplots(1, 5, figsize=(12, 3.4), sharey=True)
# bordas múltiplas de 100 que cobrem todas as sedes (há células com altitude negativa no litoral)
bins = np.arange(np.floor(sedes.altitude.min() / 100) * 100, sedes.altitude.max() + 100, 100)
linhas = []
for ax, regiao in zip(eixos, COR):
    s = sedes[sedes.region == regiao].altitude
    a = anc[anc.region == regiao].altitude
    ax.hist(s, bins=bins, density=True, color=CINZA, label=f"todas as sedes ({len(s)})")
    ax.hist(a, bins=bins, density=True, histtype="step", color="#2a78d6", lw=2, label=f"âncoras ({len(a)})")
    ax.set_title(regiao, loc="left")
    ax.set_xlabel("altitude da célula (m)")
    ax.legend(frameon=False, fontsize=7, labelcolor=TINTA)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    q = lambda x: x.quantile([0.25, 0.5, 0.75]).round(0).astype(int).tolist()  # noqa: E731
    linhas.append({"região": regiao, "sedes q25/med/q75": q(s), "âncoras q25/med/q75": q(a)})
eixos[0].set_ylabel("densidade")
fig.suptitle("Altitude: as âncoras representam o relevo das sedes de cada região?", x=0.01, ha="left",
             color=TINTA)
fig.tight_layout()
fig.savefig(FIGURAS / "altitude_ancoras_vs_sedes.png", dpi=150)
plt.close(fig)
print(pd.DataFrame(linhas).to_string(index=False))
print(f"figuras em {FIGURAS}/")
