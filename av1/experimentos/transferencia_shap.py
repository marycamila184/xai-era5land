"""Queda na transferência (slide 10 da proposta): as variáveis usadas na região de treino e na
de teste, e a relação disso com a perda de desempenho ao transferir.

Os modelos regionais com grafo (treino 1980–2014, hiperparâmetros de resultados/cv_resumo.csv)
são explicados com SHAP em todos os dias fortes de 2015–2025 de cada região, inclusive as que
não são a deles. O perfil de um modelo numa região é a média do |SHAP| de cada variável.

- similaridade entre regiões: correlação de Spearman entre o perfil do modelo de X na própria X
  e o do modelo de Y na própria Y (os dois modelos usam as mesmas variáveis?);
- queda: ganho do modelo de Y em Y (diagonal da matriz) menos o ganho do modelo de X em Y
  (resultados/matriz_regioes.csv), em pontos percentuais;
- perda de cobertura: cobertura do modelo de Y em Y menos a do modelo de X em Y, em pontos.

Na figura as duas entram com o sinal trocado, como diferença do modelo transferido para o da
região de teste: negativo é pior que o modelo local, e uma "queda negativa" não aparece.

A similaridade explicou pouco da queda (Spearman −0,30, p = 0,21) e saiu do artigo em
22/10/2026; a perda de cobertura explica bem mais (0,78). As duas ficam no csv.

Figuras, uma coluna cada:
- transferencia_pares.png: diferença de cobertura contra diferença de ganho nos 20 pares;
- transferencia_casos.png: (a) a pior e (b) a melhor transferência (maior e menor queda), com o
  perfil do modelo transferido ao lado do perfil do modelo da região de teste.

Saídas: resultados/transferencia_perfis.csv, resultados/transferencia_pares.csv e a figura.
O SHAP só é recalculado com --recalcular ou se transferencia_perfis.csv não existir.

Uso (na raiz do projeto, depois de matriz_regioes.py):
    uv run python av1/experimentos/transferencia_shap.py [--recalcular]
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figuras_resultados import CINZA, FIGURAS, TINTA, TINTA_2, nome_extenso  # noqa: E402
from treinar_p90 import (DADOS, FIXOS, LIMIAR_MM, REGIOES, SAIDA, bracos,  # noqa: E402
                         hiperparametros)

SIGLA = {"norte": "N", "nordeste": "NE", "centro_oeste": "CO", "sudeste": "SE", "sul": "S"}
NOME = {"norte": "Norte", "nordeste": "Nordeste", "centro_oeste": "Centro-Oeste",
        "sudeste": "Sudeste", "sul": "Sul"}
COR_TREINO, COR_NATIVO = "#eb6834", "#2a78d6"
N_BARRAS = 8
PERFIS = SAIDA / "transferencia_perfis.csv"


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


def calcular_perfis():
    t0 = time.time()
    dados = {}
    for r in REGIOES:
        d = pd.read_parquet(DADOS / f"features_{r}.parquet")
        dados[r] = d[d.tp_mm >= LIMIAR_MM]
    cols = bracos(dados[REGIOES[0]].columns)["com_grafo"]
    perfis = []
    for x in REGIOES:
        dx = dados[x]
        m = HistGradientBoostingRegressor(**hiperparametros(SAIDA / "cv_resumo.csv", x), **FIXOS)
        m.fit(dx[dx.date.dt.year <= 2014][cols], dx[dx.date.dt.year <= 2014].tp_mm)
        ex = shap.TreeExplainer(m)
        for y in REGIOES:
            dy = dados[y]
            teste = dy[dy.date.dt.year >= 2015]
            a = np.abs(ex.shap_values(teste[cols])).mean(0)
            perfis += [dict(treino=x, teste=y, feature=c, media_abs_shap_mm=v) for c, v in zip(cols, a)]
        print(f"modelo {x} ok ({time.time() - t0:.0f}s)", flush=True)
    perfis = pd.DataFrame(perfis)
    perfis.to_csv(PERFIS, index=False)
    return perfis


def comparar(p, g, cob):
    """Similaridade, queda de ganho e perda de cobertura para os 20 pares treino → teste."""
    linhas = []
    for x in REGIOES:
        for y in REGIOES:
            if x != y:
                linhas.append(dict(treino=x, teste=y,
                                   similaridade=spearmanr(p[(x, x)], p[(y, y)]).statistic,
                                   similaridade_aplicado=spearmanr(p[(x, y)], p[(y, y)]).statistic,
                                   queda_pp=g.loc[y, y] - g.loc[x, y],
                                   perda_cobertura_pp=cob.loc[y, y] - cob.loc[x, y]))
    return pd.DataFrame(linhas)


def barras(ax, p, g, x, y, letra, rotulo_caso):
    caso = pd.DataFrame({"aplicado": p[(x, y)], "nativo": p[(y, y)]})
    caso = caso.loc[caso.max(axis=1).sort_values(ascending=False).index[:N_BARRAS]][::-1]
    pos = np.arange(len(caso))
    ax.barh(pos + 0.2, caso.nativo, height=0.38, color=COR_NATIVO, label=f"modelo do {NOME[y]}")
    ax.barh(pos - 0.2, caso.aplicado, height=0.38, color=COR_TREINO,
            label=f"modelo do {NOME[x]} aplicado")
    ax.set_yticks(pos, [nome_extenso(c) for c in caso.index], fontsize=7.5)
    ax.set_xlabel(f"média do |SHAP| no {NOME[y]} (mm)", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2,
              labelcolor=TINTA)
    ax.tick_params(axis="y", length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.set_title(f"({letra}) {rotulo_caso}, {NOME[x]} → {NOME[y]}\nganho {num(g.loc[x, y])}% "
                 f"contra {num(g.loc[y, y])}% do modelo do {NOME[y]}", loc="left", fontsize=8)

if __name__ == "__main__":
    perfis = calcular_perfis() if ("--recalcular" in sys.argv or not PERFIS.exists()) else pd.read_csv(PERFIS)
    p = perfis.pivot_table(index="feature", columns=["treino", "teste"], values="media_abs_shap_mm")
    mat = pd.read_csv(SAIDA / "matriz_regioes.csv")
    g = 100 * mat[mat.braco == "com_grafo"].pivot(index="treino", columns="teste", values="ganho")
    cob = 100 * mat[mat.braco == "com_grafo"].pivot(index="treino", columns="teste", values="cobertura")
    pares = comparar(p, g, cob)
    pares.to_csv(SAIDA / "transferencia_pares.csv", index=False)
    rho_sim = spearmanr(pares.similaridade, pares.queda_pp).statistic
    rho = spearmanr(pares.perda_cobertura_pp, pares.queda_pp).statistic
    pior = pares.loc[pares.queda_pp.idxmax()]
    melhor = pares.loc[pares.queda_pp.idxmin()]
    print(pares.round(2).to_string(index=False))
    print(f"Spearman similaridade x queda: {rho_sim:.2f}; perda de cobertura x queda: {rho:.2f}")
    print(f"pior {pior.treino}→{pior.teste} ({pior.queda_pp:.1f}), melhor {melhor.treino}→{melhor.teste} "
          f"({melhor.queda_pp:.1f})")

    fig, ax_a = plt.subplots(figsize=(4.6, 3.4))
    dx, dy = -pares.perda_cobertura_pp, -pares.queda_pp   # transferido menos local
    ax_a.scatter(dx, dy, s=36, color=COR_NATIVO, edgecolor="white", lw=0.8, zorder=3)
    # só os pares citados no texto ganham rótulo; os demais se amontoam perto de zero
    citados = {(pior.treino, pior.teste), (melhor.treino, melhor.teste), ("sul", "nordeste")}
    for _, r in pares.iterrows():
        if r.queda_pp < 10 and (r.treino, r.teste) not in citados:
            continue
        # perto de zero os pontos se amontoam: esses rótulos saem para baixo ou para cima
        desvio = {("sudeste", "centro_oeste"): (8, 4), ("sul", "nordeste"): (6, -11)}
        ax_a.annotate(f"{SIGLA[r.treino]}→{SIGLA[r.teste]}", (-r.perda_cobertura_pp, -r.queda_pp),
                      xytext=desvio.get((r.treino, r.teste), (8, 3)), textcoords="offset points",
                      fontsize=6.5, color=TINTA_2,
                      ha="left")
    for r in (pior, melhor):   # destaca os dois casos da figura dos casos
        ax_a.scatter([-r.perda_cobertura_pp], [-r.queda_pp], s=120, facecolor="none", edgecolor=COR_TREINO,
                     lw=1.6, zorder=4)
    ax_a.axhline(0, color=CINZA, lw=0.8)
    ax_a.axvline(0, color=CINZA, lw=0.8)
    ax_a.set(xlabel="cobertura: transferido − local (p.p.)",
             ylabel="ganho: transferido − local (p.p.)")
    ax_a.xaxis.set_major_formatter(lambda v, _: num(v, 0))
    ax_a.set_title(f"Pares treino → teste. Correlação de Spearman {num(rho, 2)}", loc="left",
                   fontsize=8.5)
    for lado in ("top", "right"):
        ax_a.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURAS / "transferencia_pares.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, (ax_b, ax_c) = plt.subplots(2, 1, figsize=(4.6, 6.4))
    barras(ax_b, p, g, pior.treino, pior.teste, "a", "Pior transferência")
    barras(ax_c, p, g, melhor.treino, melhor.teste, "b", "Melhor transferência")
    fig.tight_layout(h_pad=2)
    fig.savefig(FIGURAS / "transferencia_casos.png", dpi=200, bbox_inches="tight")
    print(f"figura em {FIGURAS}/")
