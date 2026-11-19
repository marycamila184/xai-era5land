"""Explicabilidade local do modelo com grafo no teste: SHAP de quatro dias e dependência do SHAP.

Os modelos finais com grafo do Sul e do Nordeste (treino 1980–2014, hiperparâmetros de
resultados/cv_resumo.csv) preveem todos os dias fortes de 2015–2025. Os dias são escolhidos por
critério fixo, sem olhar o SHAP:

- acerto: entre os dias com chuva entre 85% e 100% do p90 previsto, o de maior chuva
  (Sul e Nordeste);
- maior evento: o dia de maior chuva da região no teste (Sul e Nordeste);

Figuras:
- shap_local_dias.png: a cascata do SHAP dos quatro dias, uma linha por região, do valor médio do modelo ao p90
  previsto, com a chuva observada;
- shap_dependencia.png: o SHAP de uma variável contra o valor dela, dia a dia, em todos os dias
  fortes do teste (convergência no Nordeste e desvio do tp no anel no Sul).

Saídas: av1/experimentos/figuras/ e resultados/shap_local_dias.csv.

Uso (na raiz do projeto):
    uv run python av1/experimentos/shap_local.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figuras_resultados import CINZA, FIGURAS, TINTA, TINTA_2, rotulo  # noqa: E402
from treinar_p90 import DADOS, FIXOS, LIMIAR_MM, SAIDA, bracos, hiperparametros  # noqa: E402

ANCORAS = Path("av1/amostra_brasil/dados_150/ancoras.csv")
SOBE, DESCE = "#e34948", "#2a78d6"   # polos da paleta divergente: aumenta e diminui o p90
N_BARRAS = 8
DEPENDENCIA = {"nordeste": "g_conv", "sul": "g_std_tp_mm"}
NOME = {"sul": "Sul", "nordeste": "Nordeste"}


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


def preparar(regiao):
    d = pd.read_parquet(DADOS / f"features_{regiao}.parquet")
    d = d[d.tp_mm >= LIMIAR_MM]
    cols = bracos(d.columns)["com_grafo"]
    ano = d.date.dt.year
    m = HistGradientBoostingRegressor(**hiperparametros(SAIDA / "cv_resumo.csv", regiao), **FIXOS)
    m.fit(d[ano <= 2014][cols], d[ano <= 2014].tp_mm)
    t = d[ano >= 2015].copy()
    t["p90"] = m.predict(t[cols])
    return t, cols, shap.TreeExplainer(m)


def escolher(t, caso):
    if caso == "acerto":
        a = t[(t.tp_mm <= t.p90) & (t.tp_mm >= 0.85 * t.p90)]
        return a.loc[a.tp_mm.idxmax()]
    if caso == "maior evento":
        return t.loc[t.tp_mm.idxmax()]
    mt, mp = t.tp_mm.median(), t.p90.median()   # típico
    return t.loc[((t.tp_mm - mt).abs() / mt + (t.p90 - mp).abs() / mp).idxmin()]


def cascata(ax, sv, base, linha, cols, titulo):
    """Do valor médio do modelo ao p90 previsto, com as N_BARRAS maiores contribuições."""
    ordem = np.argsort(-np.abs(sv))
    top, resto = ordem[:N_BARRAS], ordem[N_BARRAS:]
    nomes = [f"{rotulo(cols[k])} = {num(linha[cols[k]], 2)}" for k in top]
    valores = list(sv[top])
    if len(resto):
        nomes.append(f"demais {len(resto)} variáveis")
        valores.append(sv[resto].sum())
    nomes, valores = nomes[::-1], valores[::-1]   # a maior contribuição fica em cima
    acumulado = base + np.cumsum([0] + valores[:-1])
    lo = min(base, linha.p90, *(acumulado + np.array(valores)), *acumulado)
    hi = max(base, linha.p90, *(acumulado + np.array(valores)), *acumulado)
    folga = 0.12 * (hi - lo)
    pad = 0.015 * (hi - lo + 2 * folga)
    for i, (a, v) in enumerate(zip(acumulado, valores)):
        ax.barh(i, v, left=a, color=SOBE if v > 0 else DESCE, height=0.62, edgecolor="white", lw=1)
        ax.text(max(a, a + v) + pad, i, f"{'+' if v > 0 else ''}{num(v)}", va="center", ha="left",
                fontsize=7, color=TINTA)
    ax.set_yticks(range(len(nomes)), nomes, fontsize=7.5)
    ax.axvline(base, color=CINZA, lw=0.8, ls=":")
    # a linha do p90 para logo abaixo do rótulo da chuva, que fica no topo do painel
    ax.axvline(linha.p90, color=TINTA, lw=1.2, ymax=(len(nomes) + 1.2) / (len(nomes) + 1.9))
    topo = len(nomes) - 0.4
    xmin, xmax = lo - folga, hi + folga
    if xmin <= linha.tp_mm <= xmax:   # a chuva observada cabe no eixo
        xmin, xmax = min(xmin, linha.tp_mm - folga), max(xmax, linha.tp_mm + folga)
        ax.axvline(linha.tp_mm, color=TINTA_2, lw=1.2, ls="--")
        # o rótulo vai para o lado oposto ao p90, para não cruzar a linha dele
        esquerda = linha.tp_mm < linha.p90
        ax.text(linha.tp_mm, topo + 1.1, f"chuva {num(linha.tp_mm)} " if esquerda else f" chuva {num(linha.tp_mm)}",
                fontsize=7, color=TINTA_2, va="bottom", ha="right" if esquerda else "left")
    elif linha.tp_mm > xmax:           # muito acima: seta na borda direita
        ax.annotate(f"chuva {num(linha.tp_mm)} →", xy=(xmax, topo + 1.1), fontsize=7, color=TINTA_2,
                    ha="right", va="bottom")
    else:                              # muito abaixo: seta na borda esquerda
        ax.annotate(f"← chuva {num(linha.tp_mm)}", xy=(xmin, topo + 1.1), fontsize=7, color=TINTA_2,
                    ha="left", va="bottom")
    ax.text(base, topo, f" média {num(base)}", fontsize=7, color=TINTA_2, va="bottom")
    ax.text(linha.p90, topo + 0.55, f" p90 {num(linha.p90)}", fontsize=7, color=TINTA, va="bottom")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(nomes) + 1.3)
    ax.set_xlabel("mm")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(titulo, loc="left", fontsize=8.5)


if __name__ == "__main__":
    anc = pd.read_csv(ANCORAS).set_index("anchor_id")
    # uma linha por região
    casos = {(0, 0): ("sul", "acerto"), (0, 1): ("sul", "maior evento"),
             (1, 0): ("nordeste", "acerto"), (1, 1): ("nordeste", "maior evento")}
    regioes = {r: preparar(r) for r in ("sul", "nordeste")}

    fig, eixos = plt.subplots(2, 2, figsize=(9.5, 6.6))
    linhas = []
    for pos, (regiao, caso) in casos.items():
        ax = eixos[pos]
        t, cols, ex = regioes[regiao]
        linha = escolher(t, caso)
        sv = ex.shap_values(linha[cols].to_frame().T.astype(float))[0]
        base = float(np.ravel(ex.expected_value)[0])
        assert abs(base + sv.sum() - linha.p90) < 1e-3
        lugar = f"{anc.loc[linha.anchor_id, 'municipality']} ({anc.loc[linha.anchor_id, 'state']})"
        titulo = f"{NOME[regiao]}, {caso}\n{lugar}, {linha.date:%d/%m/%Y}"
        cascata(ax, sv, base, linha, cols, titulo)
        maiores = np.argsort(-np.abs(sv))[:3]
        linhas.append(dict(regiao=regiao, caso=caso, data=linha.date.date(), ancora=lugar,
                           chuva_mm=linha.tp_mm, p90_mm=linha.p90, base_mm=base,
                           maiores=", ".join(f"{cols[k]} {sv[k]:+.1f}" for k in maiores)))
    fig.tight_layout()
    fig.savefig(FIGURAS / "shap_local_dias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    dias = pd.DataFrame(linhas)
    dias.to_csv(SAIDA / "shap_local_dias.csv", index=False)
    print(dias.round(1).to_string(index=False))

    # dependência do SHAP
    fig, eixos = plt.subplots(1, 2, figsize=(4.8, 2.5))
    for ax, (regiao, var) in zip(eixos, DEPENDENCIA.items()):
        t, cols, ex = regioes[regiao]
        amostra = t
        sv = ex.shap_values(amostra[cols])[:, cols.index(var)]
        x = amostra[var].to_numpy()
        ax.scatter(x, sv, s=2, color=DESCE, alpha=0.15, lw=0, rasterized=True)
        bordas = np.quantile(x, np.linspace(0, 1, 16))
        meio = [(bordas[i] + bordas[i + 1]) / 2 for i in range(15)]
        mediana = [np.median(sv[(x >= bordas[i]) & (x <= bordas[i + 1])]) for i in range(15)]
        ax.plot(meio, mediana, color=TINTA, lw=1.6)
        ax.axhline(0, color=CINZA, lw=0.8)
        ax.set_xlabel(rotulo(var), fontsize=7)
        ax.set_ylabel("SHAP (mm)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"{NOME[regiao]}, {len(amostra):,} dias".replace(",", "."), loc="left", fontsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURAS / "shap_dependencia.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figuras em {FIGURAS}/")
