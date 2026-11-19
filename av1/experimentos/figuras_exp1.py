"""Figuras de explicabilidade do experimento 1 (floresta aleatória do Vitor, no estilo do artigo).

Os números vêm dos scripts em R da pasta "av1/R atualizado pdp e ice" (obs_interesse.R, shap.R e
cp_e_pdp.R), rodados com o modelo da grade reduzida (modelo_rf_grade_reduzida.rds) e salvos em
resultados/exp1/. Diferenças em relação aos gráficos originais:

- o SHAP de cada variável é a soma nas oito vizinhas, e não a média, para que as barras somem a
  diferença entre a previsão e o valor médio do modelo;
- o PDP usa uma amostra de 5000 linhas do treino, com semente fixa.

Figuras:
- exp1_shap_local.png: SHAP das quatro observações, somado por variável;
- exp1_shap_nos.png: participação de cada vizinha na soma do |SHAP|, uma barra empilhada por dia;
- exp1_cp_pdp.png: CP das quatro observações e PDP, variando a variável nas oito vizinhas juntas.

Uso (na raiz do projeto):
    uv run python av1/experimentos/figuras_exp1.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figuras_resultados import CINZA, FIGURAS, TINTA, TINTA_2  # noqa: E402
from shap_local import DESCE, SOBE, num  # noqa: E402

DADOS = Path("av1/experimentos/resultados/exp1")
ANCORAS = Path("av1/amostra_brasil/dados_150/ancoras.csv")
VARIAVEIS = {"tp_mm": ("chuva nas vizinhas", "mm"), "ssrd_wm2": ("radiação solar", "W m⁻²"),
             "pev_mm": ("evaporação potencial", "mm"), "t2m": ("temperatura", "°C"),
             "d2m": ("ponto de orvalho", "°C"), "u10": ("vento leste-oeste", "m/s"),
             "v10": ("vento norte-sul", "m/s"), "wind_res": ("vento resultante", "m/s")}
NOS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
NOME_NO = dict(zip(NOS, ["N", "NE", "L", "SE", "S", "SO", "O", "NO"]))
TIPOS = ["Maior precipitação", "Maior superestimação", "Menor erro", "Menor precipitação"]
COR_TIPO = dict(zip(TIPOS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))


def titulo(o, anc):
    a = anc.loc[o.anchor_id]
    return (f"{o.name}\n{a.municipality} ({a.state}), {pd.Timestamp(o.date):%d/%m/%Y}\n"
            f"previsto {num(o.predito)} mm, observado {num(o.real)} mm")


def shap_local(shap, base, obs, anc):
    # cada painel tem a própria escala: a maior chuva tem contribuições muito maiores que as outras
    fig, eixos = plt.subplots(2, 2, figsize=(9, 6.4))
    for ax, tipo in zip(eixos.ravel(), TIPOS):
        o = obs.loc[tipo]
        s = pd.Series({v: shap.loc[tipo, [f"{v}_no_{n}" for n in NOS]].sum() for v in VARIAVEIS})
        lim = s.abs().max() * 1.35
        s = s.reindex(s.abs().sort_values().index)
        y = np.arange(len(s))
        ax.barh(y, s, height=0.62, color=[SOBE if v > 0 else DESCE for v in s])
        for yi, v in zip(y, s):
            ax.text(v + (0.02 if v >= 0 else -0.02) * lim, yi, f"{'+' if v > 0 else ''}{num(v, 2)}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=TINTA)
        ax.set_yticks(y, [VARIAVEIS[v][0] for v in s.index], fontsize=8)
        ax.axvline(0, color=CINZA, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.tick_params(axis="y", length=0)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.set_title(titulo(o, anc) + f"\nvalor médio do modelo {num(base)} mm", loc="left",
                     fontsize=8)
    fig.supxlabel("SHAP (mm), soma nas oito vizinhas, todas as variáveis do mesmo dia", fontsize=8.5,
                  color=TINTA_2)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(FIGURAS / "exp1_shap_local.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def shap_nos(shap, obs, anc):
    """Uma barra empilhada por dia, com a participação de cada vizinha (soma 100%).

    As vizinhas vêm agrupadas pelo eixo (leste e oeste, noroeste e sudeste, nordeste e sudoeste,
    norte e sul), com um tom mais forte e outro mais claro da mesma cor em cada eixo.
    """
    ordem = [("e", "#2a78d6"), ("w", "#7fb0ea"), ("nw", "#eb6834"), ("se", "#f4a882"),
             ("ne", "#1baf7a"), ("sw", "#82d3b4"), ("n", "#eda100"), ("s", "#f5cd73")]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for i, tipo in enumerate(TIPOS):
        part = pd.Series({n: shap.loc[tipo, [f"{v}_no_{n}" for v in VARIAVEIS]].abs().sum() for n in NOS})
        part = 100 * part / part.sum()
        esquerda = 0.0
        for n, cor in ordem:
            ax.barh(i, part[n], left=esquerda, color=cor, edgecolor="white", lw=1, height=0.62,
                    label=NOME_NO[n] if i == 0 else None)
            if part[n] >= 3.5:   # abaixo disso o número não cabe dentro do segmento
                ax.text(esquerda + part[n] / 2, i, f"{part[n]:.0f}", ha="center", va="center",
                        fontsize=6 if part[n] >= 7 else 5.2, color=TINTA)
            esquerda += part[n]
    o = obs.loc[TIPOS]
    ax.set_yticks(range(len(TIPOS)), [f"{t}\n{anc.loc[a, 'municipality']} ({anc.loc[a, 'state']})"
                                      for t, a in zip(TIPOS, o.anchor_id)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("participação de cada vizinha no |SHAP| (%)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.legend(ncol=8, loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=6.5,
              labelcolor=TINTA, handlelength=1.0, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "exp1_shap_nos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def cp_pdp(cp, pdp, val):
    fig, eixos = plt.subplots(2, 4, figsize=(11, 5.4))
    for ax, (var, (nome, unidade)) in zip(eixos.ravel(), VARIAVEIS.items()):
        for tipo in TIPOS:
            c = cp[(cp.variavel == var) & (cp.tipo == tipo)]
            ax.plot(c.valor, c.predicao, color=COR_TIPO[tipo], lw=1.6)
            p = val[(val.variavel == var) & (val.tipo == tipo)]
            ax.scatter(p.valor, p.predicao, s=28, color=COR_TIPO[tipo], edgecolor="white", lw=0.7,
                       zorder=4)
        d = pdp[pdp.variavel == var]
        ax.plot(d.valor, d.predicao, color=TINTA_2, lw=1.5, ls="--")
        ax.set_title(nome, loc="left", fontsize=9)
        ax.set_xlabel(f"valor nas oito vizinhas, mesmo dia ({unidade})", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color=CINZA, lw=0.4, alpha=0.5)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    for ax in eixos[:, 0]:
        ax.set_ylabel("chuva estimada na âncora (mm)", fontsize=8)
    legenda = [Line2D([], [], color=COR_TIPO[t], lw=1.6, label=f"CP, {t.lower()}") for t in TIPOS]
    legenda.append(Line2D([], [], color=TINTA_2, lw=1.5, ls="--", label="PDP"))
    fig.legend(handles=legenda, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               frameon=False, fontsize=8, labelcolor=TINTA)
    fig.tight_layout(h_pad=1.4)
    fig.savefig(FIGURAS / "exp1_cp_pdp.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    anc = pd.read_csv(ANCORAS).set_index("anchor_id")
    obs = pd.read_csv(DADOS / "exp1_obs.csv").set_index("tipo")
    shap = pd.read_csv(DADOS / "exp1_shap.csv").set_index("tipo")
    base = float(pd.read_csv(DADOS / "exp1_shap_base.csv").baseline.iloc[0])
    shap_local(shap, base, obs, anc)
    shap_nos(shap, obs, anc)
    cp_pdp(pd.read_csv(DADOS / "exp1_cp.csv"), pd.read_csv(DADOS / "exp1_pdp.csv"),
           pd.read_csv(DADOS / "exp1_obs_valores.csv"))
    print(f"figuras em {FIGURAS}/")
