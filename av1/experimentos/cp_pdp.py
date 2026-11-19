"""Ceteris paribus (CP) dos quatro dias da cascata do SHAP, com o PDP da região no mesmo gráfico.

Para cada dia escolhido em shap_local.py (mesmo critério fixo), duas variáveis de grafo variam
uma de cada vez, com as demais fixas no valor daquele dia, e o modelo com grafo da região prevê
o p90 em cada valor (CP). No mesmo painel, o PDP é a média dessas curvas em todos os dias
fortes de 2015–2025 da região. A grade de valores vai do percentil 1 ao 99 da variável no teste.

- Sul (acerto e maior evento): desvio da chuva no anel e velocidade do vento no anel.
- Nordeste (acerto e maior evento): chuva média do anel e convergência.

Saída: av1/experimentos/figuras/cp_pdp_dias.png.

Uso (na raiz do projeto):
    uv run python av1/experimentos/cp_pdp.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figuras_resultados import CINZA, FIGURAS, TINTA, TINTA_2  # noqa: E402
from shap_local import ANCORAS, NOME, escolher, num  # noqa: E402
from treinar_p90 import DADOS, FIXOS, LIMIAR_MM, SAIDA, bracos, hiperparametros  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

CASOS = [("sul", "acerto"), ("sul", "maior evento"), ("nordeste", "acerto"), ("nordeste", "maior evento")]
VARIAVEIS = {"sul": ["g_std_tp_mm", "g_mean_wind_speed"], "nordeste": ["g_mean_tp_mm", "g_conv"]}
COR_CP, COR_DIA = "#2a78d6", "#eb6834"
NOME_VAR = {"g_std_tp_mm": ("o", "desvio da chuva no anel", "mm"),
            "g_mean_wind_speed": ("a", "velocidade média do vento no anel", "m/s"),
            "g_mean_tp_mm": ("a", "chuva média no anel", "mm"),
            "g_conv": ("a", "convergência do vento no anel", "10⁻⁵ s⁻¹")}
N_PONTOS = 40


def modelo(regiao):
    d = pd.read_parquet(DADOS / f"features_{regiao}.parquet")
    d = d[d.tp_mm >= LIMIAR_MM]
    cols = bracos(d.columns)["com_grafo"]
    ano = d.date.dt.year
    m = HistGradientBoostingRegressor(**hiperparametros(SAIDA / "cv_resumo.csv", regiao), **FIXOS)
    m.fit(d[ano <= 2014][cols], d[ano <= 2014].tp_mm)
    t = d[ano >= 2015].copy()
    t["p90"] = m.predict(t[cols])
    return m, t, cols


def curvas(m, t, cols, linha, var):
    """Grade de valores, CP do dia e PDP da região."""
    grade = np.quantile(t[var], np.linspace(0.01, 0.99, N_PONTOS))
    dia = pd.DataFrame([linha[cols].astype(float)] * N_PONTOS)
    dia[var] = grade
    cp = m.predict(dia[cols])
    x = t[cols].to_numpy(copy=True)
    k = cols.index(var)
    pdp = []
    for v in grade:
        x[:, k] = v
        pdp.append(m.predict(x).mean())
    return grade, cp, np.array(pdp)


if __name__ == "__main__":
    from matplotlib.lines import Line2D
    anc = pd.read_csv(ANCORAS).set_index("anchor_id")
    modelos = {r: modelo(r) for r in ("sul", "nordeste")}
    # uma coluna por dia; nas duas linhas, as duas variáveis da região do dia
    fig, eixos = plt.subplots(2, 4, figsize=(10.5, 5.4), sharey="col")
    for k, (regiao, caso) in enumerate(CASOS):
        m, t, cols = modelos[regiao]
        linha = escolher(t, caso)
        lugar = f"{anc.loc[linha.anchor_id, 'municipality']} ({anc.loc[linha.anchor_id, 'state']})"
        eixos[0, k].set_title(f"{NOME[regiao]}, {caso}\n{lugar}, {linha.date:%d/%m/%Y}\n"
                              f"p90 previsto {num(linha.p90)} mm, chuva {num(linha.tp_mm)} mm",
                              loc="left", fontsize=7.5)
        for j, var in enumerate(VARIAVEIS[regiao]):
            ax = eixos[j, k]
            grade, cp, pdp = curvas(m, t, cols, linha, var)
            ax.plot(grade, pdp, color=TINTA_2, lw=1.5, ls="--")
            ax.plot(grade, cp, color=COR_CP, lw=2.2)
            ax.scatter([linha[var]], [linha.p90], s=55, color=COR_DIA, edgecolor="white", lw=0.8, zorder=4)
            artigo, nome, unidade = NOME_VAR[var]
            ax.set_xlabel(f"{nome}, t−1 ({unidade})", fontsize=7)
            if k == 0:
                ax.set_ylabel("p90 previsto (mm)", fontsize=8)
            ax.grid(axis="y", color=CINZA, lw=0.4, alpha=0.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.tick_params(labelsize=6.5)
    legenda = [Line2D([], [], color=COR_CP, lw=2.2,
                      label="CP do dia"),
               Line2D([], [], color=TINTA_2, lw=1.5, ls="--",
                      label="PDP da região"),
               Line2D([], [], color=COR_DIA, marker="o", ls="", markersize=7, markeredgecolor="white",
                      label="valor do dia")]
    fig.legend(handles=legenda, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               labelcolor=TINTA, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURAS / "cp_pdp_dias.png", dpi=200, bbox_inches="tight")
    print(f"figura em {FIGURAS}/")
