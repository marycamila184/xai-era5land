"""Figuras dos resultados: a matriz entre regiões e o SHAP no teste, sem e com grafo.

- matriz_regioes.png: ganho sobre a climatologia (treino na linha, teste na coluna) sem grafo,
  com grafo e a diferença (resultados/matriz_regioes.csv).
- shap_teste_grupos.png: participação de cada grupo de features no |SHAP| do modelo com grafo,
  por região (slide 10), e uma barra do modelo completo no Brasil todo (shap_global.py).
- shap_teste_features.png: as seis features mais usadas em cada região, sem grafo (em cima) e
  com grafo (embaixo), em barras coloridas pelo grupo (resultados/shap_teste_features.csv).

Cores da paleta validada (dataviz): rampa azul para magnitude, azul ↔ vermelho com cinza no
meio para valores com sinal, e as cinco primeiras cores categóricas para os grupos (vizinhas na
barra empilhada, combinação que passa no teste de daltonismo).

Saída: av1/experimentos/figuras/.

Uso (na raiz do projeto, depois de matriz_regioes.py e shap_teste.py):
    uv run python av1/experimentos/figuras_resultados.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablacao import grupo  # noqa: E402
from treinar_p90 import REGIOES, SAIDA  # noqa: E402

FIGURAS = Path("av1/experimentos/figuras")
NOME = {"norte": "Norte", "nordeste": "Nordeste", "centro_oeste": "Centro-Oeste",
        "sudeste": "Sudeste", "sul": "Sul"}
TINTA, TINTA_2, CINZA = "#1f1f1e", "#5f5e5a", "#b4b2a9"
AZUL = LinearSegmentedColormap.from_list(
    "azul", ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# rampa neutra para a intensidade no mapa do SHAP, para a cor ficar só com os grupos
CINZA_RAMPA = LinearSegmentedColormap.from_list(
    "cinza", ["#f7f6f3", "#dddbd5", "#b4b2a9", "#87857e", "#5f5e5a", "#3a3936", "#1f1f1e"])
DIVERGENTE = LinearSegmentedColormap.from_list(
    "div", ["#a32b2a", "#e34948", "#f3b1b0", "#f0efec", "#9ec5f4", "#3987e5", "#1c5cab"])
GRUPOS = {"local": ("âncora", "#2a78d6"), "tp_anel": ("chuva no anel", "#eb6834"),
          "agregados": ("outras variáveis no anel", "#1baf7a"), "gradientes": ("gradientes", "#eda100"),
          "cinematica": ("convergência e vorticidade", "#e87ba4")}
# nomes por extenso para o mapa de calor do SHAP (o grupo vai na cor, não no nome)
VAR_EXTENSO = {"tp_mm": "chuva", "ssrd_wm2": "radiação solar", "pev_mm": "evaporação potencial",
               "t2m": "temperatura", "t2m_min": "temperatura mínima", "t2m_max": "temperatura máxima",
               "d2m": "ponto de orvalho", "dpd": "depressão do orvalho",
               "wind_speed": "vento", "wind_max": "vento máximo", "wind_res": "vento resultante",
               "wind_dir_sin": "direção do vento (seno)", "wind_dir_cos": "direção do vento (cosseno)",
               "u10": "vento leste-oeste", "v10": "vento norte-sul"}
DA = {"t2m": "da temperatura", "d2m": "do ponto de orvalho"}
VAR = {"tp_mm": "tp", "ssrd_wm2": "ssrd", "pev_mm": "pev", "wind_speed": "vento",
       "wind_max": "vento máx.", "wind_res": "vento result.", "wind_dir_sin": "dir. vento (sen)",
       "wind_dir_cos": "dir. vento (cos)", "t2m_min": "t2m mín.", "t2m_max": "t2m máx."}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": CINZA, "axes.labelcolor": TINTA_2,
                     "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titlecolor": TINTA})


def rotulo(f):
    """Nome da feature com o dia de onde ela vem: a véspera (t−1), ou o próprio dia t no dia do ano."""
    return f"{_nome(f)} ({'t' if f in ('day_sin', 'day_cos') else 't−1'})"


def _nome(f):
    v = lambda s: VAR.get(s, s)  # noqa: E731
    if f.endswith("_d1"):
        return f"âncora: {v(f[:-3])}"
    if f == "day_sin":
        return "dia do ano (sen)"
    if f == "day_cos":
        return "dia do ano (cos)"
    if f == "g_conv":
        return "anel: convergência"
    if f == "g_vort":
        return "anel: vorticidade"
    if f == "g_max_tp_mm":
        return "anel: máximo de tp"
    if f == "g_frac1mm_tp_mm":
        return "anel: fração > 1 mm"
    for pre, nome in [("g_mean_", "anel: média de "), ("g_std_", "anel: desvio de "),
                      ("g_diff_", "anel − âncora: ")]:
        if f.startswith(pre):
            return nome + v(f[len(pre):])
    for pre, nome in [("g_gradx_", "gradiente leste-oeste de "), ("g_grady_", "gradiente norte-sul de "),
                      ("g_gradmag_", "|gradiente| de ")]:
        if f.startswith(pre):
            return nome + v(f[len(pre):])
    return f


def grupo_shap(f):
    return grupo(f) if f.startswith("g_") else "local"


def nome_extenso(f):
    """Nome por extenso, sem o dia (todas são da véspera, menos o dia do ano)."""
    v = lambda s: VAR_EXTENSO.get(s, s)  # noqa: E731
    if f.endswith("_d1"):
        return v(f[:-3])
    especiais = {"day_sin": "dia do ano (seno)", "day_cos": "dia do ano (cosseno)",
                 "g_conv": "convergência do vento", "g_vort": "vorticidade do vento",
                 "g_max_tp_mm": "chuva no anel (máximo)",
                 "g_frac1mm_tp_mm": "chuva no anel (fração > 1 mm)"}
    if f in especiais:
        return especiais[f]
    for pre, est in [("g_mean_", "média"), ("g_std_", "desvio"), ("g_diff_", "menos âncora")]:
        if f.startswith(pre):
            return f"{v(f[len(pre):])} no anel ({est})"
    for pre, nome in [("g_gradx_", "gradiente leste-oeste"), ("g_grady_", "gradiente norte-sul"),
                      ("g_gradmag_", "intensidade do gradiente")]:
        if f.startswith(pre):
            return f"{nome} {DA[f[len(pre):]]}"
    return f


FONTE_CELULA = 7


def celulas(ax, m, cmap, norm, fmt, destaque_diagonal=False, divergente=True):
    """Mapa de calor com o valor escrito; texto branco nas células escuras."""
    ax.imshow(m, cmap=cmap, norm=norm, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m[i, j]
            pos = float(norm(val))   # posição na rampa, de 0 a 1
            forte = abs(pos - 0.5) > 0.3 if divergente else pos > 0.6
            ax.text(j, i, fmt(val), ha="center", va="center", fontsize=FONTE_CELULA,
                    color="white" if forte else TINTA)
    if destaque_diagonal:
        for k in range(min(m.shape)):
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec=TINTA, lw=1.4))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def matriz():
    r = pd.read_csv(SAIDA / "matriz_regioes.csv")
    linhas = list(REGIOES)
    if (SAIDA / "global_teste.csv").exists():   # o modelo global entra como sexta linha
        r = pd.concat([r, pd.read_csv(SAIDA / "global_teste.csv")])
        linhas.append("global")
    m = {b: (100 * r[r.braco == b].pivot(index="treino", columns="teste", values="ganho")
             .loc[linhas, REGIOES]).to_numpy() for b in ("sem_grafo", "com_grafo")}
    dif = m["com_grafo"] - m["sem_grafo"]
    lim = np.abs(np.r_[m["sem_grafo"].ravel(), m["com_grafo"].ravel()]).max()
    lim_d = np.abs(dif).max()
    fig, eixos = plt.subplots(1, 3, figsize=(9.5, 3.6 + 0.4 * (len(linhas) - 5)))
    paineis = [("Sem grafo, ganho (%)", m["sem_grafo"], TwoSlopeNorm(0, -lim, lim)),
               ("Com grafo, ganho (%)", m["com_grafo"], TwoSlopeNorm(0, -lim, lim)),
               ("Diferença, com grafo menos sem grafo (p.p.)", dif, TwoSlopeNorm(0, -lim_d, lim_d))]
    nomes = [NOME[x] for x in REGIOES]
    for k, (ax, (titulo, val, norm)) in enumerate(zip(eixos, paineis)):
        celulas(ax, val, DIVERGENTE, norm, lambda x: f"{x:+.1f}", destaque_diagonal=True)
        ax.set_xticks(range(5), nomes, rotation=30, ha="right")
        ax.set_yticks(range(len(linhas)), [NOME.get(x, "Brasil (completo)") for x in linhas] if k == 0 else [])
        if len(linhas) > 5:   # separa o global dos regionais
            ax.axhline(4.5, color="white", lw=4)
        ax.set_xlabel("região de teste (2015–2025)")
        if k == 0:
            ax.set_ylabel("região de treino (1980–2014)")
        ax.set_title(titulo, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURAS / "matriz_regioes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def shap_grupos():
    """Uma barra por modelo regional e, separada abaixo, a do modelo completo no Brasil todo."""
    def tabela(arquivo, regioes):
        g = pd.read_csv(SAIDA / arquivo)
        return 100 * g[g.braco == "com_grafo"].pivot(index="regiao", columns="grupo",
                                                     values="participacao").loc[regioes]
    g = pd.concat([tabela("shap_global_grupos.csv", ["brasil"]),
                   tabela("shap_teste_grupos.csv", REGIOES[::-1])])
    y = np.r_[0, np.arange(1, len(g)) + 0.5]   # meio espaço entre o Brasil e as regiões
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    esquerda = np.zeros(len(g))
    for grupo, (nome, cor) in GRUPOS.items():
        v = g[grupo].to_numpy()
        ax.barh(y, v, left=esquerda, color=cor, edgecolor="white", lw=1.2, label=nome,
                height=0.62)
        for yi, x0, w in zip(y, esquerda, v):
            if w >= 5:
                ax.text(x0 + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if grupo == "local" else TINTA)
            else:   # pedaço estreito demais: o número vai logo depois da barra
                ax.text(x0 + w + 1, yi, f"{w:.0f}%", ha="left", va="center", fontsize=6.5,
                        color=TINTA_2)
        esquerda += v
    ax.set_yticks(y, [NOME.get(x, "Brasil (completo)") for x in g.index], fontsize=7)
    ax.set(xlim=(0, 100))
    ax.set_xlabel("participação na soma do |SHAP| (%)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.25, 1.0), frameon=False, fontsize=6.5,
              labelcolor=TINTA, handlelength=1.2, columnspacing=1)
    fig.tight_layout()
    fig.savefig(FIGURAS / "shap_teste_grupos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def shap_features(n=6):
    """As n variáveis mais usadas em cada região e no modelo completo no Brasil todo, numa figura
    para o modelo sem grafo e outra para o com grafo.

    Cada figura tem três colunas e duas linhas de painéis, para caber legível na largura da
    página. O comprimento da barra é a média do |SHAP| (mm), na mesma escala nas duas figuras, e
    a cor é o grupo da variável, como na figura de barras empilhadas.
    """
    f = pd.concat([pd.read_csv(SAIDA / "shap_teste_features.csv"),
                   pd.read_csv(SAIDA / "shap_global_features.csv").query("regiao == 'brasil'")])
    colunas = REGIOES + ["brasil"]
    vmax = f.media_abs_shap_mm.max()
    fonte = 9
    for braco, sufixo in (("sem_grafo", "sem"), ("com_grafo", "com")):
        fig, eixos = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True)
        for k, regiao in enumerate(colunas):
            ax = eixos[k // 3, k % 3]
            t = (f[(f.braco == braco) & (f.regiao == regiao)]
                 .nlargest(n, "media_abs_shap_mm").iloc[::-1])
            y = np.arange(len(t))
            ax.barh(y, t.media_abs_shap_mm, height=0.68,
                    color=[GRUPOS[grupo_shap(x)][1] for x in t.feature])
            for yi, v in zip(y, t.media_abs_shap_mm):
                ax.text(v + vmax * 0.03, yi, f"{v:.2f}".replace(".", ","), va="center",
                        fontsize=fonte - 0.5, color=TINTA_2)
            ax.set_yticks(y, [textwrap.fill(nome_extenso(x), 24) for x in t.feature],
                          fontsize=fonte, linespacing=0.95)
            ax.set_xlim(0, vmax * 1.3)
            ax.tick_params(axis="both", length=0, labelsize=fonte - 0.5)
            ax.xaxis.set_tick_params(labelbottom=True)
            for lado in ("top", "right", "left"):
                ax.spines[lado].set_visible(False)
            ax.set_title(NOME.get(regiao, "Brasil (modelo completo)"), fontsize=fonte + 1,
                         color=TINTA, loc="left")
        fig.supxlabel("média do |SHAP| (mm)", fontsize=fonte, color=TINTA_2, y=0.01)
        fig.legend(handles=[Rectangle((0, 0), 1, 1, color=cor) for _, cor in GRUPOS.values()],
                   labels=[nome for nome, _ in GRUPOS.values()], ncol=5, loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=fonte + 0.5,
                   labelcolor=TINTA, handlelength=1.0, columnspacing=1.2)
        fig.tight_layout(w_pad=0.8, h_pad=1.4)
        fig.savefig(FIGURAS / f"shap_teste_features_{sufixo}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    FIGURAS.mkdir(parents=True, exist_ok=True)
    matriz()
    shap_grupos()
    shap_features()
    print(f"figuras em {FIGURAS}/")
