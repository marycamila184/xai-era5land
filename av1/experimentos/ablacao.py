"""Ablação das features de grafo por grupo, com os blocos de validação do treinar_p90.py.

As features `g_*` em cinco grupos:

- tp_anel: média, desvio, máximo, média menos âncora e fração > 1 mm do tp das vizinhas;
- agregados: média, desvio e média menos âncora das outras variáveis;
- gradientes: gradientes de t2m e d2m;
- adveccao: advecção de t2m e d2m;
- cinematica: convergência e vorticidade.

Para cada grupo, dois braços: `base+<grupo>` (só aquele grupo somado às locais) e
`completo-<grupo>` (todas as de grafo menos aquele grupo).

A ablação é sobre o conjunto completo de 51 features (com a advecção, que saiu do modelo por
causa dela, e com a altitude, que saiu depois pela validação espacial). Os braços `sem_grafo` e `com_grafo` (51 features) rodam aqui de novo.

Versão rápida (decisão de 22/10/2026, entrega no mesmo dia): em cada região, todos os braços
usam os hiperparâmetros que o braço com grafo de 51 features escolheu na validação
(resultados/antes_sem_adveccao/cv_resumo.csv), em vez de passar pela grade. Medido na grade
do treinar_p90.py, isso custa ao braço sem grafo de 0 a 0,21 p.p. (Sudeste) e favorece levemente os braços com mais features; diferenças abaixo de
~0,3 p.p. devem ser lidas como empate.

Saídas em av1/experimentos/resultados/: ablacao_blocos.parquet e ablacao_resumo.csv
(ganho em p.p.: `base+` em relação ao sem_grafo, `completo-` em relação ao com_grafo).

Uso (na raiz do projeto):
    uv run python av1/experimentos/ablacao.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treinar_p90 import REGIOES, SAIDA, avaliar, bracos, hiperparametros  # noqa: E402

RESUMO_51 = SAIDA / "antes_sem_adveccao" / "cv_resumo.csv"


def grupo(col):
    if col.endswith("tp_mm"):
        return "tp_anel"
    if col.startswith(("g_gradx_", "g_grady_", "g_gradmag_")):
        return "gradientes"
    if col.startswith("g_adv_"):
        return "adveccao"
    if col in ("g_conv", "g_vort"):
        return "cinematica"
    return "agregados"


GRUPOS = ["tp_anel", "agregados", "gradientes", "adveccao", "cinematica"]


def bracos_ablacao(colunas):
    b = bracos(colunas, excluir=(), com_altitude=True)
    locais, grafo = b["sem_grafo"], [c for c in b["com_grafo"] if c.startswith("g_")]
    out = dict(b)
    for g in GRUPOS:
        out[f"base+{g}"] = locais + [c for c in grafo if grupo(c) == g]
        out[f"completo-{g}"] = locais + [c for c in grafo if grupo(c) != g]
    return out


def resumir(r):
    ganho = r.pivot_table(index="regiao", columns="braco", values="ganho")
    pp = pd.DataFrame(index=ganho.index)
    pp["sem_grafo"] = 100 * ganho.sem_grafo
    pp["com_grafo"] = 100 * ganho.com_grafo
    for g in GRUPOS:
        pp[f"base+{g}"] = 100 * (ganho[f"base+{g}"] - ganho.sem_grafo)
        pp[f"completo-{g}"] = 100 * (ganho[f"completo-{g}"] - ganho.com_grafo)
    return pp.reindex(REGIOES)


if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    partes = []
    for regiao in REGIOES:
        hp = hiperparametros(RESUMO_51, regiao)
        partes.append(avaliar(regiao, bracos_ablacao, grade=[hp]))
        pd.concat(partes).to_parquet(SAIDA / "ablacao_blocos.parquet", index=False)
        print(f"{regiao} ok {hp} ({time.time() - t0:.0f}s)", flush=True)
    pp = resumir(pd.concat(partes))
    pp.to_csv(SAIDA / "ablacao_resumo.csv")
    print(pp.round(2).T.to_string())
