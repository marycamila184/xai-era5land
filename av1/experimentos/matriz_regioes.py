"""Matriz de generalização entre regiões (slide 9 da proposta), sem e com grafo.

Para cada região de treino X e de teste Y, o modelo é treinado nos dias fortes de X em
1980–2014 e testado nos dias fortes de Y em 2015–2025, com os hiperparâmetros que a validação
escolheu para X (resultados/cv_resumo.csv).

- **Fora da diagonal**: o modelo de X, treinado em todas as âncoras de X, aplicado em Y.
- **Na diagonal**: a mesma região, mas em âncoras que o modelo não viu. Para cada um dos 5
  grupos espaciais de X (validacao_espacial.py), um modelo treinado nos outros 4 prevê esse
  grupo; as previsões dos 5 grupos são juntadas. Assim toda a matriz é em anos e lugares novos.

Referência: a climatologia mensal do p90 de cada âncora da região de teste, tirada dos anos de
1980–2014 dela. O ganho é 1 − perda do modelo / perda da climatologia.

Saída: av1/experimentos/resultados/matriz_regioes.csv.

Uso (na raiz do projeto):
    uv run python av1/experimentos/matriz_regioes.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treinar_p90 import (ALPHA, DADOS, FIXOS, LIMIAR_MM, REGIOES, SAIDA,  # noqa: E402
                         bracos, climatologia, hiperparametros)
from validacao_espacial import N_GRUPOS, grupos_espaciais  # noqa: E402

TESTE = (2015, 2025)


def carregar(regiao, grupos):
    d = pd.read_parquet(DADOS / f"features_{regiao}.parquet")
    d = d[d.tp_mm >= LIMIAR_MM].reset_index(drop=True)
    d["month"], d["ano"] = d.date.dt.month, d.date.dt.year
    d["grupo"] = d.anchor_id.map(grupos.set_index("anchor_id").grupo)
    return d


def ajustar(treino, cols, hp):
    return HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)


if __name__ == "__main__":
    grupos = grupos_espaciais()
    dados = {r: carregar(r, grupos) for r in REGIOES}
    cols_por_braco = bracos(dados[REGIOES[0]].columns)
    t0 = time.time()
    linhas = []
    for braco, cols in cols_por_braco.items():
        for x in REGIOES:
            hp = hiperparametros(SAIDA / "cv_resumo.csv", x, braco)
            dx = dados[x]
            treino_x = dx[dx.ano <= 2014]
            modelo_x = ajustar(treino_x, cols, hp)
            for y in REGIOES:
                dy = dados[y]
                teste = dy[(dy.ano >= TESTE[0]) & (dy.ano <= TESTE[1])]
                if x == y:   # âncoras que o modelo não viu: um modelo por grupo deixado de fora
                    pred = np.empty(len(teste))
                    for g in range(N_GRUPOS):
                        m = ajustar(treino_x[treino_x.grupo != g], cols, hp)
                        sel = (teste.grupo == g).to_numpy()
                        pred[sel] = m.predict(teste.loc[sel, cols])
                else:
                    pred = modelo_x.predict(teste[cols])
                y_obs = teste.tp_mm.to_numpy()
                clima = climatologia(dy[dy.ano <= 2014], teste)
                perda = mean_pinball_loss(y_obs, pred, alpha=ALPHA)
                perda_clima = mean_pinball_loss(y_obs, clima, alpha=ALPHA)
                linhas.append(dict(braco=braco, treino=x, teste=y, dias=len(teste), perda=perda,
                                   perda_clima=perda_clima, ganho=1 - perda / perda_clima,
                                   cobertura=float(np.mean(y_obs <= pred)), **hp))
            print(f"{braco} {x} ok ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(linhas)
    r.to_csv(SAIDA / "matriz_regioes.csv", index=False)
    for braco in cols_por_braco:
        m = r[r.braco == braco].pivot(index="treino", columns="teste", values="ganho")
        print(f"\n{braco} (ganho %, linhas = treino, colunas = teste)")
        print((100 * m.loc[REGIOES, REGIOES]).round(1).to_string())
