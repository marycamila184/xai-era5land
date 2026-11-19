"""Modelo global (as cinco regiões juntas), sem e com grafo, comparado aos regionais.

Hiperparâmetros: escolhidos pela validação temporal (os blocos de 1995–2014 do treinar_p90.py,
com a perda de todas as regiões juntas), na mesma grade de 24 combinações dos regionais.

Teste: 2015–2025, em âncoras que o modelo não viu, como na diagonal da matriz entre regiões.
Para cada grupo espacial g (validacao_espacial.py), o global é treinado sem o grupo g de
nenhuma região (1980–2014) e prevê o grupo g de todas as regiões. A referência é a climatologia
de cada região de teste.

Saídas em av1/experimentos/resultados/: global_cv.csv (a escolha) e global_teste.csv (uma linha
por braço e região de teste, no formato de matriz_regioes.csv, com treino = "global").

Uso (na raiz do projeto, depois de treinar_p90.py e validacao_espacial.py):
    uv run python av1/experimentos/modelo_global.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matriz_regioes import carregar  # noqa: E402
from treinar_p90 import (ALPHA, BLOCOS, FIXOS, GRADE, REGIOES, SAIDA, bracos,  # noqa: E402
                         climatologia)
from validacao_espacial import N_GRUPOS, grupos_espaciais  # noqa: E402

TESTE = (2015, 2025)


def ajustar(treino, cols, hp):
    return HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)


if __name__ == "__main__":
    t0 = time.time()
    grupos = grupos_espaciais()
    d = pd.concat([carregar(r, grupos).assign(regiao=r) for r in REGIOES], ignore_index=True)
    cols_por_braco = bracos(d.columns)
    cv, linhas = [], []
    for braco, cols in cols_por_braco.items():
        perdas = []
        for hp in GRADE:
            p = []
            for ini, fim in BLOCOS:
                treino, valid = d[d.ano < ini], d[(d.ano >= ini) & (d.ano <= fim)]
                p.append(mean_pinball_loss(valid.tp_mm, ajustar(treino, cols, hp).predict(valid[cols]),
                                           alpha=ALPHA))
            perdas.append(np.mean(p))
            cv.append(dict(braco=braco, **hp, perda_cv=np.mean(p)))
            print(f"{braco} {hp} perda {np.mean(p):.4f} ({time.time() - t0:.0f}s)", flush=True)
        hp = GRADE[int(np.argmin(perdas))]

        treino = d[d.ano <= 2014]
        teste = d[(d.ano >= TESTE[0]) & (d.ano <= TESTE[1])].copy()
        teste["pred"] = np.nan
        for g in range(N_GRUPOS):
            m = ajustar(treino[treino.grupo != g], cols, hp)
            sel = teste.grupo == g
            teste.loc[sel, "pred"] = m.predict(teste.loc[sel, cols])
        for y in REGIOES:
            ty = teste[teste.regiao == y]
            dy = d[d.regiao == y]
            clima = climatologia(dy[dy.ano <= 2014], ty)
            perda = mean_pinball_loss(ty.tp_mm, ty.pred, alpha=ALPHA)
            perda_clima = mean_pinball_loss(ty.tp_mm, clima, alpha=ALPHA)
            linhas.append(dict(braco=braco, treino="global", teste=y, dias=len(ty), perda=perda,
                               perda_clima=perda_clima, ganho=1 - perda / perda_clima,
                               cobertura=float(np.mean(ty.tp_mm <= ty.pred)), **hp))
        print(f"{braco} teste ok ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(cv).to_csv(SAIDA / "global_cv.csv", index=False)
    r = pd.DataFrame(linhas)
    r.to_csv(SAIDA / "global_teste.csv", index=False)
    print((100 * r.pivot(index="braco", columns="teste", values="ganho")[REGIOES]).round(2).to_string())
