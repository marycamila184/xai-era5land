"""SHAP do modelo completo (as cinco regiões juntas), sem e com grafo, explicado no teste.

Como em shap_teste.py, mas com um só modelo: treinado nos dias fortes de todas as âncoras das
cinco regiões em 1980–2014, com os hiperparâmetros que a validação escolheu para ele
(resultados/global_cv.csv), e explicado em todos os dias fortes de 2015–2025. A participação
sai para o Brasil todo (regiao = "brasil", a barra da figura) e para cada região de teste.
O SHAP só descreve o modelo já fechado.

Saídas em av1/experimentos/resultados/, no formato das de shap_teste.py:
- shap_global_features.csv: média do |SHAP| (mm) e participação por região, braço e feature;
- shap_global_grupos.csv: participação por região, braço e grupo.

Uso (na raiz do projeto, depois de modelo_global.py):
    uv run python av1/experimentos/shap_global.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shap_grafo import grupo_shap  # noqa: E402
from treinar_p90 import DADOS, FIXOS, HP, LIMIAR_MM, REGIOES, SAIDA, bracos  # noqa: E402

TESTE = (2015, 2025)


def hiperparametros_global(braco):
    """A combinação de menor perda na validação do modelo completo."""
    cv = pd.read_csv(SAIDA / "global_cv.csv")
    linha = cv[cv.braco == braco].sort_values("perda_cv").iloc[0]
    return {k: int(linha[k]) for k in HP}


if __name__ == "__main__":
    t0 = time.time()
    d = pd.concat([pd.read_parquet(DADOS / f"features_{r}.parquet").assign(regiao=r)
                   for r in REGIOES], ignore_index=True)
    d = d[d.tp_mm >= LIMIAR_MM]
    ano = d.date.dt.year
    treino = d[ano <= 2014]
    teste = d[(ano >= TESTE[0]) & (ano <= TESTE[1])]
    linhas = []
    for braco, cols in bracos(d.columns).items():
        hp = hiperparametros_global(braco)
        m = HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)
        ex = shap.TreeExplainer(m)
        sv = np.abs(ex.shap_values(teste[cols]))
        regiao_teste = teste.regiao.to_numpy()
        for regiao in REGIOES + ["brasil"]:
            a = sv.mean(0) if regiao == "brasil" else sv[regiao_teste == regiao].mean(0)
            for c, v in zip(cols, a):
                linhas.append(dict(regiao=regiao, braco=braco, feature=c, grupo=grupo_shap(c),
                                   media_abs_shap_mm=v, participacao=v / a.sum()))
        print(f"{braco} ok ({time.time() - t0:.0f}s)", flush=True)
    f = pd.DataFrame(linhas)
    f.to_csv(SAIDA / "shap_global_features.csv", index=False)
    g = f.groupby(["regiao", "braco", "grupo"]).participacao.sum().reset_index()
    g.to_csv(SAIDA / "shap_global_grupos.csv", index=False)
    print((100 * g[g.braco == "com_grafo"].pivot(index="grupo", columns="regiao", values="participacao")
           [REGIOES + ["brasil"]]).round(1).to_string())
