"""SHAP dos modelos finais sem e com grafo, explicados no teste (2015–2025).

Os modelos finais de cada região são treinados nos dias fortes de 1980–2014, com os
hiperparâmetros da validação (resultados/cv_resumo.csv), e explicados em todos os dias fortes
de 2015–2025 de cada região, sem amostragem (até 22/10/2026 eram 3.000 sorteados). O SHAP só
descreve o modelo já fechado: nada do que ele mostra volta para a escolha de features ou de
hiperparâmetros.

Saídas em av1/experimentos/resultados/:
- shap_teste_features.csv: média do |SHAP| (mm) e participação na soma do |SHAP| por região,
  braço e feature;
- shap_teste_grupos.csv: participação por região, braço e grupo (local e os grupos de grafo).

Uso (na raiz do projeto):
    uv run python av1/experimentos/shap_teste.py
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
from treinar_p90 import (DADOS, FIXOS, LIMIAR_MM, REGIOES, SAIDA, bracos,  # noqa: E402
                         hiperparametros)

TESTE = (2015, 2025)

if __name__ == "__main__":
    t0 = time.time()
    linhas = []
    for regiao in REGIOES:
        d = pd.read_parquet(DADOS / f"features_{regiao}.parquet")
        d = d[d.tp_mm >= LIMIAR_MM]
        ano = d.date.dt.year
        treino = d[ano <= 2014]
        teste = d[(ano >= TESTE[0]) & (ano <= TESTE[1])]
        for braco, cols in bracos(d.columns).items():
            hp = hiperparametros(SAIDA / "cv_resumo.csv", regiao, braco)
            m = HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)
            ex = shap.TreeExplainer(m)
            sv = ex.shap_values(teste[cols])
            assert np.allclose(sv.sum(1) + ex.expected_value, m.predict(teste[cols]), atol=1e-3)
            a = np.abs(sv).mean(0)
            for c, v in zip(cols, a):
                linhas.append(dict(regiao=regiao, braco=braco, feature=c, grupo=grupo_shap(c),
                                   media_abs_shap_mm=v, participacao=v / a.sum()))
        print(f"{regiao} ok ({time.time() - t0:.0f}s)", flush=True)
    f = pd.DataFrame(linhas)
    f.to_csv(SAIDA / "shap_teste_features.csv", index=False)
    g = f.groupby(["regiao", "braco", "grupo"]).participacao.sum().reset_index()
    g.to_csv(SAIDA / "shap_teste_grupos.csv", index=False)
    print((100 * g[g.braco == "com_grafo"].pivot(index="grupo", columns="regiao", values="participacao")
           [REGIOES]).round(1).to_string())
