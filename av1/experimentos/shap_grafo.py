"""SHAP do modelo com grafo: quanto do |SHAP| vem das features locais e de cada grupo do grafo.

Para cada região, o modelo com grafo, com os hiperparâmetros escolhidos na validação
(resultados/cv_resumo.csv), é treinado nos dias fortes de 1980 a 2010 e explicado nos dias
fortes de 2011 a 2014, o último bloco da validação. São anos que o modelo não viu, e o teste
(2015–2025) continua sem uso. Com mais de 3.000 dias, sorteia 3.000 (basta para a participação dos grupos).

A participação de um grupo é a soma do |SHAP| das features dele dividida pela soma do |SHAP|
de todas, no conjunto dos dias explicados (slide 10 da proposta). Os grupos do grafo são os
da ablação (ablacao.py).

Saídas em av1/experimentos/resultados/: shap_grupos.csv (participação por região e grupo),
shap_features.csv (média do |SHAP| por região e feature) e shap_<regiao>.parquet (os valores).

Uso (na raiz do projeto):
    uv run python av1/experimentos/shap_grafo.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablacao import GRUPOS, grupo  # noqa: E402
from treinar_p90 import (DADOS, FIXOS, LIMIAR_MM, REGIOES, SAIDA, bracos,  # noqa: E402
                         hiperparametros)

ANO_CORTE = 2010
MAX_DIAS = 3_000


def grupo_shap(col):
    return grupo(col) if col.startswith("g_") else "local"


if __name__ == "__main__":
    t0 = time.time()
    grupos, features = {}, {}
    for regiao in REGIOES:
        d = pd.read_parquet(DADOS / f"features_{regiao}.parquet")
        d = d[(d.tp_mm >= LIMIAR_MM) & (d.date.dt.year <= 2014)]
        cols = bracos(d.columns)["com_grafo"]
        treino, expl = d[d.date.dt.year <= ANO_CORTE], d[d.date.dt.year > ANO_CORTE]
        if len(expl) > MAX_DIAS:
            expl = expl.sample(MAX_DIAS, random_state=0)
        hp = hiperparametros(SAIDA / "cv_resumo.csv", regiao)
        m = HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(expl[cols])
        assert np.allclose(sv.sum(1) + ex.expected_value, m.predict(expl[cols]), atol=1e-3)
        out = pd.DataFrame(sv, columns=cols, index=expl.index)
        out[["date", "anchor_id", "tp_mm"]] = expl[["date", "anchor_id", "tp_mm"]]
        out.to_parquet(SAIDA / f"shap_{regiao}.parquet")
        a = np.abs(sv).mean(0)
        features[regiao] = pd.Series(a, index=cols)
        grupos[regiao] = features[regiao].groupby(lambda c: grupo_shap(c)).sum() / a.sum()
        print(f"{regiao}: {len(expl):,} dias explicados ({time.time() - t0:.0f}s)", flush=True)
    g = pd.DataFrame(grupos).reindex(["local"] + GRUPOS).dropna(how="all")
    g.loc["grafo (total)"] = g.drop(index="local").sum()
    g.to_csv(SAIDA / "shap_grupos.csv")
    f = pd.DataFrame(features)
    f.to_csv(SAIDA / "shap_features.csv")
    print((100 * g).round(1).to_string())
    for regiao in REGIOES:
        top = f[regiao].sort_values(ascending=False).head(8)
        print(f"\n{regiao}: " + ", ".join(f"{c} {v:.2f}" for c, v in top.items()))
