"""Compara o desenho atual (35 anos de treino) com o de 10 anos sugerido pelo Vitor, no teste.

- **atual**: hiperparâmetros da validação de 1995–2014 (resultados/cv_resumo.csv), treino em
  1980–2014; avaliado em 2024–2025 (a janela de teste do Vitor) e em 2015–2025 (o teste nosso).
- **vitor_10anos**: só 2016–2025. Validação de 2 em 2 anos com janela expansiva a partir de 2016
  (2018–19, 2020–21, 2022–23), a mesma grade, treino final em 2016–2023 e teste em 2024–2025.

A climatologia de referência sai dos anos de treino de cada desenho. Métricas: perda
quantílica, ganho sobre a climatologia e cobertura (fração dos dias com chuva abaixo do p90
previsto; o ideal é 90%).

Saída: av1/experimentos/resultados/comparacao_desenhos.csv.

Uso (na raiz do projeto):
    uv run python av1/experimentos/comparar_desenhos.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treinar_p90 import (ALPHA, DADOS, FIXOS, GRADE, LIMIAR_MM, REGIOES, SAIDA,  # noqa: E402
                         bracos, climatologia, hiperparametros)

BLOCOS_VITOR = [(2018, 2019), (2020, 2021), (2022, 2023)]
INICIO_VITOR, TESTE_VITOR = 2016, (2024, 2025)


def metricas(y, pred, clima):
    perda, perda_clima = (mean_pinball_loss(y, p, alpha=ALPHA) for p in (pred, clima))
    return dict(dias=len(y), perda=perda, perda_clima=perda_clima, ganho=1 - perda / perda_clima,
                cobertura=float(np.mean(y <= pred)))


def ajustar(treino, cols, hp):
    return HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)


def escolher_vitor(d, cols):
    """Combinação da grade de menor perda média nos três blocos de validação do Vitor."""
    perdas = []
    for hp in GRADE:
        p = []
        for ini, fim in BLOCOS_VITOR:
            treino = d[(d.ano >= INICIO_VITOR) & (d.ano < ini)]
            valid = d[(d.ano >= ini) & (d.ano <= fim)]
            p.append(mean_pinball_loss(valid.tp_mm, ajustar(treino, cols, hp).predict(valid[cols]),
                                       alpha=ALPHA))
        perdas.append(np.mean(p))
    return GRADE[int(np.argmin(perdas))], float(np.min(perdas))


if __name__ == "__main__":
    linhas = []
    for regiao in REGIOES:
        d = pd.read_parquet(DADOS / f"features_{regiao}.parquet")
        d = d[d.tp_mm >= LIMIAR_MM].reset_index(drop=True)
        d["month"], d["ano"] = d.date.dt.month, d.date.dt.year
        for braco, cols in bracos(d.columns).items():
            # atual
            hp = hiperparametros(SAIDA / "cv_resumo.csv", regiao, braco)
            treino = d[d.ano <= 2014]
            m = ajustar(treino, cols, hp)
            for nome, (ini, fim) in [("2024-2025", TESTE_VITOR), ("2015-2025", (2015, 2025))]:
                teste = d[(d.ano >= ini) & (d.ano <= fim)]
                linhas.append(dict(regiao=regiao, braco=braco, desenho="atual", teste=nome,
                                   anos_treino="1980-2014", **hp,
                                   **metricas(teste.tp_mm.to_numpy(), m.predict(teste[cols]),
                                              climatologia(treino, teste))))
            # vitor
            hp, perda_cv = escolher_vitor(d, cols)
            treino = d[(d.ano >= INICIO_VITOR) & (d.ano < TESTE_VITOR[0])]
            teste = d[(d.ano >= TESTE_VITOR[0]) & (d.ano <= TESTE_VITOR[1])]
            m = ajustar(treino, cols, hp)
            linhas.append(dict(regiao=regiao, braco=braco, desenho="vitor_10anos", teste="2024-2025",
                               anos_treino="2016-2023", **hp, perda_cv=perda_cv,
                               **metricas(teste.tp_mm.to_numpy(), m.predict(teste[cols]),
                                          climatologia(treino, teste))))
        print(regiao, "ok", flush=True)
    r = pd.DataFrame(linhas)
    r.to_csv(SAIDA / "comparacao_desenhos.csv", index=False)
    t = r.assign(g=100 * r.ganho, c=100 * r.cobertura).pivot_table(
        index="regiao", columns=["desenho", "teste", "braco"], values=["g", "c"]).reindex(REGIOES)
    pd.set_option("display.width", 250)
    print(t.round(1).to_string())
