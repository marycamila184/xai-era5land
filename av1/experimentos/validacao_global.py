"""Validação temporal e espacial-temporal do modelo global, igual à dos modelos regionais.

Mesmos cinco blocos de 1995–2014 e mesmos grupos espaciais de validacao_espacial.py, com as
cinco regiões juntas no treino:

- temporal: o bloco é validado por um modelo treinado em todas as âncoras nos anos anteriores;
- espacial-temporal: o bloco i valida o grupo i de cada região, com um modelo treinado nos
  outros grupos e nos anos anteriores ao bloco.

Hiperparâmetros: os que a validação temporal escolheu para o global (resultados/global_cv.csv).
O ganho é calculado por região, sobre a climatologia de cada âncora validada nos anos anteriores
ao bloco, como no validacao_espacial.py, e é a média dos cinco blocos. O teste (2015–2025) não
é usado.

Saídas em av1/experimentos/resultados/: validacao_global_blocos.csv e validacao_global_resumo.csv.

Uso (na raiz do projeto, depois de modelo_global.py):
    uv run python av1/experimentos/validacao_global.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matriz_regioes import carregar  # noqa: E402
from treinar_p90 import (ALPHA, BLOCOS, FIXOS, HP, REGIOES, SAIDA, bracos,  # noqa: E402
                         climatologia)
from validacao_espacial import grupos_espaciais  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    grupos = grupos_espaciais()
    d = pd.concat([carregar(r, grupos).assign(regiao=r) for r in REGIOES], ignore_index=True)
    d = d[d.ano <= 2014].reset_index(drop=True)
    cv = pd.read_csv(SAIDA / "global_cv.csv")
    linhas = []
    for braco, cols in bracos(d.columns).items():
        melhor = cv[cv.braco == braco].sort_values("perda_cv").iloc[0]
        hp = {k: int(melhor[k]) for k in HP}
        for i, (ini, fim) in enumerate(BLOCOS):
            no_bloco = (d.ano >= ini) & (d.ano <= fim)
            divisoes = {"temporal": (d.ano < ini, no_bloco),
                        "espacial_temporal": ((d.ano < ini) & (d.grupo != i), no_bloco & (d.grupo == i))}
            for validacao, (m_treino, m_valid) in divisoes.items():
                treino, valid = d[m_treino], d[m_valid]
                m = HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)
                valid = valid.assign(pred=m.predict(valid[cols]))
                for regiao, v in valid.groupby("regiao"):
                    hist = d[(d.ano < ini) & d.anchor_id.isin(v.anchor_id.unique())]
                    perda = mean_pinball_loss(v.tp_mm, v.pred, alpha=ALPHA)
                    perda_clima = mean_pinball_loss(v.tp_mm, climatologia(hist, v), alpha=ALPHA)
                    linhas.append(dict(validacao=validacao, braco=braco, regiao=regiao, bloco=f"{ini}-{fim}",
                                       n_valid=len(v), perda=perda, perda_clima=perda_clima,
                                       ganho=1 - perda / perda_clima, **hp))
            print(f"{braco} {ini}-{fim} ok ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(linhas)
    r.to_csv(SAIDA / "validacao_global_blocos.csv", index=False)
    res = (100 * r.groupby(["validacao", "braco", "regiao"]).ganho.mean()).unstack()[REGIOES]
    res.to_csv(SAIDA / "validacao_global_resumo.csv")
    print(res.round(2).to_string())
    p = r.pivot_table(index=["validacao", "regiao", "bloco"], columns="braco", values="perda")
    print("grafo vence:", (p.com_grafo < p.sem_grafo).groupby("validacao").sum().to_dict())
