"""Validação espacial-temporal e o papel da altitude como identificador da âncora.

Duas validações nos dias fortes de 1980–2014, com os mesmos cinco blocos de anos:

- **temporal** (a do treinar_p90.py): todas as âncoras no treino e na validação; o bloco é
  validado com um modelo treinado nos anos anteriores.
- **espacial_temporal**: as âncoras de cada região em 5 grupos espaciais contíguos (k-means
  nas coordenadas em km). O bloco i valida só o grupo i, com um modelo treinado só nos outros
  grupos e só nos anos anteriores. Lugar e tempo ficam fora do treino ao mesmo tempo, então dois
  vizinhos no mesmo dia nunca caem um no treino e outro na validação (slide 8 da proposta).

Em cada validação, os braços sem e com grafo em três versões da altitude: como está
(`altitude`), sem altitude e em faixas de 200 m (`altitude_faixa`). Se a altitude ajuda na
temporal e não na espacial-temporal, ela estava servindo para reconhecer a âncora. Foi esse
resultado (22/10/2026) que tirou a altitude do modelo; as três versões ficam aqui como
diagnóstico. A versão `sem_altitude` é o modelo final.

Hiperparâmetros fixos: os que o braço com grafo escolheu na validação temporal
(resultados/cv_resumo.csv), como na ablação rápida.

Referência: a mesma climatologia do treinar_p90.py, o p90 por âncora e mês nos anos anteriores
ao bloco, tirada do histórico da própria âncora validada (também no espacial-temporal, onde o
modelo não viu essa âncora). Assim os ganhos das duas validações ficam comparáveis.

Saídas em av1/experimentos/resultados/: grupos_espaciais.csv, validacao_espacial_blocos.parquet
e validacao_espacial_resumo.csv.

Uso (na raiz do projeto):
    uv run python av1/experimentos/validacao_espacial.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treinar_p90 import (ALPHA, BLOCOS, DADOS, FIXOS, LIMIAR_MM, REGIOES, SAIDA,  # noqa: E402
                         bracos, climatologia, hiperparametros)

ANCORAS = Path("av1/amostra_brasil/dados_150/ancoras.csv")
N_GRUPOS = 5
FAIXA_M = 200
NOMES_REGIAO = {"norte": "Norte", "nordeste": "Nordeste", "centro_oeste": "Centro-Oeste",
                "sudeste": "Sudeste", "sul": "Sul"}


def grupos_espaciais():
    a = pd.read_csv(ANCORAS)
    out = []
    for regiao, nome in NOMES_REGIAO.items():
        g = a[a.region == nome].copy()
        xy = np.c_[g.longitude * np.cos(np.radians(g.latitude.mean())), g.latitude] * 111.32
        g["grupo"] = KMeans(N_GRUPOS, n_init=10, random_state=0).fit_predict(xy)
        g["regiao"] = regiao
        out.append(g[["regiao", "anchor_id", "grupo", "latitude", "longitude", "altitude"]])
    return pd.concat(out, ignore_index=True)


def versoes(colunas):
    b = bracos(colunas, com_altitude=False)
    out = {}
    for braco, cols in b.items():
        out[f"{braco}|altitude"] = cols + ["altitude"]
        out[f"{braco}|sem_altitude"] = cols
        out[f"{braco}|faixas"] = cols + ["altitude_faixa"]
    return out


def avaliar(regiao, grupos, hp):
    d = pd.read_parquet(DADOS / f"features_{regiao}.parquet")
    d = d[(d.tp_mm >= LIMIAR_MM) & (d.date.dt.year <= 2014)].reset_index(drop=True)
    d["month"], d["ano"] = d.date.dt.month, d.date.dt.year
    d["altitude_faixa"] = (d.altitude // FAIXA_M) * FAIXA_M
    d["grupo"] = d.anchor_id.map(grupos.set_index("anchor_id").grupo)
    assert d.grupo.notna().all()
    linhas = []
    for i, (ini, fim) in enumerate(BLOCOS):
        no_bloco = (d.ano >= ini) & (d.ano <= fim)
        divisoes = {
            "temporal": (d.ano < ini, no_bloco),
            "espacial_temporal": ((d.ano < ini) & (d.grupo != i), no_bloco & (d.grupo == i)),
        }
        for validacao, (m_treino, m_valid) in divisoes.items():
            treino, valid = d[m_treino], d[m_valid]
            historico = d[(d.ano < ini) & d.anchor_id.isin(valid.anchor_id.unique())]
            y = valid.tp_mm.to_numpy()
            perda_clima = mean_pinball_loss(y, climatologia(historico, valid), alpha=ALPHA)
            for nome, cols in versoes(d.columns).items():
                braco, alt = nome.split("|")
                m = HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)
                perda = mean_pinball_loss(y, m.predict(valid[cols]), alpha=ALPHA)
                linhas.append(dict(regiao=regiao, validacao=validacao, braco=braco, altitude=alt,
                                   bloco=f"{ini}-{fim}", grupo_validado=i, n_treino=len(treino),
                                   n_valid=len(valid), perda=perda, perda_clima=perda_clima,
                                   ganho=1 - perda / perda_clima))
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)
    grupos = grupos_espaciais()
    grupos.to_csv(SAIDA / "grupos_espaciais.csv", index=False)
    print(grupos.groupby(["regiao", "grupo"]).size().unstack().to_string())
    t0 = time.time()
    partes = []
    for regiao in REGIOES:
        hp = hiperparametros(SAIDA / "cv_resumo.csv", regiao)
        partes.append(avaliar(regiao, grupos[grupos.regiao == regiao], hp))
        pd.concat(partes).to_parquet(SAIDA / "validacao_espacial_blocos.parquet", index=False)
        print(f"{regiao} ok {hp} ({time.time() - t0:.0f}s)", flush=True)
    r = pd.concat(partes)
    res = (100 * r.groupby(["validacao", "braco", "altitude", "regiao"]).ganho.mean()).unstack()
    res = res[REGIOES]
    res.to_csv(SAIDA / "validacao_espacial_resumo.csv")
    print(res.round(2).to_string())
