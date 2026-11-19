"""Boosting quantílico p90 com e sem grafo, um modelo por região, só validação cruzada.

Alvo: o tp da âncora no dia t, nos dias com 10 mm ou mais (R10mm), como na AT3. O modelo
prevê o p90 desse tp com a perda quantílica (*pinball*, alpha = 0,9).

- **sem_grafo**: as features locais da véspera (`<var>_d1`) e o dia do ano. A altitude saiu em
  22/10/2026 (decisão da Mary Camila): na validação espacial-temporal (validacao_espacial.py)
  ela servia em parte para o modelo reconhecer a âncora, então fica fora para o resultado valer
  para lugares novos sem essa ajuda. Custo medido em lugares novos: ~0 no Norte, no Nordeste,
  no Centro-Oeste e no Sudeste, 0,6 a 0,7 p.p. no Sul (onde ela carrega a serra).
- **com_grafo**: as mesmas mais as features `g_*` (ver montar_features.py), menos a advecção.
  A advecção saiu em 22/10/2026: na ablação (ablacao.py) tirá-la mudou o ganho em menos de
  0,1 p.p. em todas as regiões e piorou em só 11 dos 25 blocos. Gradientes e agregados ficam:
  tirar os dois juntos custava até 0,8 p.p.

Validação no tempo, como no slide 8 da proposta e na AT3: cinco blocos de quatro anos de 1995 a
2014, cada um validado com um modelo treinado em todos os anos antes dele (janela expansiva,
começando em 1980). O teste (2015–2025) não é usado aqui.

Referência: a climatologia mensal do p90 de cada âncora, tirada dos anos de treino de cada
bloco (o p90 dos dias fortes daquela âncora naquele mês; com menos de 10 dias, o p90 da âncora
no ano todo). O ganho é 1 − perda do modelo / perda da climatologia, no bloco.

Os dois braços usam a mesma grade de hiperparâmetros (folhas por árvore, número de árvores e
mínimo de dias por folha; 24 combinações), e cada um fica com a combinação de menor perda média
nos blocos. O mínimo de dias por folha entrou na grade em 22/10/2026; antes era 100 fixo.

Saídas em av1/experimentos/resultados/: cv_blocos.parquet (tudo), cv_resumo.csv e
tabela_cv.md (a tabela de comparação).

Uso (na raiz do projeto):
    uv run python av1/experimentos/treinar_p90.py
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

DADOS = Path("av1/experimentos/dados")
SAIDA = Path("av1/experimentos/resultados")
REGIOES = ["norte", "nordeste", "centro_oeste", "sudeste", "sul"]
LIMIAR_MM = 10.0
ALPHA = 0.9
BLOCOS = [(1995, 1998), (1999, 2002), (2003, 2006), (2007, 2010), (2011, 2014)]
MIN_DIAS_CLIMA = 10
HP = ("max_leaf_nodes", "max_iter", "min_samples_leaf")
GRADE = [dict(zip(HP, c)) for c in itertools.product([7, 15, 31], [100, 300], [20, 50, 100, 200])]
FIXOS = dict(loss="quantile", quantile=ALPHA, learning_rate=0.05, early_stopping=False,
             random_state=0)
MIN_SAMPLES_LEAF_ANTIGO = 100   # valor fixo antes de entrar na grade
EXCLUIDAS = ("g_adv_",)   # advecção, fora do modelo desde 22/10/2026
COM_ALTITUDE = False      # altitude fora do modelo desde 22/10/2026


def hiperparametros(arquivo, regiao, braco="com_grafo"):
    """A combinação escolhida para (regiao, braco) num cv_resumo.csv."""
    linha = pd.read_csv(arquivo).set_index(["regiao", "braco"]).loc[(regiao, braco)]
    return {k: int(linha[k]) if k in linha else MIN_SAMPLES_LEAF_ANTIGO for k in HP}


def bracos(colunas, excluir=EXCLUIDAS, com_altitude=COM_ALTITUDE):
    locais = [c for c in colunas if c.endswith("_d1")] + ["day_sin", "day_cos"]
    if com_altitude:
        locais.append("altitude")
    grafo = [c for c in colunas if c.startswith("g_") and not c.startswith(excluir)]
    return {"sem_grafo": locais, "com_grafo": locais + grafo}


def climatologia(treino, valid):
    """p90 por âncora e mês nos anos de treino; com poucos dias, o p90 da âncora."""
    por_mes = treino.groupby(["anchor_id", "month"]).tp_mm.agg(
        p90=lambda s: s.quantile(ALPHA), n="size")
    por_mes.loc[por_mes.n < MIN_DIAS_CLIMA, "p90"] = np.nan
    geral = treino.groupby("anchor_id").tp_mm.quantile(ALPHA).rename("p90_geral")
    v = valid[["anchor_id", "month"]].join(por_mes.p90, on=["anchor_id", "month"])
    v = v.join(geral, on="anchor_id")
    return v.p90.fillna(v.p90_geral).to_numpy()


def avaliar(regiao, definir_bracos=bracos, grade=GRADE):
    d = pd.read_parquet(DADOS / f"features_{regiao}.parquet")
    d = d[(d.tp_mm >= LIMIAR_MM) & (d.date.dt.year <= 2014)].reset_index(drop=True)
    d["month"] = d.date.dt.month
    ano = d.date.dt.year
    linhas = []
    for ini, fim in BLOCOS:
        treino, valid = d[ano < ini], d[(ano >= ini) & (ano <= fim)]
        y = valid.tp_mm.to_numpy()
        perda_clima = mean_pinball_loss(y, climatologia(treino, valid), alpha=ALPHA)
        for braco, cols in definir_bracos(d.columns).items():
            for i, hp in enumerate(grade):
                m = HistGradientBoostingRegressor(**hp, **FIXOS).fit(treino[cols], treino.tp_mm)
                perda = mean_pinball_loss(y, m.predict(valid[cols]), alpha=ALPHA)
                linhas.append(dict(regiao=regiao, braco=braco, combo=i, **hp, bloco=f"{ini}-{fim}",
                                   n_treino=len(treino), n_valid=len(valid), n_features=len(cols),
                                   perda=perda, perda_clima=perda_clima,
                                   ganho=1 - perda / perda_clima))
    return pd.DataFrame(linhas)


def resumir(r):
    """Melhor combinação de cada braço (menor perda média) e o ganho médio dela nos blocos."""
    media = r.groupby(["regiao", "braco", "combo"]).perda.mean()
    melhor = media.groupby(["regiao", "braco"]).idxmin()
    sel = r.set_index(["regiao", "braco", "combo"]).loc[melhor.values].reset_index()
    res = sel.groupby(["regiao", "braco"]).agg(
        max_leaf_nodes=("max_leaf_nodes", "first"), max_iter=("max_iter", "first"),
        min_samples_leaf=("min_samples_leaf", "first"),
        n_features=("n_features", "first"), dias=("n_valid", "sum"),
        perda=("perda", "mean"), perda_clima=("perda_clima", "mean"), ganho=("ganho", "mean"),
        blocos_vence_clima=("ganho", lambda g: int((g > 0).sum())))
    por_bloco = sel.pivot_table(index=["regiao", "bloco"], columns="braco", values="perda")
    venceu = (por_bloco.com_grafo < por_bloco.sem_grafo).groupby("regiao").sum()
    return res.reset_index(), venceu


def tabela_md(res, venceu):
    w = res.pivot(index="regiao", columns="braco")
    linhas = ["| região | dias ≥10 mm validados | ganho sem grafo | ganho com grafo | diferença "
              "(p.p.) | blocos em que o grafo vence | perda clima | perda sem | perda com |",
              "|---|---|---|---|---|---|---|---|---|"]
    for reg in REGIOES:
        if reg not in w.index:
            continue
        g_s, g_c = w.loc[reg, ("ganho", "sem_grafo")], w.loc[reg, ("ganho", "com_grafo")]
        linhas.append(
            f"| {reg} | {int(w.loc[reg, ('dias', 'sem_grafo')]):,} | {100 * g_s:+.1f}% | "
            f"{100 * g_c:+.1f}% | {100 * (g_c - g_s):+.1f} | {int(venceu[reg])}/5 | "
            f"{w.loc[reg, ('perda_clima', 'sem_grafo')]:.3f} | "
            f"{w.loc[reg, ('perda', 'sem_grafo')]:.3f} | {w.loc[reg, ('perda', 'com_grafo')]:.3f} |")
    return "\n".join(linhas)


if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    partes = []
    for regiao in REGIOES:
        partes.append(avaliar(regiao))
        pd.concat(partes).to_parquet(SAIDA / "cv_blocos.parquet", index=False)
        print(f"{regiao} ok ({time.time() - t0:.0f}s)", flush=True)
    r = pd.concat(partes)
    res, venceu = resumir(r)
    res.to_csv(SAIDA / "cv_resumo.csv", index=False)
    md = tabela_md(res, venceu)
    (SAIDA / "tabela_cv.md").write_text(md + "\n")
    print(md)
