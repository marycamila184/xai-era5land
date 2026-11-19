# AV1: experimento com e sem grafo

Primeiro experimento da proposta (slides 6 a 9). A pergunta é se as features de grafo, tiradas
das 8 vizinhas do anel de ~150 km, ajudam o boosting quantílico a prever o p90 da chuva de
amanhã na âncora.

## Desenho

| | |
|---|---|
| dados | `av1/amostra_brasil/dados_150/` (512 âncoras, 1980 a 2025) |
| alvo | tp da âncora no dia t, só nos dias com 10 mm ou mais, como na AT3 |
| features | da véspera (t−1), menos o dia do ano (do dia t) |
| modelo | `HistGradientBoostingRegressor(loss="quantile", quantile=0.9)`, um por região |
| validação | cinco blocos de quatro anos, de 1995 a 2014, com janela expansiva a partir de 1980 |
| teste | 2015 a 2025, em `comparar_desenhos.py`, só depois de fechar as features |
| referência | climatologia mensal do p90 de cada âncora, tirada dos anos de treino de cada bloco |
| métrica | ganho da perda quantílica (*pinball loss*) sobre a climatologia |

Os dois braços:

- **sem grafo**: as variáveis da própria âncora na véspera (`<var>_d1`) e o seno e o cosseno do
  dia do ano, num total de 17 features.
- **com grafo**: as mesmas mais 31 features `g_*` das vizinhas na véspera, num total de 48:
  1. agregados do anel: média, desvio e média menos âncora de tp, t2m, d2m, dpd, ssrd, pev e
     velocidade do vento; máximo e fração de vizinhas com mais de 1 mm para o tp;
  2. gradientes de t2m e d2m (°C/100 km), do plano ajustado por mínimos quadrados às 8 vizinhas;
  3. convergência e vorticidade do vento a 10 m (10⁻⁵ s⁻¹).

  A advecção de t2m e d2m pelo vento da âncora (°C/dia) continua no `montar_features.py`, mas
  saiu do modelo em 22/10/2026. Na ablação, tirá-la mudou o ganho em menos de 0,1 p.p. em todas
  as regiões e piorou em só 11 dos 25 blocos. Os gradientes e os agregados ficaram: cada um
  sozinho pode sair, mas tirar os dois juntos custava de 0,3 a 0,8 p.p. no Centro-Oeste, no
  Sudeste e no Sul, porque um cobre o outro. Os resultados com as 51 features estão em
  `resultados/antes_sem_adveccao/`.

  A altitude também saiu do modelo em 22/10/2026. Na validação espacial-temporal
  (`validacao_espacial.py`), ela ajudava em todos os blocos quando o modelo já conhecia a
  âncora, mas em lugares novos essa ajuda sumia no Norte, no Centro-Oeste e no Sudeste: em
  parte, ela servia para reconhecer a âncora. Sem ela, o resultado vale para lugares novos sem
  essa ajuda. O custo medido em lugares novos é de 0,6 a 0,7 p.p. no Sul, onde ela carrega a
  serra, e quase zero nas outras regiões. Faixas de 200 m não resolviam (iguais ou piores que a
  altitude bruta).

O tp das vizinhas entra sem peso por vento. Na véspera dos dias fortes do Sul, o vento a 10 m
vem do norte e do nordeste (60% dos dias), mas a chuva chega pelo oeste e pelo sudoeste, então
pesar pelo vento de superfície dava peso ao lado errado (medido em 22/10/2026). O vento entra
pela convergência e pela vorticidade.

Os dois braços usam a mesma grade (`max_leaf_nodes` 7, 15 ou 31; 100 ou 300 árvores;
`min_samples_leaf` 20, 50, 100 ou 200 dias por folha; 24 combinações; taxa de aprendizado
0,05 fixa), e cada um fica com a combinação de menor perda média nos blocos. O mínimo de dias
por folha era 100 fixo até 22/10/2026, quando entrou na grade para ser escolhido pela
validação; os resultados de antes disso usam 100.

## Arquivos

| caminho | o que é |
|---|---|
| `montar_features.py` | monta `dados/features_<regiao>.parquet`, com uma linha por âncora e dia (cerca de 30 s) |
| `treinar_p90.py` | roda os dois braços e grava `resultados/cv_blocos.parquet`, `cv_resumo.csv` e `tabela_cv.md` (cerca de 25 min) |
| `ablacao.py` | ablação por grupo de features de grafo (sobre as 51), versão rápida com os hiperparâmetros do braço com grafo → `resultados/ablacao_resumo.csv` (cerca de 6 min) |
| `comparar_desenhos.py` | teste: o desenho atual (treino 1980–2014) em 2015–2025 e em 2024–2025, contra o de 10 anos do Vitor (treino 2016–2023, teste 2024–2025), com ganho e cobertura → `resultados/comparacao_desenhos.csv` |
| `validacao_espacial.py` | validação espacial-temporal (5 grupos contíguos de âncoras por região; cada bloco de anos valida um grupo que ficou fora do treino) contra a temporal, com e sem altitude e com faixas → `resultados/validacao_espacial_resumo.csv` (cerca de 8 min) |
| `matriz_regioes.py` | matriz de generalização entre regiões (slide 9), sem e com grafo: treino em X (1980–2014), teste em Y (2015–2025); na diagonal, âncoras que o modelo não viu (um modelo por grupo espacial deixado de fora) → `resultados/matriz_regioes.csv` |
| `shap_teste.py` | SHAP dos modelos finais sem e com grafo (treino 1980–2014) em 3.000 dias fortes de 2015–2025 por região → `resultados/shap_teste_features.csv` e `shap_teste_grupos.csv` |
| `shap_global.py` | SHAP do modelo completo sem e com grafo (treino em todas as âncoras, 1980–2014) em todos os dias fortes de 2015–2025, no Brasil todo e por região → `resultados/shap_global_features.csv` e `shap_global_grupos.csv` (cerca de 9 min) |
| `modelo_global.py` | modelo global (as cinco regiões juntas), hiperparâmetros escolhidos na mesma grade de 24 dos regionais (cerca de 25 min), testado em âncoras que não viu → `resultados/global_teste.csv` (entra como linha da matriz) |
| `transferencia_shap.py` | queda na transferência (slide 10): perfis de |SHAP| de cada modelo regional em todas as regiões e a relação com a queda de ganho → `resultados/transferencia_*.csv` e `figuras/transferencia_shap.png` |
| `cp_pdp.py` | perfis ceteris paribus dos quatro dias da cascata com o PDP da região → `figuras/cp_pdp_dias.png` (cerca de 7 min) |
| `figura_grade.py` | `figuras/grade_brasil.png`, a grade do ERA5-Land no Brasil e as âncoras, para a seção de Dados |
| `shap_local.py` | SHAP local de quatro dias do teste (critério fixo) e dependência do SHAP → `figuras/shap_local_dias.png` e `shap_dependencia.png` |
| `figuras_resultados.py` | `figuras/matriz_regioes.png`, `shap_teste_grupos.png` e `shap_teste_features.png` |
| `figura_grafo.py` | `figuras/grafo_ancora.png`, o grafo de uma âncora e a extração das variáveis de grafo num dia real (Itaqui, véspera de 29/07/1994) |
| `figuras_validacao.py` | figuras que explicam a validação espacial-temporal → `figuras/estrategia_sul.png`, `temporal_vs_espacial.png` e `grupos_espaciais.png` (segundos) |
| `shap_grafo.py` | SHAP do modelo com grafo (treino até 2010, explica 3.000 dias fortes de 2011–2014 por região) → `resultados/shap_grupos.csv` e `shap_features.csv` |

## Ordem de execução

Na raiz do projeto:

```bash
uv run python av1/experimentos/montar_features.py
uv run python av1/experimentos/treinar_p90.py
uv run python av1/experimentos/ablacao.py          # precisa de resultados/antes_sem_adveccao/cv_resumo.csv
uv run python av1/experimentos/comparar_desenhos.py
uv run python av1/experimentos/shap_grafo.py
uv run python av1/experimentos/validacao_espacial.py
uv run python av1/experimentos/figuras_validacao.py
uv run python av1/experimentos/figura_grafo.py
uv run python av1/experimentos/matriz_regioes.py
uv run python av1/experimentos/shap_teste.py
uv run python av1/experimentos/modelo_global.py
uv run python av1/experimentos/figuras_resultados.py
uv run python av1/experimentos/transferencia_shap.py
uv run python av1/experimentos/shap_local.py
```

Não rode dois desses scripts ao mesmo tempo: o boosting usa todos os núcleos, e dois juntos
ficam muito mais lentos que um depois do outro.

`dados/` e `resultados/` ficam fora do git (o `.gitignore` ignora parquet e csv) e se refazem
com os comandos acima.
