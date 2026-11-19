# era5land_amostra_<regiao>.parquet

Dataset diário do ERA5-Land amostrado no Brasil. Cada âncora é a célula da sede de um
município, e vem com as 8 células vizinhas a ~30 km. Todas as colunas vêm do ERA5-Land e
são do mesmo dia. O alvo é a chuva da âncora (`tp_mm` na linha `center`).

| | |
|---|---|
| arquivos | `era5land_amostra_norte`, `_nordeste`, `_sudeste`, `_sul`, `_centro_oeste` (`.parquet`) |
| âncoras | 615: Norte 146, Nordeste 159, Sudeste 116, Sul 76, Centro-Oeste 118 |
| linhas | 92.993.535 no total (615 âncoras × 9 células × 16.801 dias) |
| colunas | 32 |
| uma linha é | uma célula do ERA5-Land (a âncora ou uma vizinha) em um dia UTC |
| chave | `date` + `anchor_id` + `position` |
| período | 1980-01-01 a 2025-12-30 |
| valores faltando | só em `wind_const`, nos dias de calmaria total (por definição) |
| gerado por | `av1/amostra_brasil/gerar_dataset_amostra.py`, em 18/10/2026 |
| testado por | `av1/amostra_brasil/testar_dataset_amostra.py`, em 18/10/2026: tudo ok (113 checagens) |
| âncoras e células | `ancoras.csv` (uma linha por âncora) e `celulas.csv` (9 por âncora) |

## Como foi feito

- **Fonte:** ERA5-Land do ECMWF, grade de 0,1°, horário, no disco externo
  (`/media/mary-camila/Expansion/era5land/`).
- **Sedes:** IBGE, Localidades do Brasil 2022 (sede municipal, capital estadual e capital
  federal), baixado em 18/10/2026. São 5.570 municípios (Fernando de Noronha é distrito
  estadual e não entra). A área vem da API de malhas do IBGE; Boa Esperança do Norte (MT),
  criado depois, não tem área, mas também não virou âncora.
- **Âncora:** a célula de 0,1° que contém a sede. O município não é agregado: a âncora
  representa um lugar, e o município é só rótulo.
- **Vizinhas:** 8 células a ~30 km, todas à mesma distância. Em células de 0,1°,
  (linha, coluna), com a linha crescendo para o sul:

  | n | ne | e | se | s | sw | w | nw |
  |---|---|---|---|---|---|---|---|
  | (−3, 0) | (−2, +2) | (0, +3) | (+2, +2) | (+3, 0) | (+2, −2) | (0, −3) | (−2, −2) |

  As diagonais ficam a 2 + 2 células para não irem a ~45 km. A distância real vai de 28,3 a
  33,4 km (`dist_km`). O anel de 10 km ficou de fora: na RMC, a chuva a 11 km tem correlação
  de 0,987 com a da âncora, quase uma cópia.
- **Litoral:** saiu a âncora com alguma das 9 células sem dado em qualquer variável (mar).
  Uma vizinha fora do Brasil é aceita, porque o ERA5-Land tem dado lá.
- **Sem colisão:** as âncoras ficam a pelo menos 7 células uma da outra, na maior das duas
  direções, então nenhuma célula pertence a duas âncoras. Vale no país inteiro. Em km, a
  menor distância entre duas âncoras é 67,9 km (9 pares ficam abaixo de 70 km), porque 0,1°
  de longitude encolhe com a latitude.
- **Quantas:** o máximo que cabe em cada região, escolhido pelo guloso de menor grau
  (desempate pelo menor código do IBGE, então rodar de novo dá as mesmas âncoras). De 5.570
  sedes, 5.076 passam pelo litoral e 615 viram âncora.
- **Dia e agregação:** dia UTC; `tp`, `ssrd` e `pev` são o passo das 00 UTC de D+1 do `processed_daily`;
  temperaturas do `processed_daily`; vento do raw horário com `scripts/utils/wind_stats.daily_wind`.
- **Sem deslocamento:** todas as colunas de uma linha são do dia `date`. Sem t−1 e sem lags.

## Colunas

### Identificação

| coluna | descrição |
|---|---|
| `date` | dia UTC |
| `anchor_id` | código IBGE do município da âncora; igual nas 9 linhas da âncora |
| `position` | `center` (a âncora) ou `n`, `ne`, `e`, `se`, `s`, `sw`, `w`, `nw` |
| `dist_km` | distância do centro da célula ao centro da âncora (0 na `center`) |
| `latitude`, `longitude` | centro da célula |
| `row_era5`, `col_era5` | id da célula na grade global do ERA5-Land, `(90 − lat) × 10` e `(lon mod 360) × 10`, como na RMC |
| `altitude` | altitude da superfície do ERA5-Land na célula (m), de `commons/data/static_era5land.nc` |
| `region`, `state` | região e UF da âncora |
| `municipality`, `municipality_code` | município da âncora (nome e código IBGE). Também nas vizinhas, que podem cair em outro município |
| `municipality_area_km2` | área do município da âncora (km²). Mostra quando a âncora é uma fatia pequena de um município grande |

### Variáveis (iguais às da RMC)

| coluna | descrição |
|---|---|
| `tp_mm` | chuva do dia (mm). Na linha `center`, o alvo |
| `ssrd_wm2` | radiação solar descendente média (W/m²) |
| `pev_mm` | evaporação potencial (mm). Sinal do ECMWF: negativa |
| `t2m`, `t2m_min`, `t2m_max`, `d2m` | temperatura média, mínima e máxima e ponto de orvalho a 2 m (°C) |
| `dpd` | `t2m − d2m` (°C) |
| `u10`, `v10` | média vetorial do vento a 10 m (m/s) |
| `wind_speed`, `wind_max` | velocidade escalar média e máxima (m/s) |
| `wind_res`, `wind_const`, `wind_dir_sin`, `wind_dir_cos` | resultante, constância e direção (ver `scripts/utils/wind_stats.py`) |
| `day_sin`, `day_cos` | dia do ano, ciente do ano bissexto |

## Como usar

```python
import pandas as pd

sul = pd.read_parquet("av1/amostra_brasil/dados/era5land_amostra_sul.parquet")

# só as âncoras (o alvo)
alvo = sul[sul.position == "center"]

# formato largo: uma linha por âncora e dia, com as vizinhas em colunas (t2m_n, t2m_ne, ...)
largo = sul.pivot_table(index=["date", "anchor_id"], columns="position",
                        values=["tp_mm", "t2m"], observed=True)
largo.columns = [f"{v}_{p}" for v, p in largo.columns]
```

Um arquivo de região tem de 11 a 24 milhões de linhas; para o Brasil todo, leia as regiões
uma de cada vez ou só as colunas necessárias (`columns=[...]`).

## Cuidados

- Os cuidados do ERA5-Land como alvo valem aqui também: é
  chuva de área (~110 km²), não de ponto, e descreve a chuva do ERA5-Land, não a observada.
- As outras variáveis (temperatura, orvalho, radiação, vento) das vizinhas ficam muito
  parecidas com as da âncora (correlação acima de 0,95 até ~90 km na RMC). No modelo, as
  colunas repetidas dividem a importância entre si.
- A altitude não foi critério de escolha. A conferência está em
  `figuras/altitude_ancoras_vs_sedes.png`: no Nordeste as âncoras ficam um pouco mais altas
  que as sedes (mediana de 330 contra 253 m), porque o litoral saiu; no Centro-Oeste um pouco
  mais baixas (402 contra 474 m).
- A `altitude` do ERA5-Land vem do geopotencial suavizado do modelo e fica negativa em 12
  vizinhas (nenhuma âncora): junto ao rio Amazonas, no Pará (até −4 m), e na costa do RJ, ao pé
  da Serra do Mar (até −77 m). São células de terra, com dado em todas as variáveis.
- O município é rótulo da âncora. Em municípios grandes (Norte), a âncora é uma fatia pequena
  do município (`municipality_area_km2`).
