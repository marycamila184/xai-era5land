# era5land_amostra_<regiao>.parquet

Dataset diário do ERA5-Land amostrado no Brasil, **anel de 150 km**. Cada âncora é a célula
da sede de um município, e vem com as 8 células vizinhas a ~123–145 km. O mesmo desenho do
anel de 30 km (`../dados/README_era5land_amostra.md`), só com as vizinhas mais longe. Todas as colunas vêm do ERA5-Land e
são do mesmo dia. O alvo é a chuva da âncora (`tp_mm` na linha `center`).

| | |
|---|---|
| arquivos | `era5land_amostra_norte`, `_nordeste`, `_sudeste`, `_sul`, `_centro_oeste` (`.parquet`) |
| âncoras | 512: Norte 130, Nordeste 116, Sudeste 93, Sul 60, Centro-Oeste 113 |
| linhas | 77.419.008 no total (512 âncoras × 9 células × 16.801 dias) |
| colunas | 32 |
| uma linha é | uma célula do ERA5-Land (a âncora ou uma vizinha) em um dia UTC |
| chave | `date` + `anchor_id` + `position` |
| período | 1980-01-01 a 2025-12-30 |
| valores faltando | nenhum (`wind_const` falta em calmaria total, que não ocorreu nestas células) |
| gerado por | `av1/amostra_brasil/gerar_dataset_amostra.py --anel 150km`, em 21/10/2026 |
| testado por | `av1/amostra_brasil/testar_dataset_amostra.py --anel 150km`, em 21/10/2026: tudo ok (113 checagens) |
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
- **Vizinhas:** 8 células a ~123–145 km, o mais perto possível da mesma distância. Em células
  de 0,1°, (linha, coluna), com a linha crescendo para o sul:

  | n | ne | e | se | s | sw | w | nw |
  |---|---|---|---|---|---|---|---|
  | (−13, 0) | (−9, +9) | (0, +13) | (+9, +9) | (+13, 0) | (+9, −9) | (0, −13) | (−9, −9) |

  A distância real vai de 123,4 a 144,6 km (`dist_km`). Por que 150 km: no correlograma do tp
  diário (`../correlograma_tp.py`, 21/10/2026), a vizinha a 30 km tem correlação mediana de
  0,93 com a âncora, quase uma cópia; a 100–150 km, 0,71.
- **Litoral:** saiu a âncora com alguma das 9 células sem dado em qualquer variável (mar).
  Uma vizinha fora do Brasil é aceita, porque o ERA5-Land tem dado lá.
- **Sem colisão:** as âncoras ficam a pelo menos 7 células uma da outra, na maior das duas
  direções, e duas âncoras não dividem nenhuma célula. Vale no país inteiro. Os anéis podem
  passar por dentro da área de outras âncoras: a mediana é de 5 outras âncoras a até 145 km
  (Norte 2, Centro-Oeste 4, Nordeste 5, Sudeste 6, Sul 7). Exigir anéis sem sobreposição
  (âncoras a 27 células) deixaria 68 âncoras.
- **Quantas:** o máximo que cabe em cada região, escolhido pelo guloso de menor grau
  (desempate pelo menor código do IBGE, então rodar de novo dá as mesmas âncoras). De 5.570
  sedes, 3.881 passam pelo litoral (com o anel maior, mais âncoras perdem uma célula no mar)
  e 512 viram âncora.
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

sul = pd.read_parquet("av1/amostra_brasil/dados_150/era5land_amostra_sul.parquet")

# só as âncoras (o alvo)
alvo = sul[sul.position == "center"]

# formato largo: uma linha por âncora e dia, com as vizinhas em colunas (t2m_n, t2m_ne, ...)
largo = sul.pivot_table(index=["date", "anchor_id"], columns="position",
                        values=["tp_mm", "t2m"], observed=True)
largo.columns = [f"{v}_{p}" for v, p in largo.columns]
```

Um arquivo de região tem de 9 a 20 milhões de linhas; para o Brasil todo, leia as regiões
uma de cada vez ou só as colunas necessárias (`columns=[...]`).

## Grupos espaciais para validação (`grupos_espaciais.csv`)

Uma linha por âncora (512), com o grupo espacial usado na validação espacial-temporal da AV1
(`av1/experimentos/validacao_espacial.py`, 22/10/2026). Em cada região, as âncoras foram
divididas em 5 grupos contíguos por k-means nas coordenadas em km (semente fixa, então rodar de
novo dá os mesmos grupos). As 8 vizinhas de uma âncora seguem o grupo dela: junte pelo
`anchor_id`.

| coluna | descrição |
|---|---|
| `anchor_id`, `municipality`, `state`, `region`, `row_era5`, `col_era5`, `latitude`, `longitude` | a âncora, como em `ancoras.csv` |
| `region_file` | o sufixo do parquet da região (`era5land_amostra_<region_file>.parquet`) |
| `spatial_group` | o grupo, de 0 a 4 (dentro da região) |
| `validation_block`, `validation_start`, `validation_end` | o bloco de anos em que esse grupo é validado |

O protocolo: o grupo i é validado no bloco i (0: 1995–1998, 1: 1999–2002, 2: 2003–2006,
3: 2007–2010, 4: 2011–2014), com um modelo treinado só nas âncoras dos outros grupos e só nos
anos antes do bloco. O teste (2015–2025) fica de fora. Como o treino é sempre anterior ao
bloco, duas âncoras vizinhas nunca ficam uma no treino e outra na validação no mesmo dia.
As figuras estão em `av1/experimentos/figuras/`.

```python
grupos = pd.read_csv("av1/amostra_brasil/dados_150/grupos_espaciais.csv")
sul = sul.merge(grupos[["anchor_id", "spatial_group"]], on="anchor_id")
ano = sul.date.dt.year
for i, (ini, fim) in enumerate([(1995, 1998), (1999, 2002), (2003, 2006), (2007, 2010), (2011, 2014)]):
    treino = sul[(ano < ini) & (sul.spatial_group != i)]
    valid = sul[(ano >= ini) & (ano <= fim) & (sul.spatial_group == i)]
```

## Cuidados

- Os cuidados do ERA5-Land como alvo valem aqui também: é
  chuva de área (~110 km²), não de ponto, e descreve a chuva do ERA5-Land, não a observada.
- O litoral fica vazio: uma sede a menos de ~145 km do mar não vira âncora (ver
  `../figuras_150/mapa_ancoras.png`).
- Como os anéis se sobrepõem, as linhas de âncoras próximas no mesmo dia são ainda menos
  independentes que no anel de 30 km (onde duas âncoras a 70 km já têm correlação de tp de ~0,84).
  O corte de treino e teste por data não vaza informação; um corte por âncora vazaria.
- A altitude não foi critério de escolha. A conferência está em
  `../figuras_150/altitude_ancoras_vs_sedes.png`: no Nordeste as âncoras ficam bem mais altas
  que as sedes (mediana de 390 contra 253 m; no anel de 30 km, 330), porque o litoral saiu;
  no Centro-Oeste mais baixas (395 contra 474 m).
- A `altitude` do ERA5-Land vem do geopotencial suavizado do modelo e fica negativa em 5
  vizinhas (nenhuma âncora): no Pará, junto ao rio Amazonas (−4 m), e na costa do ES, de SP e
  do PR, ao pé da Serra do Mar (até −18 m, na baía de Paranaguá). São células de terra, com
  dado em todas as variáveis.
- O município é rótulo da âncora. Em municípios grandes (Norte), a âncora é uma fatia pequena
  do município (`municipality_area_km2`).
