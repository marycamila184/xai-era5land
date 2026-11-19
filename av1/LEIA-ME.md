# AV1: explicabilidade na previsão de chuva com ERA5-Land no Brasil

Código do pôster da 70ª RBras ([../papers/rbras70_poster.pdf](../papers/rbras70_poster.pdf))
e da versão estendida, o artigo da AV1 ([../papers/av1_artigo.pdf](../papers/av1_artigo.pdf)).

## O que tem aqui

| caminho | o que é | no artigo |
|---|---|---|
| `static_fields.py` | gera `commons/data/static_era5land.nc` (altitude e máscara terra-mar), usado na escolha das âncoras. O uso está na docstring | Seção II |
| `amostra_brasil/` | as 512 âncoras nas sedes municipais, com as 8 vizinhas do anel de 150 km, e o dataset diário de 1980 a 2025. Comece por `amostra_brasil/LEIA-ME.md` | Seção II, Figura 1 |
| `experimentos/` | experimento 2, o p90 da chuva forte do dia seguinte, com validação, SHAP, CP e PDP, e as figuras dos dois experimentos. Comece por `experimentos/LEIA-ME.md` | Seções III-D e IV-B a IV-E |
| `R atualizado/` | experimento 1, do Vitor. A floresta aleatória (ranger) que estima a chuva do Sul no mesmo dia só a partir das vizinhas, com a grade de hiperparâmetros e a validação espaço-temporal | Seções III-C e IV-A |
| `R atualizado pdp e ice/` | experimento 1, do Vitor. KernelSHAP, perfis CP e PDP das quatro observações escolhidas | Seção IV-A, Figuras 8 a 10 |

## Ordem para reproduzir

Na raiz do projeto, com o ERA5-Land já agregado ao dia (ver [../DATA.md](../DATA.md)):

```bash
uv run python av1/static_fields.py <geopotencial.nc> <mascara_terra_mar.nc>
uv run python av1/amostra_brasil/escolher_ancoras.py --anel 150km
uv run python av1/amostra_brasil/gerar_dataset_amostra.py --anel 150km
uv run python av1/amostra_brasil/testar_dataset_amostra.py --anel 150km
```

Depois, os scripts de `experimentos/`, na ordem do `experimentos/LEIA-ME.md`.

O correlograma que justifica o anel de 150 km (`amostra_brasil/correlograma_tp.py`) usa o
anel de 30 km, gerado pelos mesmos scripts sem o `--anel 150km`.

O experimento 1 roda em R. O `interpretacao_dos_grafos_150.R` lê o parquet do Sul gerado
acima e os grupos espaciais de `experimentos/resultados/grupos_espaciais.csv`, e o
`treino_ancoras_150.R` treina e valida a floresta. Os caminhos no início dos scripts são os
da máquina do Vitor. Os números que ele salvou estão em
`experimentos/resultados/exp1/`, e as figuras do artigo saem de
`experimentos/figuras_exp1.py`.

## Cuidados ao usar o ERA5-Land como alvo

Medidos em 16/10/2026 na Região Metropolitana de Curitiba, contra pluviômetros.

- **É chuva de área**, de uma célula de cerca de 110 km², e não de um ponto. Os extremos
  dele são menores que os de um pluviômetro, em parte por escala.
- **Os resultados descrevem a chuva do ERA5-Land, não a observada.** Contra os
  pluviômetros, a correlação diária é de cerca de 0,55, e nos dias de 30 mm ou mais ele
  marca cerca de um terço do valor. O verão convectivo é o pior (correlação de 0,42).
