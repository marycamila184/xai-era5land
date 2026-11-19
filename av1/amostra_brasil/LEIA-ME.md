# AV1: dataset ERA5-Land amostrado no Brasil

Âncoras nas sedes dos municípios, nas cinco regiões do IBGE, cada uma com 8 vizinhas num
anel. Serve para medir o quanto as células vizinhas ajudam a prever a chuva da âncora.

Há dois anéis, cada um na sua pasta, escolhidos com `--anel` em todos os scripts:

| anel | vizinhas | âncoras | pasta |
|---|---|---|---|
| `30km` (padrão) | n/s/e/w a 3 células, diagonais 2 + 2 (28 a 33 km) | 615 | `dados/`, `figuras/` |
| `150km` | n/s/e/w a 13 células, diagonais 9 + 9 (123 a 145 km) | 512 | `dados_150/`, `figuras_150/` |

Nos dois as âncoras ficam a pelo menos 7 células uma da outra e não dividem célula. O anel de
150 km foi escolhido em 21/10/2026 pelo correlograma do tp (`correlograma_tp.py`): a 30 km a
vizinha tem correlação mediana de 0,93 com a âncora, quase uma cópia, e a 100 a 150 km, de
0,71. Com 150 km os anéis passam por dentro da área de outras âncoras (só não caem nas mesmas
células); exigir anéis sem sobreposição deixaria 68 âncoras.
O desenho e as decisões estão em `docs/superpowers/specs/2026-10-18-amostra-brasil-design.md`
(cópia local, fora do git; a contagem de âncoras ali é a de antes da geração, 613, e a
final é 615).
O dicionário do dataset está em `dados/README_era5land_amostra.md`.

## O que tem aqui

| caminho | o que é |
|---|---|
| `ancoras.py` | a geometria (posições das vizinhas, índices da grade) e a escolha sem colisão. Testado em `tests/test_amostra_brasil.py` |
| `escolher_ancoras.py` | baixa as sedes e as áreas do IBGE e escolhe as âncoras → `dados/ancoras.csv`, `dados/celulas.csv`, `dados/sedes.csv` |
| `era5land_diario.py` | a leitura diária do ERA5-Land, com a agregação de `commons/build_curitiba_daily.py` |
| `gerar_dataset_amostra.py` | gera `dados/era5land_amostra_<regiao>.parquet`, um por região |
| `testar_dataset_amostra.py` | testa o dataset |
| `figuras_amostra.py` | `mapa_ancoras.png` e `altitude_ancoras_vs_sedes.png` na pasta de figuras do anel |
| `correlograma_tp.py` | correlação do tp diário entre a âncora (anel de 30 km) e células de 11 a ~550 km → `dados/correlograma_tp.parquet` |
| `climatologia_ancoras.py` | uma linha por âncora com altitude e médias de 1980 a 2025 (temperatura, ponto de orvalho, chuva anual), colunas em português → `ancoras_brasil_<anel>.csv`. Feito em 30/10/2026 para um colega do professor que faz k-means |
| `testar_climatologia_ancoras.py` | testa esse CSV recalculando as médias direto dos NetCDF (~30 min) |
| `dados/ibge/` | downloads do IBGE (fora do git) |
| `dados/cache/`, `dados_150/cache/` | vento diário por ano, para retomar uma geração interrompida (fora do git; apague se mudar as âncoras) |

## Ordem de execução

Na raiz do projeto:

```bash
uv run pytest tests/test_amostra_brasil.py
uv run python av1/amostra_brasil/escolher_ancoras.py      # menos de 1 min
uv run python av1/amostra_brasil/gerar_dataset_amostra.py # demorado, lê o disco externo
uv run python av1/amostra_brasil/testar_dataset_amostra.py
uv run python av1/amostra_brasil/figuras_amostra.py
```

Para o anel de 150 km, os mesmos quatro scripts com `--anel 150km` no fim. Os downloads do
IBGE ficam em `dados/ibge/` para os dois anéis.

Nada de `dados/` vai para o git (o `.gitignore` ignora `*.parquet`, `*.csv` e `*.json`): nem
os parquets (cerca de 6 GB), nem `ancoras.csv`, `celulas.csv` e `sedes.csv`. Eles se refazem
com os comandos acima, e a escolha das âncoras é determinística enquanto o arquivo de
localidades do IBGE for o mesmo (2022, baixado em 18/10/2026). O disco externo
(`/media/mary-camila/Expansion/era5land/`) precisa estar montado para gerar e testar.
