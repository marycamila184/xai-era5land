"""Climatologia de cada âncora num CSV simples, para quem quer agrupar os pontos (k-means).

Uma linha por âncora, com município, UF, região, latitude e longitude da célula, altitude e
três médias da célula da âncora (linha `center`) de 1980-01-01 a 2025-12-30 (16.801 dias):
- temperatura_media_c e ponto_orvalho_medio_c: média dos valores diários de t2m e d2m (°C);
- chuva_media_anual_mm: cada linha do parquet já é o acumulado do dia (o passo das 00 UTC de
  D+1 do processed_daily), então soma-se o tp_mm dos ~365 dias de cada ano e tira-se a média
  dos 46 totais anuais. Em 2025 falta o dia 31/12, que o dataset não tem.
Colunas em português e sem arredondar, a pedido dela (30/10/2026), porque o arquivo vai para
fora do projeto.

Uso (na raiz do projeto, depois de gerar_dataset_amostra.py):
    uv run python av1/amostra_brasil/climatologia_ancoras.py --anel 150km
    -> dados_150/ancoras_brasil_150km.csv
Testado em testar_climatologia_ancoras.py, que recalcula tudo direto dos NetCDF.
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz
from av1.amostra_brasil.ancoras import anel_da_linha_de_comando  # noqa: E402

ANEL, CONFIG = anel_da_linha_de_comando()
DADOS = CONFIG["dados"]
SAIDA = DADOS / f"ancoras_brasil_{ANEL}.csv"
NOMES = {"anchor_id": "codigo_ibge", "municipality": "municipio", "state": "uf",
         "region": "regiao", "altitude": "altitude_m"}


def main():
    ancoras = pd.read_csv(DADOS / "ancoras.csv")
    d = pd.concat(ds.dataset(f).to_table(columns=["date", "anchor_id", "tp_mm", "t2m", "d2m"],
                                         filter=ds.field("position") == "center").to_pandas()
                  for f in sorted(DADOS.glob("era5land_amostra_*.parquet")))
    print(f"{d.anchor_id.nunique()} âncoras, {d.date.min():%Y-%m-%d} a {d.date.max():%Y-%m-%d}, "
          f"{len(d) // d.anchor_id.nunique()} dias por âncora")

    ano = d.date.dt.year
    clima = d.groupby("anchor_id").agg(temperatura_media_c=("t2m", "mean"),
                                       ponto_orvalho_medio_c=("d2m", "mean"))
    clima["chuva_media_anual_mm"] = (d.groupby([d.anchor_id, ano]).tp_mm.sum()
                                     .groupby("anchor_id").mean())

    saida = (ancoras[list(NOMES) + ["latitude", "longitude"]]
             .merge(clima.reset_index(), on="anchor_id", how="left")
             .rename(columns=NOMES)
             [["codigo_ibge", "municipio", "uf", "regiao", "latitude", "longitude", "altitude_m",
               "temperatura_media_c", "ponto_orvalho_medio_c", "chuva_media_anual_mm"]])
    if len(saida) != len(ancoras) or saida.isna().any().any():
        raise ValueError("âncora sem climatologia")
    saida.to_csv(SAIDA, index=False)
    print(f"{len(saida)} linhas -> {SAIDA}")


if __name__ == "__main__":
    main()
