"""Escolhe as âncoras do dataset amostrado no Brasil.

1. Sedes dos municípios: IBGE, Localidades do Brasil 2022 (sede municipal, capital estadual
   e capital federal, uma por município). Área: API de malhas do IBGE.
2. Âncora = célula de 0,1° da sede. Sai a âncora que tenha alguma das 9 células (ela e as 8
   vizinhas do anel) sem dado em qualquer variável do dataset (mar).
3. Maior conjunto sem colisão (âncoras a >= 7 células e sem célula em comum), guloso por
   menor grau, desempate pelo menor código do IBGE. Vale no país inteiro.

Saídas na pasta do anel (av1/amostra_brasil/dados/ no de 30 km, dados_150/ no de 150 km):
    ancoras.csv   uma linha por âncora
    celulas.csv   9 linhas por âncora (center e as 8 vizinhas)
    sedes.csv     todas as sedes, com a altitude da célula (para a conferência)

Uso (na raiz do projeto):
    uv run python av1/amostra_brasil/escolher_ancoras.py               # anel de 30 km
    uv run python av1/amostra_brasil/escolher_ancoras.py --anel 150km
"""
import json
import sqlite3
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # raiz, para av1.amostra_brasil
from av1.amostra_brasil.ancoras import (COL0, REGIOES, ROW0, anel_da_linha_de_comando,  # noqa: E402
                                         baixar_json, celula_era5, centro_da_celula, distancia_km,
                                         empacotar, tem_dado)

ANEL, CONFIG = anel_da_linha_de_comando()
POSICOES = CONFIG["posicoes"]
DADOS = CONFIG["dados"]
IBGE = Path("av1/amostra_brasil/dados/ibge")          # o mesmo download para os dois anéis
URL_LOCALIDADES = ("https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/"
                   "localidades/Localidades_do_Brasil/2022/Localidades_Brasil_gpkg.zip")
URL_AREAS = "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR/metadados?intrarregiao=municipio"
ZIP = IBGE / "Localidades_Brasil_gpkg.zip"
GPKG = IBGE / "BR_localidades_2022.gpkg"
AREAS = IBGE / "areas_municipios.json"
ERA5 = Path("/media/mary-camila/Expansion/era5land")
MASCARA = {                                   # arquivo -> variáveis que precisam ter dado
    ERA5 / "processed_daily/tp/tp_1980_01.nc": ["tp"],
    ERA5 / "processed_daily/ssrd/ssrd_1980_01.nc": ["ssrd"],
    ERA5 / "processed_daily/pev/pev_1980_01.nc": ["pev"],
    ERA5 / "processed_daily/temp/temp_1980_01.nc": ["t2m", "d2m", "t2m_min", "t2m_max"],
    ERA5 / "raw/wind/wind_1980_01.nc": ["u10", "v10"],
}
ESTATICO = Path("commons/data/static_era5land.nc")
SEDES = ("Sede Municipal", "Capital Estadual", "Capital Federal")


def baixar_fontes():
    IBGE.mkdir(parents=True, exist_ok=True)
    if not GPKG.exists():
        if not ZIP.exists():
            urllib.request.urlretrieve(URL_LOCALIDADES, ZIP)
        with zipfile.ZipFile(ZIP) as z:
            z.extract(GPKG.name, IBGE)
    if not AREAS.exists():
        AREAS.write_text(json.dumps(baixar_json(URL_AREAS)))


def ler_sedes():
    con = sqlite3.connect(f"file:{GPKG}?mode=ro", uri=True)
    sedes = pd.read_sql(
        "select CD_MUN, NM_MUN, SIGLA_UF, SCT_LOCALIDADE, LAT_LOCALIDADE, LONG_LOCALIDADE "
        f"from BR_localidades_2022 where SCT_LOCALIDADE in {SEDES}", con)
    con.close()
    repetidas = sedes.CD_MUN.duplicated().sum()
    sedes = sedes.drop_duplicates("CD_MUN")
    area = {m["id"]: float(m["area"]["dimensao"]) for m in json.loads(AREAS.read_text())}
    sedes = pd.DataFrame({
        "municipality_code": sedes.CD_MUN.astype(int).values,
        "municipality": sedes.NM_MUN.values,
        "state": sedes.SIGLA_UF.values,
        "region": sedes.CD_MUN.str[0].astype(int).map(REGIOES).values,
        "municipality_area_km2": sedes.CD_MUN.map(area).values,
        "lat_sede": sedes.LAT_LOCALIDADE.values,
        "lon_sede": sedes.LONG_LOCALIDADE.values,
    })
    print(f"sedes do IBGE: {len(sedes)} municípios ({repetidas} linhas repetidas descartadas); "
          f"sem área: {sedes.municipality_area_km2.isna().sum()}")
    return sedes


def mascara_de_dado():
    mascara = None
    for arquivo, nomes in MASCARA.items():
        with xr.open_dataset(arquivo) as ds:
            for n in nomes:
                ok = np.isfinite(ds[n].isel(valid_time=1).values)
                mascara = ok if mascara is None else mascara & ok
    if mascara.shape != (411, 451):
        raise ValueError(f"janela inesperada: {mascara.shape}")
    print(f"células com dado em todas as variáveis: {mascara.sum()}")
    return mascara


def main():
    baixar_fontes()
    sedes = ler_sedes()
    sedes["row_era5"], sedes["col_era5"] = celula_era5(sedes.lat_sede, sedes.lon_sede)
    with xr.open_dataset(ESTATICO) as st:
        alt = st.altitude.values
    iy, ix = sedes.row_era5 - ROW0, sedes.col_era5 - COL0
    dentro = (iy >= 0) & (iy < alt.shape[0]) & (ix >= 0) & (ix < alt.shape[1])
    sedes["altitude"] = np.where(dentro, alt[iy.clip(0, alt.shape[0] - 1), ix.clip(0, alt.shape[1] - 1)], np.nan)

    mascara = mascara_de_dado()
    cand = sedes[tem_dado(sedes.row_era5, sedes.col_era5, mascara, POSICOES)].reset_index(drop=True)
    anc = cand[empacotar(cand.row_era5, cand.col_era5, cand.municipality_code, POSICOES)].copy()
    anc = anc.sort_values("municipality_code").reset_index(drop=True)
    anc.insert(0, "anchor_id", anc.municipality_code)
    anc["latitude"], anc["longitude"] = centro_da_celula(anc.row_era5, anc.col_era5)

    partes = []
    for pos, (dr, dc) in POSICOES.items():
        p = pd.DataFrame({"anchor_id": anc.anchor_id, "position": pos,
                          "row_era5": anc.row_era5 + dr, "col_era5": anc.col_era5 + dc})
        p["latitude"], p["longitude"] = centro_da_celula(p.row_era5, p.col_era5)
        p["dist_km"] = distancia_km(anc.latitude, anc.longitude, p.latitude, p.longitude).round(2)
        p["altitude"] = alt[p.row_era5 - ROW0, p.col_era5 - COL0]
        p["_ordem"] = list(POSICOES).index(pos)
        partes.append(p)
    cel = pd.concat(partes).sort_values(["anchor_id", "_ordem"]).drop(columns="_ordem").reset_index(drop=True)

    resumo = pd.concat([sedes.region.value_counts().rename("sedes"),
                        cand.region.value_counts().rename("fora do litoral"),
                        anc.region.value_counts().rename("âncoras")], axis=1).reindex(list(REGIOES.values()))
    resumo.loc["Brasil"] = resumo.sum()
    print(f"anel {ANEL}")
    print(resumo.to_string())
    print(f"dist_km das vizinhas: {cel[cel.position != 'center'].dist_km.min():.1f} a "
          f"{cel[cel.position != 'center'].dist_km.max():.1f}")

    DADOS.mkdir(parents=True, exist_ok=True)
    anc[["anchor_id", "municipality", "municipality_code", "state", "region", "municipality_area_km2",
         "lat_sede", "lon_sede", "row_era5", "col_era5", "latitude", "longitude", "altitude"]
        ].to_csv(DADOS / "ancoras.csv", index=False)
    cel.to_csv(DADOS / "celulas.csv", index=False)
    sedes.to_csv(DADOS / "sedes.csv", index=False)
    print(f"gravado: {DADOS}/ancoras.csv ({len(anc)}), celulas.csv ({len(cel)}), sedes.csv ({len(sedes)})")


if __name__ == "__main__":
    main()
