"""Campos estáticos do ERA5-Land: altitude e máscara terra-mar.

Lê os arquivos globais de geopotencial e de land-sea mask (netCDF4 da
documentação do ERA5-Land), recorta para a área do projeto, confere o
alinhamento com a grade do `tp` e salva tudo num arquivo só, com um mapa
para conferência.

A saída, commons/data/static_era5land.nc, dá a coluna `altitude` do dataset
de amostra_brasil/ (e a janela que escolher_ancoras.py indexa).

Uso (na raiz do projeto), com o geopotencial e a máscara terra-mar, nessa ordem:
    uv run python av1/static_fields.py \
        /media/mary-camila/Expansion/era5land_documentation/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc \
        /media/mary-camila/Expansion/era5land_documentation/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc
"""
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

if len(sys.argv) != 3:
    sys.exit("uso: static_fields.py <geopotencial.nc> <mascara_terra_mar.nc> (ver a docstring)")
GEO = Path(sys.argv[1])
LSM = Path(sys.argv[2])
TP_GLOB = "/media/mary-camila/Expansion/era5land/processed_daily/tp/tp_*.nc"
SAIDA = Path("commons/data/static_era5land.nc")
FIGURA = Path("commons/data/static_era5land.png")

LAT = slice(6, -35)    # latitude vem de 90 para -90
LON = slice(-75, -30)


def preparar(caminho: Path, var: str) -> xr.DataArray:
    """Abre um campo global, tira o tempo, passa a longitude para -180..180 e recorta."""
    da = xr.open_dataset(caminho)[var].squeeze(drop=True)
    if float(da.longitude.max()) > 180:
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
    return da.sel(latitude=LAT, longitude=LON).astype("float32")


altitude = (preparar(GEO, "z") / 9.80665).rename("altitude")
altitude.attrs = {"units": "m", "long_name": "altitude da superfície do ERA5-Land (z / g)"}

lsm = preparar(LSM, "lsm").rename("lsm")
lsm.attrs = {"units": "1", "long_name": "fração de terra na célula (0 = água, 1 = terra)"}

# Confere se a grade bate com a do tp (mesmas coordenadas, sem meio passo de diferença)
tp = xr.open_dataset(sorted(glob(TP_GLOB))[0])
lat_tp, lon_tp = tp.latitude.values, tp.longitude.values
print(f"grade estática: {altitude.sizes['latitude']} x {altitude.sizes['longitude']}")
print(f"grade do tp:    {lat_tp.size} x {lon_tp.size}")
mesma_grade = (
    altitude.latitude.size == lat_tp.size
    and altitude.longitude.size == lon_tp.size
    and np.allclose(altitude.latitude.values, lat_tp, atol=1e-4)
    and np.allclose(altitude.longitude.values, lon_tp, atol=1e-4)
)
print(f"mesma grade do tp: {'sim' if mesma_grade else 'NÃO, alinhar antes de juntar'}")
if mesma_grade:
    # copia as coordenadas exatas do tp, para o .sel funcionar sem erro de arredondamento
    altitude = altitude.assign_coords(latitude=lat_tp, longitude=lon_tp)
    lsm = lsm.assign_coords(latitude=lat_tp, longitude=lon_tp)

# Onde o tp tem dado (o ERA5-Land só tem valor sobre terra)
tem_tp = tp["tp"].isel({d: 0 for d in tp["tp"].dims if d not in ("latitude", "longitude")}).notnull()

# Sanidade
cwb = altitude.sel(latitude=-25.43, longitude=-49.27, method="nearest")
print(f"\naltitude em Curitiba: {float(cwb):.0f} m (esperado perto de 900 m, suavizado na célula)")
print(f"altitude: min {float(altitude.min()):.0f} m, max {float(altitude.max()):.0f} m")
for limiar in (0.0, 0.5, 0.9):
    print(f"células com lsm > {limiar}: {int((lsm > limiar).sum())}")
if mesma_grade:
    print(f"células com tp válido:  {int(tem_tp.sum())}")
    print(f"tp válido mas lsm <= 0.5: {int((tem_tp & (lsm <= 0.5)).sum())}  (litoral e lagos)")

SAIDA.parent.mkdir(parents=True, exist_ok=True)
xr.Dataset({"altitude": altitude, "lsm": lsm}).to_netcdf(SAIDA)
print(f"\nsalvo em {SAIDA}")

# Mapa para conferir
fig, eixos = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
altitude.where(lsm > 0.5).plot(ax=eixos[0], cmap="terrain", vmin=0, vmax=2000,
                               cbar_kwargs={"label": "m"})
eixos[0].set_title("Altitude (só terra, lsm > 0,5)")
lsm.plot(ax=eixos[1], cmap="Blues_r", vmin=0, vmax=1, cbar_kwargs={"label": "fração de terra"})
eixos[1].set_title("Máscara terra-mar")
for eixo in eixos:
    eixo.plot(-49.27, -25.43, "r^", ms=7)
    eixo.set_aspect("equal")
fig.savefig(FIGURA, dpi=120)
print(f"mapa salvo em {FIGURA}")