"""Geometria das âncoras e das vizinhas na grade do ERA5-Land, e a escolha sem colisão.

Âncora = célula de 0,1° da sede do município. Vizinhas = 8 células num anel, todas à
mesma distância. Há dois anéis (--anel na linha de comando, cada um com a sua pasta):
- 30km (padrão): n/s/e/w a 3 células, diagonais a 2 + 2 células, em dados/;
- 150km: n/s/e/w a 13 células, diagonais a 9 + 9 células (123 a 145 km), em dados_150/.
Nos dois, as âncoras ficam a pelo menos 7 células uma da outra (Chebyshev) e duas âncoras
não podem dividir nenhuma célula. No anel de 30 km a segunda regra já vem da primeira; no de
150 km os anéis passam por dentro da área de outras âncoras, só não caem nas mesmas células.

Índices: row_era5 = round((90 − lat) × 10) e col_era5 = round((lon mod 360) × 10). A janela do disco começa no canto noroeste (6,0, −75,0) = (ROW0, COL0).
"""
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

# (linha, coluna) em células; a linha cresce para o sul
POSICOES = {
    "center": (0, 0),
    "n": (-3, 0),
    "ne": (-2, 2),
    "e": (0, 3),
    "se": (2, 2),
    "s": (3, 0),
    "sw": (2, -2),
    "w": (0, -3),
    "nw": (-2, -2),
}
POSICOES_150KM = {
    "center": (0, 0),
    "n": (-13, 0),
    "ne": (-9, 9),
    "e": (0, 13),
    "se": (9, 9),
    "s": (13, 0),
    "sw": (9, -9),
    "w": (0, -13),
    "nw": (-9, -9),
}
PASTA = Path("av1/amostra_brasil")
ANEIS = {   # nome -> posições, pasta dos dados, pasta das figuras, faixa de dist_km das vizinhas
    "30km": {"posicoes": POSICOES, "dados": PASTA / "dados", "figuras": PASTA / "figuras",
             "dist_km": (27, 34)},
    "150km": {"posicoes": POSICOES_150KM, "dados": PASTA / "dados_150",
              "figuras": PASTA / "figuras_150", "dist_km": (120, 148)},
}
ESPACAMENTO = 7
ROW0, COL0 = 840, 2850
REGIOES = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}


def anel_da_linha_de_comando(argv=None):
    """--anel 30km (padrão) ou --anel 150km: o nome e a configuração em ANEIS."""
    argv = sys.argv[1:] if argv is None else argv
    nome = argv[argv.index("--anel") + 1] if "--anel" in argv else "30km"
    if nome not in ANEIS:
        raise ValueError(f"anel desconhecido: {nome} (use {', '.join(ANEIS)})")
    return nome, ANEIS[nome]


def celula_era5(lat, lon):
    """row_era5, col_era5 da célula que contém cada ponto."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (np.rint((90 - lat) * 10).astype(int),
            np.rint(np.mod(lon, 360) * 10).astype(int) % 3600)


def centro_da_celula(row_era5, col_era5):
    """Latitude e longitude (de −180 a 180) do centro, com 2 casas."""
    lat = np.round(90 - np.asarray(row_era5) / 10, 2)
    lon = np.asarray(col_era5) / 10
    return lat, np.round(np.where(lon > 180, lon - 360, lon), 2)


def distancia_km(lat1, lon1, lat2, lon2):
    """Distância de haversine, raio de 6371 km."""
    la1, lo1, la2, lo2 = (np.radians(np.asarray(x, float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def tem_dado(row_era5, col_era5, mascara, posicoes=POSICOES):
    """True se as 9 posições de cada âncora caem dentro da janela e com dado na máscara."""
    iy0, ix0 = np.asarray(row_era5) - ROW0, np.asarray(col_era5) - COL0
    ok = np.ones(len(iy0), bool)
    for dr, dc in posicoes.values():
        iy, ix = iy0 + dr, ix0 + dc
        dentro = (iy >= 0) & (iy < mascara.shape[0]) & (ix >= 0) & (ix < mascara.shape[1])
        ok &= dentro
        ok[dentro] &= mascara[iy[dentro], ix[dentro]]
    return ok


def empacotar(row_era5, col_era5, codigo, posicoes=POSICOES):
    """Maior conjunto sem colisão, pelo guloso de menor grau.

    Duas candidatas conflitam se a distância de Chebyshev for menor que ESPACAMENTO ou se
    dividirem alguma célula (a diferença entre elas é a diferença de duas posições). A cada
    passo entra a candidata viva com menos conflitos vivos (empate: menor código), e saem ela
    e as que conflitam com ela.
    """
    r, c, codigo = np.asarray(row_era5), np.asarray(col_era5), np.asarray(codigo)
    dr, dc = r[:, None] - r[None], c[:, None] - c[None]
    conflito = np.maximum(abs(dr), abs(dc)) < ESPACAMENTO
    pos = np.array(list(posicoes.values()))
    for ddr, ddc in {tuple(p - q) for p in pos for q in pos}:
        conflito |= (dr == ddr) & (dc == ddc)
    np.fill_diagonal(conflito, False)
    viva = np.ones(len(r), bool)
    escolhida = np.zeros(len(r), bool)
    grau = conflito.sum(1)
    while viva.any():
        vivas = np.flatnonzero(viva)
        i = vivas[np.lexsort((codigo[vivas], grau[vivas]))[0]]
        escolhida[i] = True
        sai = viva & (conflito[i] | (np.arange(len(r)) == i))
        viva &= ~sai
        grau -= conflito[:, sai].sum(1)
    return escolhida


def baixar_json(url):
    """JSON de uma API do IBGE, que manda gzip mesmo sem pedir."""
    with urllib.request.urlopen(url, timeout=120) as resposta:
        bruto = resposta.read()
    return json.loads(gzip.decompress(bruto) if bruto[:2] == b"\x1f\x8b" else bruto)
