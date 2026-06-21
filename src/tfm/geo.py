import pandas as pd


def normalize_mun_code(mun_code):
    """Colapsa códigos de distrito BCN/MAD al código de ciudad (8019 / 28079)."""
    if pd.isna(mun_code):
        return pd.NA
    if isinstance(mun_code, float):
        if mun_code != mun_code:
            return pd.NA
        code_str = str(int(mun_code))
    elif isinstance(mun_code, int):
        code_str = str(mun_code)
    else:
        code_str = str(mun_code).strip().upper()

    if code_str.startswith("BCN"):
        return 8019
    if code_str.startswith("MAD"):
        return 28079

    digits = "".join(c for c in code_str if c.isdigit())
    if not digits:
        return pd.NA
    if digits.startswith("08019") or digits.startswith("8019"):
        return 8019
    if digits.startswith("28079"):
        return 28079
    return int(digits)


def get_prov_code(mun_code):
    """Extrae el código de provincia desde mun_code, incluyendo distritos BCN/MAD."""
    # Normalizamos primero (admite int/float/str, incluidos códigos "BCN"/"MAD", y NaN),
    # de modo que int() nunca falle ante un código no numérico.
    code = normalize_mun_code(mun_code)
    if pd.isna(code):
        return pd.NA
    s = str(int(code))
    if s.startswith("8019"):
        return 8
    if s.startswith("28079"):
        return 28
    return int(s.zfill(5)[:2])


# Códigos de distrito según municipios.csv (orden alfabético, no INE oficial)
BCN_DISTRICT_MUN_CODES = {
    "Ciutat Vella":        80192,
    "Eixample":            80193,
    "Gràcia":              80194,
    "Horta-Guinardó":      80195,
    "Les Corts":           80196,
    "Nou Barris":          80197,
    "Sant Andreu":         80198,
    "Sant Martí":          80199,
    "Sants-Montjuïc":      801910,
    "Sarrià-Sant Gervasi": 801911,
}

MAD_DISTRICT_MUN_CODES = {
    "Arganzuela":          280792,
    "Barajas":             280793,
    "Carabanchel":         280794,
    "Centro":              280795,
    "Chamartín":           280796,
    "Chamberí":            280797,
    "Ciudad Lineal":       280798,
    "Fuencarral-El Pardo": 280799,
    "Hortaleza":           2807910,
    "Latina":              2807911,
    "Moncloa-Aravaca":     2807912,
    "Moratalaz":           2807913,
    "Puente de Vallecas":  2807914,
    "Retiro":              2807915,
    "Salamanca":           2807916,
    "San Blas-Canillejas": 2807917,
    "Tetuán":              2807918,
    "Usera":               2807919,
    "Vicálvaro":           2807920,
    "Villa de Vallecas":   2807921,
    "Villaverde":          2807922,
}

MADRID_CP_TO_INE = {
    28004: 280795, 28005: 280795, 28012: 280795,
    28013: 280795, 28014: 280795, 28015: 280795,   # Centro
    28026: 280792, 28045: 280792,                   # Arganzuela
    28007: 2807915, 28009: 2807915,                 # Retiro
    28001: 2807916, 28006: 2807916, 28028: 2807916, # Salamanca
    28002: 280796,  28016: 280796,  28036: 280796,  # Chamartín
    28020: 2807918, 28029: 2807918, 28039: 2807918, # Tetuán
    28003: 280797,  28010: 280797,                  # Chamberí
    28034: 280799,  28035: 280799,  28049: 280799,  28050: 280799,  # Fuencarral
    28008: 2807912, 28023: 2807912, 28040: 2807912, # Moncloa-Aravaca
    28011: 2807911, 28024: 2807911, 28044: 2807911, 28047: 2807911, # Latina
    28019: 280794,  28025: 280794,                  # Carabanchel
    28041: 2807919,                                 # Usera
    28018: 2807914, 28038: 2807914, 28053: 2807914, # Puente de Vallecas
    28030: 2807913,                                 # Moratalaz
    28017: 280798,  28027: 280798,  28037: 280798,  # Ciudad Lineal
    28033: 2807910, 28043: 2807910, 28055: 2807910, # Hortaleza
    28021: 2807922,                                 # Villaverde
    28031: 2807921, 28051: 2807921,                 # Villa de Vallecas
    28032: 2807920,                                 # Vicálvaro
    28022: 2807917,                                 # San Blas-Canillejas
    28042: 280793,                                  # Barajas
}

BCN_CP_TO_INE = {
    8001: 80192, 8002: 80192, 8003: 80192,          # Ciutat Vella
    8007: 80193, 8008: 80193, 8009: 80193,
    8010: 80193, 8011: 80193, 8013: 80193,
    8015: 80193, 8025: 80193, 8029: 80193,           # Eixample
    8004: 801910, 8014: 801910, 8028: 801910, 8038: 801910,  # Sants-Montjuïc
    8017: 80196, 8034: 80196,                        # Les Corts
    8006: 801911, 8021: 801911, 8022: 801911, 8023: 801911,  # Sarrià-St. Gervasi
    8012: 80194, 8024: 80194,                        # Gràcia
    8031: 80195, 8032: 80195, 8035: 80195,           # Horta-Guinardó
    8016: 80197, 8042: 80197,                        # Nou Barris
    8027: 80198, 8030: 80198,                        # Sant Andreu
    8005: 80199, 8018: 80199, 8019: 80199, 8020: 80199,  # Sant Martí
}

CP_TO_CUSTOM_INE = {**MADRID_CP_TO_INE, **BCN_CP_TO_INE}
