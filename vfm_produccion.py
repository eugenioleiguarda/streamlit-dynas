from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODELO_PREDETERMINADO = Path(__file__).with_name(
    "modelos_finales.joblib.gz"
)


GRUPOS_DE_VARIABLES = {
    "cb": [
        "CargaMaximaBomba",
        "CargaMinimaBomba",
        "CarreraMaximaBomba",
        "CarreraMinimaBomba",
        "CarreraEfectivaBombaInicio",
        "CarreraEfectivaBombaFin",
        "ValvulaFija",
        "ValvulaMovil",
        "ProfundidadBomba",
        "DesplazamientoEfectivo",
        "DesplazamientoPetroleoEfectivo",
        "DesplazamientoTotal",
        "DesplazamientoPetroleoTotal",
        "EficienciaVolumetrica",
        "Escurrimiento",
        "LlenadoBomba",
        "SobreRecorrido",
        "SubRecorrido",
        "PesoFluidoPromedio",
        "RendimientoBomba",
    ],
    "cba": [
        "PorcentajeCargaEstructural",
        "GravedadEspecifica",
        "Foc",
        "Nivel",
        "PresionDeFondo",
        "Sumergencia",
    ],
    "cm": [
        "EficienciaElevacion",
        "EficienciaGlobal",
        "PerdidaPotencia",
        "PotenciaBomba",
        "PotenciaEnRegimen",
        "PotenciaHidraulica",
        "PotenciaMotorEnBalance",
        "PotenciaMotorExistente",
        "PotenciaMotorMaximaEnBalance",
        "PotenciaMotorMaximaExistente",
        "PotenciaVastago",
    ],
    "cr": [
        "EfectoContrapesoEnBalance",
        "EfectoContrapesoExistente",
        "EficienciaTorsionalEnBalance",
        "EficienciaTorsionalExistente",
        "FactorCargaCiclicaEnBalance",
        "FactorCargaCiclicaExistente",
        "PorcentajeBalanceo",
        "PorcentajeTorqueReductorEnBalance",
        "PorcentajeTorqueReductorExistente",
        "TorqueMaximoContrapesoEnBalance",
        "TorqueMaximoContrapesoExistente",
        "TorqueNetoMaximoEnBalance",
        "TorqueNetoMaximoExistente",
        "TorquePozoMaximoExistente",
        "TorqueReductorEnBalance",
        "TorqueReductorExistente",
    ],
}


def _leer_items(contenido_json: bytes | str | dict | list) -> list[dict]:
    if isinstance(contenido_json, bytes):
        respuesta = json.load(
            io.TextIOWrapper(
                io.BytesIO(contenido_json),
                encoding="utf-8-sig",
            )
        )
    elif isinstance(contenido_json, str):
        respuesta = json.loads(contenido_json)
    else:
        respuesta = contenido_json

    items = (
        respuesta["items"]
        if isinstance(respuesta, dict) and "items" in respuesta
        else respuesta
    )
    if not isinstance(items, list) or not items:
        raise ValueError(
            "El JSON no contiene una colección 'items' válida."
        )
    return items


@lru_cache(maxsize=2)
def _cargar_bundle(ruta_modelo: str):
    ruta = Path(ruta_modelo)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo VFM: {ruta}"
        )
    bundle = joblib.load(ruta)
    if not {"modelos", "features"} <= set(bundle):
        raise ValueError(
            "El bundle VFM no contiene 'modelos' y 'features'."
        )
    return bundle


def construir_features_vfm(
    contenido_json: bytes | str | dict | list,
    features: list[str],
) -> pd.DataFrame:
    """Construye una fila de features por pozo y día."""
    datos = pd.json_normalize(_leer_items(contenido_json))
    faltantes = {"Pozo", "Fecha"} - set(datos.columns)
    if faltantes:
        raise ValueError(
            f"Faltan campos obligatorios: {sorted(faltantes)}"
        )

    datos["Pozo"] = datos["Pozo"].astype("string")
    datos["Fecha"] = pd.to_datetime(datos["Fecha"], errors="coerce")
    datos["Fecha_Dia"] = datos["Fecha"].dt.normalize()

    for prefijo, columnas in GRUPOS_DE_VARIABLES.items():
        for columna in columnas:
            nombre_modelo = f"{prefijo}_{columna}"
            datos[nombre_modelo] = (
                pd.to_numeric(datos[columna], errors="coerce")
                if columna in datos.columns
                else np.nan
            )

    columnas_base = [
        nombre.removesuffix("_median")
        for nombre in features
    ]
    faltantes_modelo = [
        nombre
        for nombre in columnas_base
        if nombre not in datos.columns
    ]
    if faltantes_modelo:
        raise ValueError(
            "No fue posible construir estas variables: "
            f"{faltantes_modelo}"
        )

    grupos = datos.groupby(["Pozo", "Fecha_Dia"], dropna=False)
    medianas = (
        grupos[columnas_base]
        .median()
        .add_suffix("_median")
        .reset_index()
    )
    cantidades = (
        grupos.size()
        .rename("VFM_Num_Cartas_Dia")
        .reset_index()
    )
    salida = cantidades.merge(
        medianas,
        on=["Pozo", "Fecha_Dia"],
        how="left",
        validate="one_to_one",
    )

    faltantes_finales = [
        nombre for nombre in features
        if nombre not in salida.columns
    ]
    if faltantes_finales:
        raise ValueError(
            f"Faltan features para inferencia: {faltantes_finales}"
        )

    filas_incompletas = salida[features].isna().any(axis=1)
    if filas_incompletas.any():
        pozos = salida.loc[
            filas_incompletas, "Pozo"
        ].astype(str).tolist()
        raise ValueError(
            "Hay variables VFM faltantes para estos pozos: "
            f"{pozos[:20]}"
        )

    return salida


def predecir_vfm(
    contenido_json: bytes | str | dict | list,
    ruta_modelo: str | Path = MODELO_PREDETERMINADO,
) -> pd.DataFrame:
    """Predice producción diaria usando las cartas del JSON."""
    bundle = _cargar_bundle(str(Path(ruta_modelo).resolve()))
    modelos = bundle["modelos"]
    features = list(bundle["features"])
    salida = construir_features_vfm(contenido_json, features)

    for objetivo, modelo in modelos.items():
        salida[f"_pred_{objetivo}"] = modelo.predict(
            salida[features]
        )

    if "_pred_bruta" in salida:
        salida["VFM_Bruta_m3_d"] = np.maximum(
            salida["_pred_bruta"], 0.0
        )
    if "_pred_petroleo" in salida:
        salida["VFM_Petroleo_m3_d"] = np.maximum(
            salida["_pred_petroleo"], 0.0
        )
    if "_pred_porcentaje_agua" in salida:
        salida["VFM_Agua_pct"] = np.clip(
            salida["_pred_porcentaje_agua"], 0.0, 100.0
        )
    if "_pred_residuo_bruta" in salida:
        salida["VFM_Bruta_Via_Residuo_m3_d"] = np.maximum(
            salida["cb_DesplazamientoEfectivo_median"]
            + salida["_pred_residuo_bruta"],
            0.0,
        )
    if {"VFM_Bruta_m3_d", "VFM_Agua_pct"} <= set(salida):
        salida["VFM_Petroleo_Via_Agua_m3_d"] = np.maximum(
            salida["VFM_Bruta_m3_d"]
            * (1.0 - salida["VFM_Agua_pct"] / 100.0),
            0.0,
        )

    columnas = [
        "Pozo",
        "Fecha_Dia",
        "VFM_Num_Cartas_Dia",
        "VFM_Bruta_m3_d",
        "VFM_Petroleo_m3_d",
        "VFM_Agua_pct",
        "VFM_Bruta_Via_Residuo_m3_d",
        "VFM_Petroleo_Via_Agua_m3_d",
    ]
    return salida[[c for c in columnas if c in salida]].copy()
