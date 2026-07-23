from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVO_CONTROLES_PREDETERMINADO = Path(__file__).with_name(
    "controles_reales.xlsx"
)


def normalizar_pozo(valor) -> str:
    texto = "" if pd.isna(valor) else str(valor)
    texto = re.sub(r"\s+", "", texto).upper()
    return re.sub(r"^YPF\.SC\.", "", texto)


def _fecha_operativa(serie: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(serie, errors="coerce", dayfirst=True)
    numeros = pd.to_numeric(serie, errors="coerce")
    # Solo tratar como serial de Excel los valores plausibles que no
    # hayan sido interpretados ya como fecha. Los Timestamp convertidos
    # a entero representan nanosegundos y no deben entrar aquí.
    mascara = (
        fechas.isna()
        & numeros.between(20_000, 80_000, inclusive="both")
    )
    if mascara.any():
        fechas.loc[mascara] = (
            pd.Timestamp("1899-12-30")
            + pd.to_timedelta(numeros.loc[mascara], unit="D")
        )
    return fechas.dt.normalize()


def leer_controles(contenido: bytes | None = None) -> pd.DataFrame:
    """Lee el reporte de controles y devuelve campos normalizados."""
    origen = (
        io.BytesIO(contenido)
        if contenido is not None
        else ARCHIVO_CONTROLES_PREDETERMINADO
    )
    if contenido is None and not ARCHIVO_CONTROLES_PREDETERMINADO.exists():
        return pd.DataFrame()

    crudo = pd.read_excel(origen, sheet_name=0, header=6)
    requeridas = {
        "Pozo",
        "Día Operativo",
        "Estado",
        "Producción Diaria de Líquido [m³]",
        "Producción Diaria de Petróleo [m³]",
        "Producción Diaria de Agua [m³]",
        "Relación Agua Líquido",
    }
    faltantes = requeridas.difference(crudo.columns)
    if faltantes:
        raise ValueError(
            "El Excel de controles no contiene estas columnas: "
            + ", ".join(sorted(faltantes))
        )

    salida = pd.DataFrame({
        "Pozo_Control": crudo["Pozo"].astype("string"),
        "Pozo_Clave": crudo["Pozo"].map(normalizar_pozo),
        "Fecha_Control": _fecha_operativa(crudo["Día Operativo"]),
        "Estado_Control": crudo["Estado"].astype("string"),
        "Control_Bruta_m3_d": pd.to_numeric(
            crudo["Producción Diaria de Líquido [m³]"],
            errors="coerce",
        ),
        "Control_Petroleo_m3_d": pd.to_numeric(
            crudo["Producción Diaria de Petróleo [m³]"],
            errors="coerce",
        ),
        "Control_Agua_m3_d": pd.to_numeric(
            crudo["Producción Diaria de Agua [m³]"],
            errors="coerce",
        ),
        "Control_Agua_pct": 100 * pd.to_numeric(
            crudo["Relación Agua Líquido"],
            errors="coerce",
        ),
    })
    salida = salida.loc[
        salida["Pozo_Clave"].ne("")
        & salida["Fecha_Control"].notna()
    ].copy()
    return salida.sort_values(
        ["Pozo_Clave", "Fecha_Control"]
    ).reset_index(drop=True)


def _comentario(fila: pd.Series) -> str:
    if pd.isna(fila.get("Fecha_Control")):
        return "Sin control real disponible para este pozo."

    partes = []
    estado = str(fila.get("Estado_Control", "")).strip().lower()
    antiguedad = fila.get("Control_Antiguedad_dias", np.nan)
    if estado == "a revisar":
        partes.append("El control real figura como A Revisar.")
    if np.isfinite(antiguedad) and antiguedad > 60:
        partes.append(f"El control tiene {int(antiguedad)} días de antigüedad.")

    error = fila.get("Error_Bruta_pct", np.nan)
    if not np.isfinite(error):
        partes.append("No se pudo calcular el error relativo de caudal bruto.")
    elif abs(error) <= 15:
        partes.append("Muy buen acuerdo en caudal bruto (diferencia ≤ 15%).")
    elif abs(error) <= 30:
        partes.append("Diferencia moderada en caudal bruto (15%–30%).")
    elif error > 0:
        partes.append("El VFM sobreestima el caudal bruto en más de 30%.")
    else:
        partes.append("El VFM subestima el caudal bruto en más de 30%.")

    delta_agua = fila.get("Delta_Agua_pp", np.nan)
    if np.isfinite(delta_agua) and abs(delta_agua) > 10:
        sentido = "mayor" if delta_agua > 0 else "menor"
        partes.append(
            f"El corte de agua VFM es {abs(delta_agua):.1f} pp {sentido}."
        )
    return " ".join(partes)


def cruzar_controles(
    produccion_vfm: pd.DataFrame,
    controles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Asigna a cada VFM el control más reciente del mismo pozo cuya fecha
    no sea posterior a la fecha de la predicción.
    """
    izquierda = produccion_vfm.copy()
    izquierda["Fecha_Dia"] = pd.to_datetime(
        izquierda["Fecha_Dia"], errors="coerce"
    ).dt.normalize()
    izquierda["Pozo_Clave"] = izquierda["Pozo"].map(normalizar_pozo)

    if controles.empty:
        return izquierda

    partes = []
    por_pozo = {
        clave: grupo.sort_values("Fecha_Control")
        for clave, grupo in controles.groupby("Pozo_Clave", sort=False)
    }
    for _, fila in izquierda.iterrows():
        candidatos = por_pozo.get(fila["Pozo_Clave"])
        seleccion = None
        if candidatos is not None and pd.notna(fila["Fecha_Dia"]):
            previos = candidatos.loc[
                candidatos["Fecha_Control"] <= fila["Fecha_Dia"]
            ]
            if not previos.empty:
                seleccion = previos.iloc[-1]
        combinado = fila.to_dict()
        if seleccion is not None:
            combinado.update(seleccion.to_dict())
        partes.append(combinado)

    salida = pd.DataFrame(partes)
    salida["Control_Antiguedad_dias"] = (
        salida["Fecha_Dia"] - pd.to_datetime(
            salida.get("Fecha_Control"), errors="coerce"
        )
    ).dt.days
    salida["Delta_Bruta_m3_d"] = (
        salida["VFM_Bruta_m3_d"] - salida["Control_Bruta_m3_d"]
    )
    salida["Error_Bruta_pct"] = np.where(
        salida["Control_Bruta_m3_d"] > 0,
        100 * salida["Delta_Bruta_m3_d"] / salida["Control_Bruta_m3_d"],
        np.nan,
    )
    salida["Delta_Petroleo_m3_d"] = (
        salida["VFM_Petroleo_m3_d"] - salida["Control_Petroleo_m3_d"]
    )
    salida["Error_Petroleo_pct"] = np.where(
        salida["Control_Petroleo_m3_d"] > 0,
        100
        * salida["Delta_Petroleo_m3_d"]
        / salida["Control_Petroleo_m3_d"],
        np.nan,
    )
    salida["Delta_Agua_pp"] = (
        salida["VFM_Agua_pct"] - salida["Control_Agua_pct"]
    )
    salida["Comentario_VFM_Control"] = salida.apply(_comentario, axis=1)
    return salida


def resumen_comparacion(comparacion: pd.DataFrame) -> dict:
    coincidentes = comparacion.loc[
        comparacion["Control_Bruta_m3_d"].notna()
    ].copy()
    cantidad_total = len(comparacion)
    cantidad = len(coincidentes)
    if cantidad == 0:
        return {
            "cantidad": 0,
            "total": cantidad_total,
            "cobertura_pct": 0.0,
            "caudales": pd.DataFrame(),
            "porcentajes": pd.DataFrame(),
        }

    real_bruta = coincidentes["Control_Bruta_m3_d"].sum()
    real_petroleo = coincidentes["Control_Petroleo_m3_d"].sum()
    real_agua = coincidentes["Control_Agua_m3_d"].sum()
    vfm_bruta = coincidentes["VFM_Bruta_m3_d"].sum()
    vfm_petroleo = coincidentes["VFM_Petroleo_m3_d"].sum()
    vfm_agua = (
        coincidentes["VFM_Bruta_m3_d"]
        * coincidentes["VFM_Agua_pct"]
        / 100
    ).sum()

    caudales = pd.DataFrame(
        {
            "Control real": [real_bruta, real_petroleo],
            "VFM": [vfm_bruta, vfm_petroleo],
        },
        index=["Bruta [m³/d]", "Petróleo [m³/d]"],
    )
    porcentajes = pd.DataFrame(
        {
            "Control real": [
                100 * real_agua / real_bruta if real_bruta > 0 else np.nan
            ],
            "VFM": [
                100 * vfm_agua / vfm_bruta if vfm_bruta > 0 else np.nan
            ],
        },
        index=["Corte de agua [%]"],
    )
    return {
        "cantidad": cantidad,
        "total": cantidad_total,
        "cobertura_pct": 100 * cantidad / cantidad_total
        if cantidad_total
        else 0.0,
        "caudales": caudales,
        "porcentajes": porcentajes,
    }
