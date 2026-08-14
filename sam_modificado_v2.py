"""Detector geométrico experimental de transferencias y rodillas.

Este módulo funciona en modo sombra: no participa de diagnósticos, llenado,
carta patrones ni del SAM Modificado productivo.
"""

from __future__ import annotations

import numpy as np


def _rama(rama):
    x = np.asarray(rama.get("posicion", []), dtype=float)
    y = np.asarray(rama.get("carga", []), dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _suavizar(valores):
    valores = np.asarray(valores, dtype=float)
    if len(valores) < 5:
        return valores.copy()
    relleno = np.pad(valores, (2, 2), mode="edge")
    return np.asarray([
        np.median(relleno[i:i + 5]) for i in range(len(valores))
    ])


def _secuencia_lateral(xa, ya, xd, yd, lado, xmin, xmax, rx):
    limite = 0.30 * rx
    if lado == "izquierda":
        ia = np.flatnonzero(xa <= xmin + limite)
        id_ = np.flatnonzero(xd <= xmin + limite)
        # Fin de descendente -> inicio de ascendente: continuidad circular.
        return (
            np.r_[xd[id_], xa[ia]],
            np.r_[yd[id_], ya[ia]],
        )
    ia = np.flatnonzero(xa >= xmax - limite)
    id_ = np.flatnonzero(xd >= xmax - limite)
    # Fin de ascendente -> inicio de descendente.
    return (
        np.r_[xa[ia], xd[id_]],
        np.r_[ya[ia], yd[id_]],
    )


def _mejor_transferencia(x, y, rx, ry):
    """Devuelve los extremos del núcleo recto vertical/oblicuo."""
    if len(x) < 7:
        return None
    xs, ys = _suavizar(x), _suavizar(y)
    dx = np.diff(xs) / rx
    dy = np.diff(ys) / ry
    modulo = np.hypot(dx, dy)
    verticalidad = np.divide(
        np.abs(dy), modulo, out=np.zeros_like(dy), where=modulo > 1e-6
    )
    activos = (verticalidad >= 0.62) & (np.abs(dy) >= 0.012)
    # Cierra huecos aislados producidos por ruido dentro de una recta.
    for i in range(1, len(activos) - 1):
        if not activos[i] and activos[i - 1] and activos[i + 1]:
            activos[i] = True
    corridas, inicio = [], None
    for i, activo in enumerate(np.r_[activos, False]):
        if activo and inicio is None:
            inicio = i
        elif not activo and inicio is not None:
            corridas.append((inicio, i - 1))
            inicio = None
    mejor = None
    for i0, i1 in corridas:
        p0, p1 = i0, i1 + 1
        span = abs(float(ys[p1] - ys[p0])) / ry
        if i1 - i0 + 1 < 2 or span < 0.18:
            continue
        mediana = float(np.median(verticalidad[i0:i1 + 1]))
        xx, yy = xs[p0:p1 + 1] / rx, ys[p0:p1 + 1] / ry
        if np.ptp(yy) <= 1e-6:
            continue
        coef = np.polyfit(yy, xx, 1)
        residuo = float(np.mean((xx - np.polyval(coef, yy)) ** 2))
        linealidad = float(np.exp(-residuo / 0.0025))
        score = span * mediana * linealidad
        if mejor is None or score > mejor[0]:
            mejor = (score, p0, p1, span, mediana, linealidad)
    if mejor is None:
        return None
    _, p0, p1, span, mediana, linealidad = mejor
    puntos = [(float(x[p0]), float(y[p0])), (float(x[p1]), float(y[p1]))]
    inferior, superior = sorted(puntos, key=lambda p: p[1])
    confianza = float(np.clip(
        0.35 * min(span / 0.50, 1.0) + 0.35 * mediana + 0.30 * linealidad,
        0.0, 1.0,
    ))
    return inferior, superior, confianza


def calcular_segmentacion_geometrica_v2(
    ascendente,
    descendente,
    profundidad_bomba_m=np.nan,
    diametro_piston_pulg=np.nan,
    presion_tubing_kg_cm2=10.0,
    presion_casing_kg_cm2=10.0,
    gravedad_especifica=0.994,
    gradiente_psi_m=None,
):
    salida = {
        "Calculo_SAM_V2_Valido": False,
        "Motivo_SAM_V2_No_Valido": "",
        "Metodo_SAM_V2": "SEGMENTACION_TRANSFERENCIAS_RODILLAS",
        "Confianza_SAM_V2": np.nan,
    }
    for color in ("Roja", "Azul"):
        for lado in ("Izquierda", "Derecha"):
            salida[f"Posicion_{color}_{lado}_SAM_V2_pulg"] = np.nan
            salida[f"Carga_{color}_{lado}_SAM_V2_lbf"] = np.nan
    salida.update({
        "Carga_Superior_SAM_V2_lbf": np.nan,
        "Carga_Inferior_SAM_V2_lbf": np.nan,
        "Peso_Fluido_SAM_V2_lbf": np.nan,
        "Sumergencia_SAM_V2_m": np.nan,
        "Sumergencia_Relativa_SAM_V2_pct": np.nan,
    })
    try:
        xa, ya = _rama(ascendente)
        xd, yd = _rama(descendente)
        if min(len(xa), len(xd)) < 5:
            raise ValueError("RAMAS_INSUFICIENTES")
        xmin, xmax = float(min(xa.min(), xd.min())), float(max(xa.max(), xd.max()))
        ymin, ymax = float(min(ya.min(), yd.min())), float(max(ya.max(), yd.max()))
        rx, ry = xmax - xmin, ymax - ymin
        if rx <= 0 or ry <= 0:
            raise ValueError("CARRERA_NULA")
        detectados = {}
        for lado in ("izquierda", "derecha"):
            x, y = _secuencia_lateral(xa, ya, xd, yd, lado, xmin, xmax, rx)
            detectados[lado] = _mejor_transferencia(x, y, rx, ry)
            if detectados[lado] is None:
                raise ValueError(f"TRANSFERENCIA_{lado.upper()}_NO_DETECTADA")
        azul_i, rojo_i, conf_i = detectados["izquierda"]
        azul_d, rojo_d, conf_d = detectados["derecha"]
        for nombre, punto in (
            ("Azul_Izquierda", azul_i), ("Roja_Izquierda", rojo_i),
            ("Azul_Derecha", azul_d), ("Roja_Derecha", rojo_d),
        ):
            salida[f"Posicion_{nombre}_SAM_V2_pulg"] = punto[0]
            salida[f"Carga_{nombre}_SAM_V2_lbf"] = punto[1]
        superior = 0.5 * (rojo_i[1] + rojo_d[1])
        inferior = 0.5 * (azul_i[1] + azul_d[1])
        peso = superior - inferior
        if peso <= 0:
            raise ValueError("PESO_NO_POSITIVO")
        salida.update({
            "Calculo_SAM_V2_Valido": True,
            "Carga_Superior_SAM_V2_lbf": float(superior),
            "Carga_Inferior_SAM_V2_lbf": float(inferior),
            "Peso_Fluido_SAM_V2_lbf": float(peso),
            "Confianza_SAM_V2": float(min(conf_i, conf_d)),
        })
        profundidad = float(profundidad_bomba_m)
        diametro = float(diametro_piston_pulg)
        gradiente = (
            float(gradiente_psi_m)
            if gradiente_psi_m is not None
            else 0.433 * 3.280839895013123 * float(gravedad_especifica)
        )
        if profundidad > 0 and diametro > 0 and gradiente > 0:
            area = float(np.pi * diametro ** 2 / 4.0)
            presion_descarga = (
                float(presion_tubing_kg_cm2) * 14.223343307
                + gradiente * profundidad
            )
            pip = presion_descarga - peso / area
            sumergencia = (
                pip - float(presion_casing_kg_cm2) * 14.223343307
            ) / gradiente
            salida["Sumergencia_SAM_V2_m"] = float(sumergencia)
            salida["Sumergencia_Relativa_SAM_V2_pct"] = float(
                100.0 * sumergencia / profundidad
            )
    except (TypeError, ValueError) as error:
        salida["Motivo_SAM_V2_No_Valido"] = str(error)
    return salida
