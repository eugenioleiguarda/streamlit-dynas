"""Pipeline consolidado de diagnóstico de cartas dinamométricas.

Generado a partir del notebook calibrado por Diego. La aplicación Streamlit
lo ejecuta una vez por archivo y reutiliza sus resultados para los filtros.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PIPELINE_IMPLEMENTATION_VERSION = (
    "2026-09-01-v93-codos-laterales-y-admision-suave-9f2c"
)



# Hipotesis explicitas para el calculo paralelo de sumergencia.
# Por ahora no intervienen en ninguna regla diagnostica.
API_PETROLEO_SUMERGENCIA_PROPIA = 25.0
FRACCION_AGUA_SUMERGENCIA_PROPIA = 0.90
FRACCION_PETROLEO_SUMERGENCIA_PROPIA = 0.10
METROS_A_PIES = 3.280839895013123
PSI_POR_KG_CM2 = 14.223343307
GRAVEDAD_ESPECIFICA_REFERENCIA = 0.9904
KG_CM2_A_PSI = 14.223343307
PSI_PIE_AGUA = 0.433
PIES_POR_METRO = 3.280839895013123


def calcular_sam_modificado(
    ascendente,
    descendente,
    profundidad_bomba_m,
    diametro_piston_pulg,
    presion_tubing_kg_cm2=10.0,
    presion_casing_kg_cm2=10.0,
    gravedad_especifica=0.994,
    gradiente_psi_m=None,
):
    """Detecta cuatro codos laterales y calcula el SAM Modificado.

    Los dos codos superiores pertenecen a la rama ascendente y los dos
    inferiores a la descendente. Cada codo se busca desde el extremo de la
    rama hacia el interior, donde el trazo deja de ser aproximadamente
    vertical y pasa a ser aproximadamente horizontal. Las horizontales son
    los promedios de las cargas de cada par. El metodo es unico para todos
    los diagnosticos y no usa etiquetas diagnosticas ni valores de la API.
    """
    salida = {
        "Calculo_SAM_Modificado_Valido": False,
        "Motivo_SAM_Modificado_No_Valido": "",
        "Metodo_SAM_Seleccionado": "SAM_MODIFICADO_EXTREMOS_TRANSFERENCIA",
        "Regla_Inferior_SAM_Modificado": "PROMEDIO_DOS_CODOS_AZULES",
        "Azul_Izquierdo_Incluido_SAM_Modificado": True,
        "Carga_Roja_Izquierda_SAM_Modificado_lbf": np.nan,
        "Carga_Roja_Derecha_SAM_Modificado_lbf": np.nan,
        "Carga_Azul_Izquierda_SAM_Modificado_lbf": np.nan,
        "Carga_Azul_Derecha_SAM_Modificado_lbf": np.nan,
        "Posicion_Roja_Izquierda_SAM_Modificado_pulg": np.nan,
        "Posicion_Roja_Derecha_SAM_Modificado_pulg": np.nan,
        "Posicion_Azul_Izquierda_SAM_Modificado_pulg": np.nan,
        "Posicion_Azul_Derecha_SAM_Modificado_pulg": np.nan,
        "Carga_Superior_SAM_Seleccionada_lbf": np.nan,
        "Carga_Inferior_SAM_Seleccionada_lbf": np.nan,
        "Peso_Fluido_SAM_Seleccionado_lbf": np.nan,
        "Area_Piston_SAM_pulg2": np.nan,
        "Diferencial_Carga_SAM_psi": np.nan,
        "Presion_Tubing_SAM_kg_cm2": np.nan,
        "Presion_Casing_SAM_kg_cm2": np.nan,
        "Gravedad_Especifica_SAM": np.nan,
        "Gradiente_SAM_psi_m": np.nan,
        "Presion_Descarga_Bomba_SAM_psi": np.nan,
        "PIP_SAM_Seleccionado_psi": np.nan,
        "Sumergencia_SAM_Seleccionada_m": np.nan,
        "Sumergencia_Relativa_SAM_Seleccionada_pct": np.nan,
        "Nivel_Dinamico_SAM_Modificado_m": np.nan,
    }
    try:
        x_asc = np.asarray(ascendente["posicion"], dtype=float)
        y_asc = np.asarray(ascendente["carga"], dtype=float)
        x_desc = np.asarray(descendente["posicion"], dtype=float)
        y_desc = np.asarray(descendente["carga"], dtype=float)
        va = np.isfinite(x_asc) & np.isfinite(y_asc)
        vd = np.isfinite(x_desc) & np.isfinite(y_desc)
        x_asc, y_asc = x_asc[va], y_asc[va]
        x_desc, y_desc = x_desc[vd], y_desc[vd]
        if min(len(x_asc), len(x_desc)) < 5:
            raise ValueError("RAMAS_INSUFICIENTES")

        rango_x = float(max(np.max(x_asc), np.max(x_desc)) - min(np.min(x_asc), np.min(x_desc)))
        rango_y = float(max(np.max(y_asc), np.max(y_desc)) - min(np.min(y_asc), np.min(y_desc)))
        if rango_x <= 0 or rango_y <= 0:
            raise ValueError("CARRERA_NULA")

        def detectar_lateral(x, y, lado):
            # Busca la transferencia aproximadamente recta del lateral. La
            # carrera sirve para seguir la secuencia original, pero el codo
            # se define por el fin de la recta y no por una meseta de carga.
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            x_min_global = float(min(np.min(x_asc), np.min(x_desc)))
            x_max_global = float(max(np.max(x_asc), np.max(x_desc)))
            if lado == "izquierda":
                mascara = x <= x_min_global + 0.52 * rango_x
            else:
                # Incluye transferencias derechas desplazadas hacia el
                # interior (EG-232), pero la selección sigue exigiendo una
                # recta consecutiva con transferencia apreciable de carga.
                mascara = x >= x_max_global - 0.72 * rango_x
            puntos = np.column_stack([
                (x - x_min_global) / rango_x,
                (y - np.min(y)) / rango_y,
            ])
            if np.count_nonzero(mascara) < 5:
                raise ValueError("LATERAL_INSUFICIENTE")

            # Se enumeran rectas consecutivas contenidas en el lateral. La
            # ganadora maximiza el recorrido vertical y la colinealidad; sus
            # extremos son exactamente las inflexiones pedidas.
            mejor = None
            for inicio in range(len(puntos) - 2):
                if not mascara[inicio]:
                    continue
                for fin in range(inicio + 2, len(puntos)):
                    if not np.all(mascara[inicio:fin + 1]):
                        break
                    tramo = puntos[inicio:fin + 1]
                    amplitud = float(np.ptp(tramo[:, 1]))
                    if amplitud < 0.10:
                        continue
                    delta = tramo[-1] - tramo[0]
                    norma = float(np.linalg.norm(delta))
                    if norma <= 1e-9:
                        continue
                    verticalidad = abs(float(delta[1])) / norma
                    if verticalidad < 0.48:
                        continue
                    centrado = tramo - np.mean(tramo, axis=0)
                    _, singulares, _ = np.linalg.svd(
                        centrado, full_matrices=False
                    )
                    if len(singulares) < 2 or singulares[0] <= 1e-9:
                        continue
                    no_linealidad = float(singulares[1] / singulares[0])
                    # Laterales oblicuos de perdida en viajera o friccion
                    # pueden ser suavemente curvos.
                    if no_linealidad > 0.16:
                        continue
                    score = amplitud * (0.65 + verticalidad) * (
                        1.0 + 0.025 * len(tramo)
                    ) * (
                        1.0 - min(0.75, 2.5 * no_linealidad)
                    ) / (1.0 + 0.08 * inicio)
                    if mejor is None or score > mejor[0]:
                        mejor = (score, inicio, fin)
            if mejor is None:
                # Respaldo acotado para cartas cortas: exige tres muestras
                # consecutivas, transferencia apreciable y predominio de
                # carga. Solo entra cuando el ajuste auditable principal no
                # encontró ninguna recta, por lo que no altera cartas ya
                # resueltas.
                for inicio in range(len(puntos) - 2):
                    for fin in range(inicio + 2, min(len(puntos), inicio + 6)):
                        if not np.all(mascara[inicio:fin + 1]):
                            continue
                        tramo = puntos[inicio:fin + 1]
                        amplitud = float(np.ptp(tramo[:, 1]))
                        delta = tramo[-1] - tramo[0]
                        norma = float(np.linalg.norm(delta))
                        if amplitud < 0.075 or norma <= 1e-9:
                            continue
                        verticalidad = abs(float(delta[1])) / norma
                        if verticalidad < 0.42:
                            continue
                        score = amplitud * (0.6 + verticalidad)
                        if mejor is None or score > mejor[0]:
                            mejor = (score, inicio, fin)
                if mejor is None:
                    raise ValueError("RECTA_TRANSFERENCIA_NO_ENCONTRADA")
            inicio_elegido, fin_elegido = mejor[1], mejor[2]

            # Recorta los extremos redondeados que todavia pueden caber en un
            # ajuste global. Se conserva la corrida consecutiva mas larga de
            # segmentos con direccion estable respecto del eje principal de
            # la transferencia.
            if fin_elegido - inicio_elegido >= 4:
                tramo = puntos[inicio_elegido:fin_elegido + 1]
                kernel = np.array([0.25, 0.50, 0.25])
                suavizado = np.column_stack([
                    np.convolve(
                        np.pad(tramo[:, eje], 1, mode="edge"),
                        kernel,
                        mode="valid",
                    )
                    for eje in range(2)
                ])
                centrado = suavizado - np.mean(suavizado, axis=0)
                _, _, vh = np.linalg.svd(centrado, full_matrices=False)
                eje_recta = vh[0]
                vectores = np.diff(suavizado, axis=0)
                normas = np.linalg.norm(vectores, axis=1)
                alineados = np.zeros(len(vectores), dtype=bool)
                validos = normas > 1e-9
                cosenos = np.zeros(len(vectores), dtype=float)
                cosenos[validos] = np.abs(
                    vectores[validos] @ eje_recta
                ) / normas[validos]
                alineados[validos] = cosenos[validos] >= np.cos(
                    np.deg2rad(24.0)
                )
                mejor_corrida = None
                k = 0
                while k < len(alineados):
                    if not alineados[k]:
                        k += 1
                        continue
                    fin_corrida = k
                    while (
                        fin_corrida + 1 < len(alineados)
                        and alineados[fin_corrida + 1]
                    ):
                        fin_corrida += 1
                    longitud = float(np.sum(normas[k:fin_corrida + 1]))
                    if mejor_corrida is None or longitud > mejor_corrida[0]:
                        mejor_corrida = (longitud, k, fin_corrida)
                    k = fin_corrida + 1
                if mejor_corrida is not None and mejor_corrida[2] - mejor_corrida[1] >= 1:
                    inicio_elegido += mejor_corrida[1]
                    fin_elegido = (
                        mejor[1] + mejor_corrida[2] + 1
                    )

            indices = np.arange(inicio_elegido, fin_elegido + 1)
            # Los cuatro puntos deben quedar dentro de la transferencia, no
            # en el vértice ni sobre las pseudo horizontales. Se recortan una
            # o dos muestras de cada extremo de la recta lateral; esto vuelve
            # estable el criterio ante ondulaciones y pequeños rulos.
            orden_carga = indices[np.argsort(y[indices])]
            recorte_azul = (
                2 if len(orden_carga) >= 7
                else 1 if len(orden_carga) >= 5
                else 0
            )
            indice_azul = orden_carga[recorte_azul]
            if lado == "izquierda":
                recorte_rojo = (
                    3 if len(orden_carga) >= 9
                    else 2 if len(orden_carga) >= 7
                    else 1 if len(orden_carga) >= 5
                    else 0
                )
            else:
                extremo_bajo = puntos[orden_carga[0]]
                extremo_alto = puntos[orden_carga[-1]]
                vector_transferencia = extremo_alto - extremo_bajo
                norma_transferencia = float(np.linalg.norm(
                    vector_transferencia
                ))
                apertura_transferencia = (
                    abs(float(vector_transferencia[0]))
                    / norma_transferencia
                    if norma_transferencia > 1e-9 else 0.0
                )
                recorte_rojo = (
                    2 if len(orden_carga) >= 7
                    else 1 if len(orden_carga) >= 5
                    else 0
                )
            indice_rojo = orden_carga[-1 - recorte_rojo]

            # En la derecha la rama descendente continúa, después de la
            # transferencia, por la envolvente inferior. El codo azul es el
            # primer quiebre sostenido de una dirección vertical/oblicua a
            # una dirección predominantemente horizontal; no necesariamente
            # coincide con la carga mínima del lateral.
            if lado == "derecha":
                inicio_busqueda = int(indice_rojo)
                fin_busqueda = inicio_busqueda
                while fin_busqueda + 1 < len(x) and mascara[fin_busqueda + 1]:
                    fin_busqueda += 1
                if fin_busqueda - inicio_busqueda >= 4:
                    tramo_busqueda = puntos[inicio_busqueda:fin_busqueda + 1]
                    vec = np.diff(tramo_busqueda, axis=0)
                    norma_vec = np.linalg.norm(vec, axis=1)
                    componente_horizontal = np.zeros(len(vec), dtype=float)
                    ok_vec = norma_vec > 1e-9
                    componente_horizontal[ok_vec] = (
                        np.abs(vec[ok_vec, 0]) / norma_vec[ok_vec]
                    )
                    # La transferencia puede ser vertical u oblicua. Por eso
                    # no se usa un umbral absoluto de horizontalidad: se
                    # estima su dirección basal con los primeros segmentos y
                    # se busca el primer aumento sostenido de apertura lateral.
                    # Ese punto todavía pertenece a la transferencia; si el
                    # giro es abrupto, se conserva la muestra inmediatamente
                    # anterior al salto.
                    minimo_apertura = np.inf
                    indice_minimo_apertura = None
                    candidatos_codo = []
                    for k in range(1, len(componente_horizontal) - 1):
                        apertura_actual = float(componente_horizontal[k])
                        apertura_siguiente = float(componente_horizontal[k + 1])
                        carga_en_k = float(y[inicio_busqueda + k])
                        mitad_baja_carga = (
                            carga_en_k
                            <= float(min(np.min(y_asc), np.min(y_desc)))
                            + 0.45 * rango_y
                        )
                        if not mitad_baja_carga:
                            continue
                        if apertura_actual < minimo_apertura:
                            minimo_apertura = apertura_actual
                            indice_minimo_apertura = k
                            continue
                        aumento = apertura_actual - minimo_apertura
                        if (
                            indice_minimo_apertura is not None
                            and aumento >= 0.12
                            and apertura_siguiente >= minimo_apertura + 0.10
                        ):
                            if apertura_actual >= 0.45:
                                indice_codo = indice_minimo_apertura
                            else:
                                indice_codo = k
                            candidatos_codo.append(indice_codo)
                            break

                    # Estimador complementario para transferencias largas y
                    # oblicuas: localiza el primer paso sostenido desde un
                    # núcleo poco lateral hacia una apertura marcada. Ambos
                    # estimadores describen el mismo fenómeno; se conserva el
                    # candidato de mayor carga, que es el primero sobre la
                    # transferencia antes de que se forme la rama inferior.
                    vio_nucleo = False
                    for k in range(1, len(componente_horizontal) - 1):
                        carga_en_k = float(y[inicio_busqueda + k])
                        mitad_baja_carga = (
                            carga_en_k
                            <= float(min(np.min(y_asc), np.min(y_desc)))
                            + 0.55 * rango_y
                        )
                        previo = float(np.median(
                            componente_horizontal[max(0, k - 2):k + 1]
                        ))
                        posterior = float(np.median(
                            componente_horizontal[k:min(
                                len(componente_horizontal), k + 2
                            )]
                        ))
                        if mitad_baja_carga and previo <= 0.40:
                            vio_nucleo = True
                        if (
                            vio_nucleo
                            and posterior >= 0.45
                            and componente_horizontal[k] >= 0.38
                        ):
                            candidatos_codo.append(max(0, k - 1))
                            break

                    if candidatos_codo:
                        indice_relativo = max(
                            candidatos_codo,
                            key=lambda candidato: float(
                                y[inicio_busqueda + candidato]
                            ),
                        )
                        indice_azul = inicio_busqueda + indice_relativo
            return (
                float(x[indice_azul]), float(y[indice_azul]),
                float(x[indice_rojo]), float(y[indice_rojo]),
            )

        def detectar_hombro_superior_derecho(x, y):
            """Quiebre meseta--caída en una envolvente superior arqueada."""
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            x_min_global = float(min(np.min(x_asc), np.min(x_desc)))
            mascara = x >= x_min_global + 0.42 * rango_x
            indices = np.flatnonzero(mascara)
            if len(indices) < 7:
                return None
            # La ascendente conserva el orden de adquisición de izquierda a
            # derecha. En ese orden se ajustan una meseta y una caída, con un
            # punto compartido que representa el hombro físico.
            xx = (x[indices] - x_min_global) / rango_x
            yy = (y[indices] - min(np.min(y_asc), np.min(y_desc))) / rango_y
            mejor = None
            for corte in range(3, len(indices) - 2):
                x_pre, y_pre = xx[:corte + 1], yy[:corte + 1]
                x_post, y_post = xx[corte:], yy[corte:]
                if np.ptp(x_pre) <= 1e-6 or np.ptp(x_post) <= 1e-6:
                    continue
                coef_pre = np.polyfit(x_pre, y_pre, 1)
                coef_post = np.polyfit(x_post, y_post, 1)
                pendiente_pre = float(coef_pre[0])
                pendiente_post = float(coef_post[0])
                # Tipo arqueado: antes casi horizontal o suavemente variable;
                # después una pérdida sostenida de carga hacia la derecha.
                if pendiente_post >= -0.35:
                    continue
                if pendiente_post >= pendiente_pre - 0.25:
                    continue
                residuo = float(np.mean(
                    (y_pre - np.polyval(coef_pre, x_pre)) ** 2
                ) + np.mean(
                    (y_post - np.polyval(coef_post, x_post)) ** 2
                ))
                caida = float(y_pre[-1] - y_post[-1])
                if caida < 0.10:
                    continue
                score = residuo + 0.015 * abs(pendiente_pre)
                if mejor is None or score < mejor[0]:
                    mejor = (score, corte)
            if mejor is None:
                return None
            indice = int(indices[mejor[1]])
            return float(x[indice]), float(y[indice])

        def detectar_recta_superior_derecha(x, y):
            """Primer punto ya contenido en la transferencia descendente."""
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            x_min = float(min(np.min(x_asc), np.min(x_desc)))
            indices = np.flatnonzero(x >= x_min + 0.45 * rango_x)
            if len(indices) < 7:
                return None
            xx = x[indices] / rango_x
            yy = y[indices] / rango_y
            dx = np.diff(xx)
            dy = np.diff(yy)
            pendientes = np.full(len(dx), np.nan, dtype=float)
            validas = dx > 1e-4
            pendientes[validas] = dy[validas] / dx[validas]
            # La meseta debe preceder a por lo menos tres segmentos de caída.
            for k in range(2, len(pendientes) - 2):
                previas = pendientes[max(0, k - 3):k]
                futuras = pendientes[k:k + 3]
                previas = previas[np.isfinite(previas)]
                futuras = futuras[np.isfinite(futuras)]
                if len(previas) < 2 or len(futuras) < 2:
                    continue
                meseta_previa = float(np.median(np.abs(previas))) <= 0.32
                caida_sostenida = (
                    float(np.median(futuras)) <= -0.38
                    and np.count_nonzero(futuras <= -0.22) >= 2
                )
                if meseta_previa and caida_sostenida:
                    # Una muestra después del cambio: ya está sobre la recta
                    # de transferencia y no en el codo de la meseta.
                    indice = int(indices[min(k + 1, len(indices) - 1)])
                    return float(x[indice]), float(y[indice])
            return None

        try:
            candidato_izquierdo_base = detectar_lateral(
                x_asc, y_asc, "izquierda"
            )
        except ValueError:
            # Respaldo exclusivo para una transferencia izquierda corta pero
            # geométricamente inequívoca (pocas muestras en cartas CG-16).
            # Al ejecutarse solo tras el rechazo principal no cambia ningún
            # caso previamente resuelto.
            x_respaldo = np.concatenate([x_asc, x_desc])
            y_respaldo = np.concatenate([y_asc, y_desc])
            x_min_respaldo = float(min(np.min(x_asc), np.min(x_desc)))
            mascara_respaldo = (
                x_respaldo <= x_min_respaldo + 0.25 * rango_x
            )
            indices_respaldo = np.flatnonzero(mascara_respaldo)
            y_min_respaldo = float(min(np.min(y_asc), np.min(y_desc)))
            if (
                len(indices_respaldo) < 4
                or float(np.ptp(y_respaldo[indices_respaldo])) < 0.40 * rango_y
            ):
                raise
            objetivo_azul = y_min_respaldo + 0.30 * rango_y
            objetivo_rojo = y_min_respaldo + 0.82 * rango_y
            indice_azul = int(indices_respaldo[np.argmin(
                abs(y_respaldo[indices_respaldo] - objetivo_azul)
            )])
            indice_rojo = int(indices_respaldo[np.argmin(
                abs(y_respaldo[indices_respaldo] - objetivo_rojo)
            )])
            candidato_izquierdo_base = (
                float(x_respaldo[indice_azul]), float(y_respaldo[indice_azul]),
                float(x_respaldo[indice_rojo]), float(y_respaldo[indice_rojo]),
            )
        candidatos_izquierda = [
            ("ascendente", candidato_izquierdo_base)
        ]
        (
            x_azul_izq, azul_izquierda,
            x_roja_izq, roja_izquierda,
        ) = candidato_izquierdo_base
        candidatos_derecha = []
        for x_rama, y_rama, nombre_rama in (
            (x_desc, y_desc, "descendente"),
            (x_asc, y_asc, "ascendente"),
        ):
            try:
                candidato = detectar_lateral(x_rama, y_rama, "derecha")
                amplitud_candidato = abs(candidato[3] - candidato[1])
                candidatos_derecha.append((
                    amplitud_candidato, nombre_rama, candidato
                ))
            except ValueError:
                pass
        if not candidatos_derecha:
            raise ValueError("TRANSFERENCIA_DERECHA_NO_ENCONTRADA")
        # La transferencia física domina por carga transferida. Un pequeño
        # desempate favorece la descendente, que contiene el lateral derecho
        # convencional; la ascendente gana cuando realmente porta la
        # transferencia completa (pérdida en viajera).
        candidatos_derecha.sort(
            key=lambda item: (
                item[0], item[1] == "descendente"
            ),
            reverse=True,
        )
        (
            _, rama_derecha_elegida,
            (
                x_azul_der, azul_derecha,
                x_roja_der, roja_derecha,
            ),
        ) = candidatos_derecha[0]
        candidato_derecho_base = candidatos_derecha[0][2]

        # Si ambas ramas describen transferencias de amplitud similar, se
        # conserva la exterior: evita que ondulaciones de la meseta compitan
        # con los laterales reales en cartas EG-10.
        try:
            exterior = max(
                candidatos_derecha,
                key=lambda item: max(item[2][0], item[2][2]),
            )
            amplitud_ganadora = candidatos_derecha[0][0]
            if exterior[0] >= 0.88 * amplitud_ganadora:
                (
                    x_azul_der, azul_derecha,
                    x_roja_der, roja_derecha,
                ) = exterior[2]
                rama_derecha_elegida = exterior[1]
                candidato_derecho_base = exterior[2]
        except (ValueError, IndexError):
            pass

        def transferencia_oblicua(candidato):
            dx_normalizado = abs(candidato[2] - candidato[0]) / rango_x
            dy_normalizado = abs(candidato[3] - candidato[1]) / rango_y
            norma = float(np.hypot(dx_normalizado, dy_normalizado))
            return bool(
                norma > 1e-9 and dx_normalizado / norma >= 0.20
            )

        candidatos_ascendentes_derecha = [
            item for item in candidatos_derecha
            if item[1] == "ascendente"
        ]
        candidato_derecho_viajera = (
            max(candidatos_ascendentes_derecha, key=lambda item: item[0])[2]
            if candidatos_ascendentes_derecha
            else None
        )
        morfologia_valvula_viajera = bool(
            candidato_derecho_viajera is not None
            and transferencia_oblicua(candidato_izquierdo_base)
            and transferencia_oblicua(candidato_derecho_viajera)
        )

        # Reconciliación independiente de las cuatro esquinas. La selección
        # base v59 se conserva salvo que otra pareja de extremos mejore de
        # manera material la alineación entre lados. Los azules desempatan
        # hacia menor carga (extremos inferiores); los rojos hacia mayor carga
        # y, débilmente, hacia el corredor exterior derecho.
        candidatos_azules_izq = [
            (candidato[0], candidato[1])
            for _, candidato in candidatos_izquierda
        ]
        candidatos_rojos_izq = [
            (candidato[2], candidato[3])
            for _, candidato in candidatos_izquierda
        ]
        # Candidatos locales baratos alrededor del mínimo geométrico. Permiten
        # que un codo izquierdo pertenezca al tramo contiguo de la otra carrera
        # sin ejecutar una cuarta búsqueda exhaustiva de rectas.
        x_min_global = float(min(np.min(x_asc), np.min(x_desc)))
        for x_rama, y_rama, indices_locales in (
            (x_asc, y_asc, range(0, min(8, len(x_asc)))),
            (
                x_desc, y_desc,
                range(max(0, len(x_desc) - 8), len(x_desc)),
            ),
        ):
            for indice_local in indices_locales:
                if x_rama[indice_local] <= x_min_global + 0.14 * rango_x:
                    punto_local = (
                        float(x_rama[indice_local]),
                        float(y_rama[indice_local]),
                    )
                    candidatos_azules_izq.append(punto_local)
                    candidatos_rojos_izq.append(punto_local)
        candidatos_azules_der = [
            (item[2][0], item[2][1]) for item in candidatos_derecha
        ]
        candidatos_rojos_der = [
            (item[2][2], item[2][3]) for item in candidatos_derecha
        ]

        y_min_global = float(min(np.min(y_asc), np.min(y_desc)))
        y_max_global = float(max(np.max(y_asc), np.max(y_desc)))
        x_max_global = float(max(np.max(x_asc), np.max(x_desc)))

        def score_azul(izquierdo, derecho):
            media = 0.5 * (izquierdo[1] + derecho[1])
            return (
                abs(izquierdo[1] - derecho[1]) / rango_y
                + 0.045 * (media - y_min_global) / rango_y
            )

        def score_rojo(izquierdo, derecho):
            media = 0.5 * (izquierdo[1] + derecho[1])
            exterioridad = max(0.0, (x_max_global - derecho[0]) / rango_x)
            return (
                abs(izquierdo[1] - derecho[1]) / rango_y
                + 0.035 * (y_max_global - media) / rango_y
                + 0.08 * exterioridad
            )

        actual_azul = (
            (x_azul_izq, azul_izquierda),
            (x_azul_der, azul_derecha),
        )
        original_azul = actual_azul
        mejor_azul = min(
            (
                (izquierdo, derecho)
                for izquierdo in candidatos_azules_izq
                for derecho in candidatos_azules_der
            ),
            key=lambda pareja: score_azul(*pareja),
        )
        mejora_azul = (
            score_azul(*actual_azul) - score_azul(*mejor_azul)
        )
        media_actual_azul = 0.5 * (
            azul_izquierda + azul_derecha
        )
        ambos_azules_altos = (
            media_actual_azul > y_min_global + 0.55 * rango_y
        )
        propuesta_azul = (
            mejor_azul
            if mejora_azul >= 0.025 or ambos_azules_altos
            else actual_azul
        )

        actual_rojo = (
            (x_roja_izq, roja_izquierda),
            (x_roja_der, roja_derecha),
        )
        original_rojo = actual_rojo
        mejor_rojo = min(
            (
                (izquierdo, derecho)
                for izquierdo in candidatos_rojos_izq
                for derecho in candidatos_rojos_der
            ),
            key=lambda pareja: score_rojo(*pareja),
        )
        propuesta_rojo = (
            mejor_rojo
            if score_rojo(*actual_rojo) - score_rojo(*mejor_rojo) >= 0.035
            else actual_rojo
        )

        def extremos_lateral_recto(x, y, lado):
            """Codos interiores de una transferencia lateral casi vertical."""
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            limite = (
                x <= x_min_global + 0.20 * rango_x
                if lado == "izquierda"
                else x >= x_max_global - 0.20 * rango_x
            )
            indices = np.flatnonzero(limite)
            if len(indices) < 5:
                return None
            # Exige que el lateral cubra buena parte de la transferencia y
            # que su núcleo no sea una rampa oblicua.
            carga_lateral = y[indices]
            if float(np.ptp(carga_lateral)) < 0.45 * rango_y:
                return None
            centro = indices[
                (carga_lateral >= y_min_global + 0.18 * rango_y)
                & (carga_lateral <= y_min_global + 0.86 * rango_y)
            ]
            if len(centro) < 3:
                return None
            objetivo_azul = y_min_global + 0.30 * rango_y
            objetivo_rojo = y_min_global + 0.82 * rango_y
            indice_azul = int(indices[np.argmin(abs(y[indices] - objetivo_azul))])
            indice_rojo = int(indices[np.argmin(abs(y[indices] - objetivo_rojo))])
            return (
                (float(x[indice_azul]), float(y[indice_azul])),
                (float(x[indice_rojo]), float(y[indice_rojo])),
            )

        # Patrón rectangular redondeado: en lugar de tomar los extremos de
        # carga, recorta ambos laterales en el punto donde la recta empieza a
        # curvarse hacia las envolventes. Se evalúa como una propuesta única
        # de cuatro codos para no forzar un punto aislado a la otra carrera.
        x_ciclo = np.concatenate([x_asc, x_desc])
        y_ciclo = np.concatenate([y_asc, y_desc])
        rectangular_izq = extremos_lateral_recto(
            x_ciclo, y_ciclo, "izquierda"
        )
        rectangular_der = extremos_lateral_recto(
            x_ciclo, y_ciclo, "derecha"
        )
        traslacion_cuatro_codos = False
        if rectangular_izq is not None and rectangular_der is not None:
            azul_rect_izq, rojo_rect_izq = rectangular_izq
            azul_rect_der, rojo_rect_der = rectangular_der
            desplazamientos_crudos = np.asarray([
                actual_rojo[0][1] - rojo_rect_izq[1],
                actual_rojo[1][1] - rojo_rect_der[1],
                actual_azul[0][1] - azul_rect_izq[1],
                actual_azul[1][1] - azul_rect_der[1],
            ], dtype=float)
            # Si los cuatro codos están trasladados hacia arriba, conserva
            # exactamente la separación de cargas y desplaza los cuatro por
            # un mismo delta robusto. Los puntos finales siempre se ajustan
            # a muestras reales del lateral correspondiente.
            if np.all(desplazamientos_crudos > 0.08 * rango_y):
                traslacion_cuatro_codos = True
                delta_comun = float(np.median(desplazamientos_crudos))

                def punto_lateral_cercano(lado, carga_objetivo):
                    mascara_lado = (
                        x_ciclo <= x_min_global + 0.20 * rango_x
                        if lado == "izquierda"
                        else x_ciclo >= x_max_global - 0.20 * rango_x
                    )
                    indices_lado = np.flatnonzero(mascara_lado)
                    indice = int(indices_lado[np.argmin(
                        abs(y_ciclo[indices_lado] - carga_objetivo)
                    )])
                    return float(x_ciclo[indice]), float(y_ciclo[indice])

                rojo_rect_izq = punto_lateral_cercano(
                    "izquierda", actual_rojo[0][1] - delta_comun
                )
                rojo_rect_der = punto_lateral_cercano(
                    "derecha", actual_rojo[1][1] - delta_comun
                )
                azul_rect_izq = punto_lateral_cercano(
                    "izquierda", actual_azul[0][1] - delta_comun
                )
                azul_rect_der = punto_lateral_cercano(
                    "derecha", actual_azul[1][1] - delta_comun
                )
            # Conserva cada esquina que ya estaba razonablemente cerca del
            # codo interior; evita mover puntos correctos solo para forzar
            # simetría entre lados.
            if abs(azul_rect_izq[1] - actual_azul[0][1]) < 0.10 * rango_y:
                azul_rect_izq = actual_azul[0]
            if abs(azul_rect_der[1] - actual_azul[1][1]) < 0.10 * rango_y:
                azul_rect_der = actual_azul[1]
            if abs(rojo_rect_izq[1] - actual_rojo[0][1]) < 0.10 * rango_y:
                rojo_rect_izq = actual_rojo[0]
            if rojo_rect_izq[1] > actual_rojo[0][1]:
                rojo_rect_izq = actual_rojo[0]
            if abs(rojo_rect_der[1] - actual_rojo[1][1]) < 0.10 * rango_y:
                rojo_rect_der = actual_rojo[1]
            if rojo_rect_der[1] > actual_rojo[1][1]:
                rojo_rect_der = actual_rojo[1]
            peso_rectangular = (
                0.5 * (rojo_rect_izq[1] + rojo_rect_der[1])
                - 0.5 * (azul_rect_izq[1] + azul_rect_der[1])
            )
            peso_candidato_actual = (
                0.5 * (actual_rojo[0][1] + actual_rojo[1][1])
                - 0.5 * (actual_azul[0][1] + actual_azul[1][1])
            )
            peso_previo_propuesto = (
                0.5 * (propuesta_rojo[0][1] + propuesta_rojo[1][1])
                - 0.5 * (propuesta_azul[0][1] + propuesta_azul[1][1])
            )
            peso_referencia = (
                min(peso_candidato_actual, peso_previo_propuesto)
                if peso_previo_propuesto > 0
                else peso_candidato_actual
            )
            desplazamiento_material = max(
                abs(rojo_rect_der[1] - actual_rojo[1][1]),
                abs(azul_rect_der[1] - actual_azul[1][1]),
                abs(azul_rect_izq[1] - actual_azul[0][1]),
            ) >= 0.08 * rango_y
            azules_cruzados = (
                (
                    actual_azul[0][1] > azul_rect_izq[1] + 0.10 * rango_y
                    and actual_azul[1][1] < azul_rect_der[1] - 0.10 * rango_y
                )
                or (
                    actual_azul[0][1] < azul_rect_izq[1] - 0.10 * rango_y
                    and actual_azul[1][1] > azul_rect_der[1] + 0.10 * rango_y
                )
            )
            patron_extremos_cruzados = (
                azules_cruzados
                and actual_rojo[1][1] > rojo_rect_der[1] + 0.08 * rango_y
                and peso_rectangular < peso_candidato_actual
            )
            traslacion_carga_conservada = (
                traslacion_cuatro_codos
                and abs(peso_rectangular - peso_candidato_actual)
                <= 0.03 * rango_y
            )
            if (
                desplazamiento_material
                and 0 < peso_rectangular
                and (
                    peso_rectangular <= peso_referencia - 0.01 * rango_y
                    or patron_extremos_cruzados
                    or traslacion_carga_conservada
                )
            ):
                propuesta_azul = (azul_rect_izq, azul_rect_der)
                propuesta_rojo = (rojo_rect_izq, rojo_rect_der)

        # Los ajustes independientes se reservan para cartas cuya solución
        # conservadora previa todavía arroja sumergencia negativa. Así no se
        # reabren cartas positivas que ya habían quedado validadas.
        peso_original_previo = (
            0.5 * (original_rojo[0][1] + original_rojo[1][1])
            - 0.5 * (original_azul[0][1] + original_azul[1][1])
        )
        peso_propuesta_previa = (
            0.5 * (propuesta_rojo[0][1] + propuesta_rojo[1][1])
            - 0.5 * (propuesta_azul[0][1] + propuesta_azul[1][1])
        )
        peso_base_previo = (
            peso_propuesta_previa
            if 0 < peso_propuesta_previa <= peso_original_previo + 1e-9
            else peso_original_previo
        )
        habilitar_ajuste_negativos = False
        try:
            profundidad_previa = float(profundidad_bomba_m)
            diametro_previo = float(diametro_piston_pulg)
            area_previa = float(np.pi * diametro_previo ** 2 / 4.0)
            gradiente_previo = pd.to_numeric(gradiente_psi_m, errors="coerce")
            if not np.isfinite(gradiente_previo) or gradiente_previo <= 0:
                gradiente_previo = (
                    PSI_PIE_AGUA * PIES_POR_METRO * float(gravedad_especifica)
                )
            pd_previa = (
                float(presion_tubing_kg_cm2) * KG_CM2_A_PSI
                + gradiente_previo * profundidad_previa
            )
            pip_previa = pd_previa - peso_base_previo / area_previa
            sumergencia_previa = (
                pip_previa
                - float(presion_casing_kg_cm2) * KG_CM2_A_PSI
            ) / gradiente_previo
            habilitar_ajuste_negativos = sumergencia_previa < 0
        except (TypeError, ValueError, ZeroDivisionError):
            habilitar_ajuste_negativos = False

        # Correcciones independientes que solo reducen el peso calculado.
        # 1) El azul derecho no puede quedar en el extremo inferior cuando
        # existe un quiebre lateral claramente más alto.
        if (
            habilitar_ajuste_negativos
            and rectangular_der is not None
            and not traslacion_cuatro_codos
        ):
            mascara_quiebre_der = (
                x_ciclo >= x_max_global - 0.20 * rango_x
            )
            indices_quiebre_der = np.flatnonzero(mascara_quiebre_der)
            objetivo_quiebre_der = max(
                y_min_global + 0.30 * rango_y,
                -0.05 * rango_y,
            )
            indice_quiebre_der = int(indices_quiebre_der[np.argmin(
                abs(y_ciclo[indices_quiebre_der] - objetivo_quiebre_der)
            )])
            azul_quiebre_der = (
                float(x_ciclo[indice_quiebre_der]),
                float(y_ciclo[indice_quiebre_der]),
            )
            if azul_quiebre_der[1] > propuesta_azul[1][1] + 0.08 * rango_y:
                propuesta_azul = (
                    propuesta_azul[0], azul_quiebre_der
                )

        # 2) En cartas arqueadas, recorta cada rojo hasta el corredor interior
        # del lateral. Nunca eleva un rojo ni prolonga la pendiente hacia la
        # envolvente inferior.
        if (
            habilitar_ajuste_negativos
            and rectangular_izq is not None
            and not traslacion_cuatro_codos
        ):
            rojo_quiebre_izq = rectangular_izq[1]
            if rojo_quiebre_izq[1] < propuesta_rojo[0][1] - 0.08 * rango_y:
                propuesta_rojo = (rojo_quiebre_izq, propuesta_rojo[1])
        if (
            habilitar_ajuste_negativos
            and not traslacion_cuatro_codos
        ):
            # En cartas con pérdida en viajera, el codo superior derecho
            # puede quedar sobre la meseta. Priorizamos el primer punto que
            # ya pertenece a la caída sostenida de transferencia de carga.
            candidatos_rojo_der = []
            rojo_transferencia_der = detectar_recta_superior_derecha(
                x_asc, y_asc
            )
            if rojo_transferencia_der is not None:
                candidatos_rojo_der.append(rojo_transferencia_der)
            if rectangular_der is not None:
                candidatos_rojo_der.append(rectangular_der[1])
            rojo_quiebre_der = (
                min(candidatos_rojo_der, key=lambda punto: punto[1])
                if candidatos_rojo_der
                else None
            )
            if (
                rojo_quiebre_der is not None
                and rojo_quiebre_der[1]
                < propuesta_rojo[1][1] - 0.05 * rango_y
            ):
                propuesta_rojo = (propuesta_rojo[0], rojo_quiebre_der)

        # Salvaguarda hidráulica conservadora: una reconciliación visual no
        # puede aumentar materialmente el peso de fluido respecto del método
        # ya validado. Así los candidatos independientes corrigen sesgos
        # negativos, pero no convierten cartas antes razonables en negativas.
        peso_original = (
            0.5 * (original_rojo[0][1] + original_rojo[1][1])
            - 0.5 * (original_azul[0][1] + original_azul[1][1])
        )
        peso_propuesto = (
            0.5 * (propuesta_rojo[0][1] + propuesta_rojo[1][1])
            - 0.5 * (propuesta_azul[0][1] + propuesta_azul[1][1])
        )
        if 0 < peso_propuesto <= peso_original + 1e-9:
            (x_azul_izq, azul_izquierda), (
                x_azul_der, azul_derecha
            ) = propuesta_azul
            (x_roja_izq, roja_izquierda), (
                x_roja_der, roja_derecha
            ) = propuesta_rojo

        def frontera_geometrica_rama(x, y, lado):
            """Frontera robusta entre una lateral y una pseudo horizontal.

            Ajusta dos rectas contiguas en coordenadas normalizadas y sólo
            acepta el quiebre cuando una es claramente más horizontal que la
            otra. Esto admite transferencias verticales u oblicuas y evita
            usar una carga SAM como nivel de corte.
            """
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            orden = np.argsort(x, kind="stable")
            x, y = x[orden], y[orden]
            limite = (
                x <= x_min_global + 0.42 * rango_x
                if lado == "izquierda"
                else x >= x_max_global - 0.42 * rango_x
            )
            indices = np.flatnonzero(limite)
            if len(indices) < 6:
                return None
            xx = (x[indices] - x_min_global) / rango_x
            yy = (y[indices] - y_min_global) / rango_y

            def ajuste_tls(a, b):
                puntos = np.column_stack([a, b])
                centro = np.mean(puntos, axis=0)
                centrados = puntos - centro
                _, singulares, vh = np.linalg.svd(
                    centrados, full_matrices=False
                )
                if not len(singulares) or singulares[0] <= 1e-9:
                    return None
                eje = vh[0]
                horizontalidad = abs(float(eje[0]))
                residuo = float(np.mean(
                    (centrados @ np.array([-eje[1], eje[0]])) ** 2
                ))
                return horizontalidad, residuo

            mejor = None
            for corte in range(2, len(indices) - 2):
                primero = ajuste_tls(xx[:corte + 1], yy[:corte + 1])
                segundo = ajuste_tls(xx[corte:], yy[corte:])
                if primero is None or segundo is None:
                    continue
                if lado == "izquierda":
                    lateral, horizontal = primero, segundo
                    amplitud_lateral = float(np.ptp(yy[:corte + 1]))
                    amplitud_horizontal = float(np.ptp(xx[corte:]))
                else:
                    horizontal, lateral = primero, segundo
                    amplitud_horizontal = float(np.ptp(xx[:corte + 1]))
                    amplitud_lateral = float(np.ptp(yy[corte:]))
                diferencia = horizontal[0] - lateral[0]
                if (
                    horizontal[0] < 0.72
                    or lateral[0] > 0.82
                    or diferencia < 0.16
                    or amplitud_lateral < 0.12
                    or amplitud_horizontal < 0.12
                ):
                    continue
                score = (
                    primero[1] + segundo[1]
                    + 0.012 / max(diferencia, 1e-6)
                )
                if mejor is None or score < mejor[0]:
                    mejor = (score, corte, diferencia)
            if mejor is None:
                return None

            corte = mejor[1]
            # En ambos lados se devuelve la muestra perteneciente a la
            # transferencia: fin de la lateral izquierda o comienzo de la
            # lateral derecha. Así el punto no invade la pseudo horizontal.
            indice_local = max(0, corte - 1) if lado == "izquierda" else corte
            indice_ordenado = int(indices[indice_local])
            return (
                float(x[indice_ordenado]),
                float(y[indice_ordenado]),
                float(mejor[2]),
            )

        # En la morfología de pérdida en válvula viajera, la transferencia
        # completa está en la ascendente y es oblicua. Se restauran sus
        # extremos geométricos antes de cualquier reconciliación por nivel:
        # los rojos quedan arriba y los azules en el pie de las oblicuas.
        if morfologia_valvula_viajera:
            (
                x_azul_izq, azul_izquierda,
                x_roja_izq, roja_izquierda,
            ) = candidato_izquierdo_base
            (
                x_azul_der, azul_derecha,
                x_roja_der, roja_derecha,
            ) = candidato_derecho_viajera
            # El ajuste de la recta completa puede dejar el rojo derecho en
            # la última muestra alta de la meseta. Para esta morfología se
            # prefiere el primer punto ya contenido en la caída oblicua
            # sostenida, es decir, inmediatamente después del codo superior.
            rojo_derecho_oblicua = detectar_recta_superior_derecha(
                x_asc, y_asc
            )
            # Se calcula siempre un segundo candidato sin exigir una meseta
            # perfectamente horizontal. Así una detección temprana dentro
            # del codo no impide encontrar la oblicua ya consolidada.
            rojo_derecho_caida = None
            orden_asc = np.argsort(x_asc, kind="stable")
            x_sup = x_asc[orden_asc]
            y_sup = y_asc[orden_asc]
            dx_sup = np.diff(x_sup / rango_x)
            dy_sup = np.diff(y_sup / rango_y)
            pendientes_sup = np.full(len(dx_sup), np.nan)
            validas_sup = dx_sup > 1e-4
            pendientes_sup[validas_sup] = (
                dy_sup[validas_sup] / dx_sup[validas_sup]
            )
            inicio_sup = int(np.searchsorted(x_sup, x_roja_der))
            for k in range(inicio_sup, len(pendientes_sup) - 2):
                futuras = pendientes_sup[k:k + 3]
                futuras = futuras[np.isfinite(futuras)]
                if len(futuras) < 2:
                    continue
                if (
                    float(np.median(futuras)) <= -0.38
                    and np.count_nonzero(futuras <= -0.22) >= 2
                ):
                    # Una muestra adicional deja atrás el redondeo del codo
                    # y ubica el punto inequívocamente sobre la oblicua.
                    indice_caida = min(k + 4, len(x_sup) - 1)
                    rojo_derecho_caida = (
                        float(x_sup[indice_caida]),
                        float(y_sup[indice_caida]),
                    )
                    break
            candidatos_rojo_oblicua = [
                punto
                for punto in (
                    rojo_derecho_oblicua,
                    rojo_derecho_caida,
                )
                if punto is not None
            ]
            rojo_derecho_oblicua = (
                max(candidatos_rojo_oblicua, key=lambda punto: punto[0])
                if candidatos_rojo_oblicua
                else None
            )
            if (
                rojo_derecho_oblicua is not None
                and rojo_derecho_oblicua[0] >= x_roja_der
                and rojo_derecho_oblicua[1]
                < roja_derecha - 0.01 * rango_y
                and rojo_derecho_oblicua[1]
                > azul_derecha + 0.35 * rango_y
            ):
                x_roja_der, roja_derecha = rojo_derecho_oblicua

            # Respaldo para falsas morfologías de viajera en cartas con los
            # cuatro codos tomados demasiado arriba. Se exige que las dos
            # fronteras inferiores de la descendente estén materialmente por
            # debajo de los azules actuales; sólo entonces se reemplaza el
            # conjunto completo por las transiciones laterales geométricas.
            rojo_valvula_der = frontera_geometrica_rama(
                x_asc, y_asc, "derecha"
            )
            azul_valvula_izq = frontera_geometrica_rama(
                x_desc, y_desc, "izquierda"
            )
            azul_valvula_der = frontera_geometrica_rama(
                x_desc, y_desc, "derecha"
            )
            indices_rojo_valvula_izq = np.flatnonzero(
                (x_asc == x_roja_izq) & (y_asc == roja_izquierda)
            )
            rojo_valvula_izq = None
            if len(indices_rojo_valvula_izq):
                # La primera muestra es el pie de la ascendente y la segunda
                # ya está inequívocamente dentro de la vertical izquierda.
                # Se evita el punto siguiente, que en rodillas cortas puede
                # pertenecer ya a la pseudo-horizontal superior.
                indice_rojo_valvula_izq = min(1, len(x_asc) - 1)
                rojo_valvula_izq = (
                    float(x_asc[indice_rojo_valvula_izq]),
                    float(y_asc[indice_rojo_valvula_izq]),
                )
            if all(punto is not None for punto in (
                rojo_valvula_izq,
                rojo_valvula_der,
                azul_valvula_izq,
                azul_valvula_der,
            )):
                azules_demasiado_altos = bool(
                    azul_izquierda
                    > azul_valvula_izq[1] + 0.18 * rango_y
                    and azul_derecha
                    > azul_valvula_der[1] + 0.18 * rango_y
                    and azul_izquierda >= y_min_global + 0.85 * rango_y
                    and azul_derecha >= y_min_global + 0.60 * rango_y
                )
                orden_vertical_valido = bool(
                    rojo_valvula_izq[1]
                    > azul_valvula_izq[1] + 0.35 * rango_y
                    and rojo_valvula_der[1]
                    > azul_valvula_der[1] + 0.35 * rango_y
                    and azul_valvula_der[1]
                    <= azul_valvula_izq[1] - 0.08 * rango_y
                )
                if azules_demasiado_altos and orden_vertical_valido:
                    x_roja_izq, roja_izquierda = rojo_valvula_izq
                    x_roja_der, roja_derecha = rojo_valvula_der[:2]
                    x_azul_izq, azul_izquierda = azul_valvula_izq[:2]
                    x_azul_der, azul_derecha = azul_valvula_der[:2]
                    salida["Metodo_SAM_Seleccionado"] = (
                        "SAM_MODIFICADO_TRANSICIONES_LATERALES_VALVULA"
                    )

            if (
                salida.get("Metodo_SAM_Seleccionado")
                != "SAM_MODIFICADO_TRANSICIONES_LATERALES_VALVULA"
            ):
                salida["Metodo_SAM_Seleccionado"] = (
                    "SAM_MODIFICADO_MORFOLOGIA_VALVULA_VIAJERA"
                )
        else:
            # Salvaguarda local para el error observado en cartas llenas: una
            # reconciliación de niveles no puede alejar un azul hacia arriba
            # del pie de la transferencia recta que ya detectó el método
            # base. Se corrige cada lado de manera independiente.
            azul_base_izq = candidato_izquierdo_base[:2]
            azul_base_der = candidato_derecho_base[:2]
            if azul_izquierda > azul_base_izq[1] + 0.12 * rango_y:
                x_azul_izq, azul_izquierda = azul_base_izq
            if azul_derecha > azul_base_der[1] + 0.12 * rango_y:
                x_azul_der, azul_derecha = azul_base_der

        # Fine tuning morfológico: sólo reemplaza el resultado conservador
        # cuando las cuatro esquinas presentan transiciones claras. Las
        # cartas irregulares conservan íntegramente los criterios anteriores.
        geometria = {
            "rojo_izq": frontera_geometrica_rama(
                x_asc, y_asc, "izquierda"
            ),
            "rojo_der": frontera_geometrica_rama(
                x_asc, y_asc, "derecha"
            ),
            "azul_izq": frontera_geometrica_rama(
                x_desc, y_desc, "izquierda"
            ),
            "azul_der": frontera_geometrica_rama(
                x_desc, y_desc, "derecha"
            ),
        }

        # Morfología de compresión/interferencia de gas: envolvente superior
        # extensa y casi plana, valle inferior profundo y recuperación fuerte
        # hacia la derecha. Se reconoce sólo por geometría porque el SAM se
        # calcula antes que las etiquetas diagnósticas.
        mascara_meseta_gas = (
            (x_asc >= x_min_global + 0.22 * rango_x)
            & (x_asc <= x_min_global + 0.72 * rango_x)
        )
        mascara_derecha_gas = x_desc >= x_min_global + 0.72 * rango_x
        meseta_gas = y_asc[mascara_meseta_gas]
        derecha_gas = y_desc[mascara_derecha_gas]
        morfologia_compresion_gas = bool(
            not morfologia_valvula_viajera
            and len(meseta_gas) >= 5
            and len(derecha_gas) >= 3
            and float(np.ptp(meseta_gas)) <= 0.22 * rango_y
            and float(np.max(derecha_gas) - np.min(y_desc))
            >= 0.30 * rango_y
        )

        # Carta llena no rectangular con envolvente superior arqueada. Los
        # máximos interiores no son codos: los rojos deben permanecer en los
        # corredores laterales, antes/después de las rodillas. Esta ruta no
        # se mezcla con gas ni con la transferencia oblicua de viajera.
        nivel_central_superior = (
            float(np.median(meseta_gas))
            if len(meseta_gas) >= 5
            else np.nan
        )
        sobrealtura_superior = (
            (float(np.max(y_asc)) - nivel_central_superior) / rango_y
            if np.isfinite(nivel_central_superior)
            else np.nan
        )
        morfologia_superior_arqueada = bool(
            not morfologia_valvula_viajera
            and not morfologia_compresion_gas
            and np.isfinite(sobrealtura_superior)
            and sobrealtura_superior >= 0.10
            and len(meseta_gas) >= 5
            and float(np.ptp(meseta_gas)) <= 0.24 * rango_y
        )

        def frontera_recuperacion_inferior(x, y):
            """Cambio de recuperación fuerte a pseudo horizontal inferior."""
            orden = np.argsort(x, kind="stable")
            xx = np.asarray(x, dtype=float)[orden]
            yy = np.asarray(y, dtype=float)[orden]
            indice_minimo = int(np.argmin(yy))
            xx = xx[indice_minimo:]
            yy = yy[indice_minimo:]
            if len(xx) < 8:
                return None
            xn = (xx - x_min_global) / rango_x
            yn = (yy - y_min_global) / rango_y

            mejor = None
            for corte in range(3, len(xx) - 3):
                if (
                    np.ptp(xn[:corte + 1]) < 0.10
                    or np.ptp(xn[corte:]) < 0.16
                ):
                    continue
                coef_antes = np.polyfit(xn[:corte + 1], yn[:corte + 1], 1)
                coef_despues = np.polyfit(xn[corte:], yn[corte:], 1)
                pendiente_antes = float(coef_antes[0])
                pendiente_despues = float(coef_despues[0])
                if (
                    pendiente_antes < 0.45
                    or pendiente_antes - pendiente_despues < 0.22
                ):
                    continue
                residuo = float(
                    np.mean((yn[:corte + 1] - np.polyval(
                        coef_antes, xn[:corte + 1]
                    )) ** 2)
                    + np.mean((yn[corte:] - np.polyval(
                        coef_despues, xn[corte:]
                    )) ** 2)
                )
                if mejor is None or residuo < mejor[0]:
                    mejor = (residuo, corte)
            if mejor is None:
                return None
            indice = mejor[1]
            return float(xx[indice]), float(yy[indice])

        def fin_rodilla_superior_derecha_gas(x, y, x_inicio):
            """Hombro posterior a la recuperación de carga en cartas con gas.

            Ajusta dos rectas sobre la recuperación derecha: una al tramo
            empinado que sale de la zona inferior y otra al tramo posterior,
            de menor pendiente. El empalme representa el final de la rodilla,
            aun cuando la curva no llegue a formar una meseta horizontal.
            """
            orden = np.argsort(x, kind="stable")
            xx = np.asarray(x, dtype=float)[orden]
            yy = np.asarray(y, dtype=float)[orden]

            # Los retornos verticales y las muestras repetidas en posición
            # sesgan fuertemente las derivadas. Se consolidan por mediana.
            x_unicos = np.unique(xx)
            y_unicos = np.asarray([
                float(np.median(yy[np.isclose(xx, valor)]))
                for valor in x_unicos
            ])
            mascara = x_unicos >= max(
                float(x_inicio), x_min_global + 0.45 * rango_x
            )
            xx = x_unicos[mascara]
            yy = y_unicos[mascara]
            if len(xx) < 10 or np.ptp(xx) < 0.22 * rango_x:
                return None

            xn = (xx - x_min_global) / rango_x
            yn = (yy - y_min_global) / rango_y

            # La transferencia de gas es progresiva: recorrer del 20 al
            # 80 % de carga consume una fracción apreciable de la carrera.
            # Este filtro evita aplicar el ajuste a golpes de fluido, cuya
            # transferencia derecha es mucho más abrupta.
            amplitud_recuperacion = float(np.ptp(yn))
            if amplitud_recuperacion < 0.20:
                return None
            yn_recuperacion = (yn - float(np.min(yn))) / amplitud_recuperacion
            cruces_20 = np.flatnonzero(yn_recuperacion >= 0.20)
            cruces_80 = np.flatnonzero(yn_recuperacion >= 0.80)
            if len(cruces_20) == 0 or len(cruces_80) == 0:
                return None
            ancho_transferencia = float(
                xn[int(cruces_80[0])] - xn[int(cruces_20[0])]
            )
            if (
                ancho_transferencia < 0.20
                and x_min_global >= 0.0
            ):
                return None

            mejor = None
            for corte in range(4, len(xx) - 3):
                ancho_antes = float(xn[corte] - xn[0])
                ancho_despues = float(xn[-1] - xn[corte])
                if ancho_antes < 0.10 or ancho_despues < 0.055:
                    continue
                coef_antes = np.polyfit(xn[:corte + 1], yn[:corte + 1], 1)
                coef_despues = np.polyfit(xn[corte:], yn[corte:], 1)
                pendiente_antes = float(coef_antes[0])
                pendiente_despues = float(coef_despues[0])
                reduccion = pendiente_antes - pendiente_despues
                if (
                    pendiente_antes < 0.42
                    or reduccion < max(0.12, 0.16 * pendiente_antes)
                    or pendiente_despues < -0.12
                ):
                    continue
                residuo = float(
                    np.mean((yn[:corte + 1] - np.polyval(
                        coef_antes, xn[:corte + 1]
                    )) ** 2)
                    + np.mean((yn[corte:] - np.polyval(
                        coef_despues, xn[corte:]
                    )) ** 2)
                )
                # Evita que pequeñas ondulaciones del extremo ganen frente
                # al hombro principal de la S de recuperación.
                residuo += 0.0025 * max(0.0, 0.09 - ancho_despues)
                if mejor is None or residuo < mejor[0]:
                    mejor = (residuo, corte)
            if mejor is None:
                return None
            indice = mejor[1]
            return float(xx[indice]), float(yy[indice])

        ajuste_superior_arqueado_aplicado = False
        if morfologia_compresion_gas:
            # Cada esquina se corrige de forma independiente. Una esquina
            # que ya coincide con su transición queda intacta.
            rojo_gas_izq = geometria["rojo_izq"]
            azul_gas_izq = geometria["azul_izq"]
            if (
                rojo_gas_izq is not None
                and x_roja_izq > rojo_gas_izq[0] + 0.025 * rango_x
            ):
                x_roja_izq, roja_izquierda = rojo_gas_izq[:2]
            if (
                azul_gas_izq is not None
                and azul_izquierda > azul_gas_izq[1] + 0.06 * rango_y
            ):
                x_azul_izq, azul_izquierda = azul_gas_izq[:2]

            # En algunas cartas de gas la rama ascendente comienza ya a media
            # altura del lateral izquierdo. Ese primer punto no representa el
            # codo azul: el pie real permanece en la descendente, apenas
            # iniciada la transferencia vertical. Se usa solamente cuando el
            # candidato mejora materialmente la coherencia con el azul derecho.
            azul_izquierdo_antes_ajuste_gas = (
                x_azul_izq, azul_izquierda
            )
            ajuste_pie_izquierdo_gas = False
            indices_pie_izquierdo = np.flatnonzero(
                (x_desc <= x_min_global + 0.025 * rango_x)
                & (y_desc <= y_min_global + 0.48 * rango_y)
            )
            if len(indices_pie_izquierdo):
                indice_pie_izquierdo = int(indices_pie_izquierdo[
                    np.argmin(y_desc[indices_pie_izquierdo])
                ])
                azul_pie_izquierdo = (
                    float(x_desc[indice_pie_izquierdo]),
                    float(y_desc[indice_pie_izquierdo]),
                )
                diferencia_actual_azules = abs(
                    azul_izquierda - azul_derecha
                )
                diferencia_propuesta_azules = abs(
                    azul_pie_izquierdo[1] - azul_derecha
                )
                if (
                    azul_izquierda
                    > azul_pie_izquierdo[1] + 0.12 * rango_y
                    and azul_izquierda >= y_min_global + 0.55 * rango_y
                    and azul_derecha <= y_min_global + 0.30 * rango_y
                    and diferencia_propuesta_azules
                    < diferencia_actual_azules - 0.08 * rango_y
                    and diferencia_propuesta_azules <= 0.22 * rango_y
                ):
                    x_azul_izq, azul_izquierda = azul_pie_izquierdo
                    ajuste_pie_izquierdo_gas = True

            # En fondos ondulados, un valle interior puede parecer un codo
            # inferior. El azul izquierdo debe haber alcanzado realmente el
            # lateral: poca variación de posición mientras se transfiere una
            # fracción importante de carga. El detector lateral original de
            # la ascendente aporta ese pie y sólo reemplaza al candidato si
            # éste todavía está materialmente desplazado hacia el interior.
            azul_lateral_izq = candidato_izquierdo_base[:2]
            if (
                x_azul_izq > azul_lateral_izq[0] + 0.055 * rango_x
                and azul_lateral_izq[0]
                <= x_min_global + 0.055 * rango_x
                and azul_lateral_izq[1]
                > azul_izquierda + 0.03 * rango_y
                and azul_lateral_izq[1]
                <= roja_izquierda - 0.18 * rango_y
            ):
                x_azul_izq, azul_izquierda = azul_lateral_izq

            # Cierre derecho no rectangular: el rojo pertenece a la rama
            # descendente. Marca el final de la recuperación fuerte de carga,
            # cuando termina la rodilla y comienza el tramo alto casi plano.
            orden_sup_gas = np.argsort(x_desc, kind="stable")
            x_sup_gas = x_desc[orden_sup_gas]
            y_sup_gas = y_desc[orden_sup_gas]
            dx_gas = np.diff(x_sup_gas / rango_x)
            dy_gas = np.diff(y_sup_gas / rango_y)
            pendientes_gas = np.full(len(dx_gas), np.nan)
            validas_gas = dx_gas > 1e-4
            pendientes_gas[validas_gas] = (
                dy_gas[validas_gas] / dx_gas[validas_gas]
            )
            inicio_gas = int(np.searchsorted(
                x_sup_gas, max(
                    x_azul_der,
                    x_min_global + 0.45 * rango_x,
                )
            ))
            rojo_gas_der = fin_rodilla_superior_derecha_gas(
                x_desc, y_desc, x_azul_der
            )
            vio_recuperacion_fuerte = False
            for k in range(inicio_gas, len(pendientes_gas) - 1):
                if rojo_gas_der is not None:
                    break
                entorno = pendientes_gas[max(inicio_gas, k - 1):k + 2]
                entorno = entorno[np.isfinite(entorno)]
                if len(entorno) < 2:
                    continue
                pendiente_local = float(np.median(entorno))
                if pendiente_local >= 0.45:
                    vio_recuperacion_fuerte = True
                    continue
                if vio_recuperacion_fuerte and pendiente_local <= 0.22:
                    indice = min(k + 1, len(x_sup_gas) - 1)
                    rojo_gas_der = (
                        float(x_sup_gas[indice]),
                        float(y_sup_gas[indice]),
                    )
                    break
            # En rodillas derechas muy agudas el ajuste puede quedarse en la
            # posición máxima, todavía sobre la horizontal superior. Como
            # corrección incremental se exige haber iniciado realmente la
            # carrera descendente: desplazamiento a la izquierda y caída de
            # carga simultáneos, sin avanzar hasta la horizontal inferior.
            carga_extremo_derecho = float(np.max(y_desc[
                x_desc >= x_max_global - 0.02 * rango_x
            ]))
            indices_descenso_establecido = np.flatnonzero(
                (x_desc <= x_max_global - 0.06 * rango_x)
                & (x_desc >= x_max_global - 0.30 * rango_x)
                & (y_desc <= carga_extremo_derecho - 0.025 * rango_y)
                & (y_desc >= y_min_global + 0.52 * rango_y)
            )
            if len(indices_descenso_establecido):
                indice_descenso = int(indices_descenso_establecido[
                    np.argmax(y_desc[indices_descenso_establecido])
                ])
                rojo_descenso_establecido = (
                    float(x_desc[indice_descenso]),
                    float(y_desc[indice_descenso]),
                )
                if (
                    rojo_gas_der is None
                    or rojo_gas_der[1]
                    > rojo_descenso_establecido[1] + 0.015 * rango_y
                ):
                    rojo_gas_der = rojo_descenso_establecido
            if (
                rojo_gas_der is not None
                and rojo_gas_der[1] > azul_derecha + 0.12 * rango_y
            ):
                x_roja_der, roja_derecha = rojo_gas_der

            # Cuando la rodilla superior derecha pertenece a la ascendente,
            # el detector general entrega el primer punto dentro de la caída.
            # Para el SAM se avanza hasta tres muestras desde ese inicio: las
            # primeras todavía pueden pertenecer a la rodilla, mientras que
            # la tercera ya se encuentra en la pseudovertical de transferencia.
            rojo_transferencia_asc = detectar_recta_superior_derecha(
                x_asc, y_asc
            )
            ajuste_rojo_derecho_gas = False
            if (
                ajuste_pie_izquierdo_gas
                and rojo_transferencia_asc is not None
            ):
                indices_transferencia_asc = np.flatnonzero(
                    (x_asc == rojo_transferencia_asc[0])
                    & (y_asc == rojo_transferencia_asc[1])
                )
                if len(indices_transferencia_asc):
                    indice_transferencia_asc = min(
                        int(indices_transferencia_asc[0]) + 3,
                        len(x_asc) - 1,
                    )
                    rojo_vertical_asc = (
                        float(x_asc[indice_transferencia_asc]),
                        float(y_asc[indice_transferencia_asc]),
                    )
                    if (
                        rojo_vertical_asc[1]
                        > azul_derecha + 0.35 * rango_y
                        and roja_derecha
                        > rojo_vertical_asc[1] + 0.015 * rango_y
                    ):
                        x_roja_der, roja_derecha = rojo_vertical_asc
                        ajuste_rojo_derecho_gas = True

            # Las dos desviaciones forman una misma firma geométrica. Si el
            # rojo derecho no estaba todavía sobre la rodilla, no se modifica
            # aisladamente el azul izquierdo: eso sería una recalibración más
            # amplia de cartas cuya horizontal superior ya era coherente.
            if (
                ajuste_pie_izquierdo_gas
                and not ajuste_rojo_derecho_gas
            ):
                x_azul_izq, azul_izquierda = (
                    azul_izquierdo_antes_ajuste_gas
                )
                ajuste_pie_izquierdo_gas = False

            azul_gas_der = frontera_recuperacion_inferior(x_desc, y_desc)
            if (
                azul_gas_der is not None
                and (
                    x_azul_der > azul_gas_der[0] + 0.12 * rango_x
                    or azul_derecha > azul_gas_der[1] + 0.10 * rango_y
                )
            ):
                x_azul_der, azul_derecha = azul_gas_der

            # Rectangular con ondulaciones de fondo: los agarres pueden
            # producir falsos codos interiores. Se exige una meseta central
            # extensa y se recorren los laterales reales en el orden de la
            # traza. El azul derecho se toma antes del salto a la horizontal.
            mascara_central_asc = (
                (x_asc >= x_min_global + 0.18 * rango_x)
                & (x_asc <= x_min_global + 0.76 * rango_x)
            )
            mascara_central_desc = (
                (x_desc >= x_min_global + 0.18 * rango_x)
                & (x_desc <= x_min_global + 0.76 * rango_x)
            )
            lateral_derecho_desc = y_desc[
                x_desc >= x_max_global - 0.08 * rango_x
            ]
            rectangular_ruidosa = bool(
                not ajuste_pie_izquierdo_gas
                and
                np.count_nonzero(mascara_central_asc) >= 5
                and np.count_nonzero(mascara_central_desc) >= 5
                and len(lateral_derecho_desc) >= 5
                and np.ptp(y_asc[mascara_central_asc]) <= 0.20 * rango_y
                and np.ptp(y_desc[mascara_central_desc]) <= 0.36 * rango_y
                and np.ptp(lateral_derecho_desc) >= 0.34 * rango_y
            )
            if rectangular_ruidosa:
                # Ascendente orientada desde el lateral izquierdo.
                if x_asc[0] <= x_min_global + 0.08 * rango_x:
                    xa_rect, ya_rect = x_asc, y_asc
                else:
                    xa_rect, ya_rect = x_asc[::-1], y_asc[::-1]
                candidatos_inicio_izq = np.flatnonzero(
                    xa_rect <= x_min_global + 0.025 * rango_x
                )
                rojo_rect_izq = None
                if len(candidatos_inicio_izq):
                    indice = int(candidatos_inicio_izq[0])
                    if ya_rect[indice] > azul_izquierda + 0.25 * rango_y:
                        rojo_rect_izq = (
                            float(xa_rect[indice]), float(ya_rect[indice])
                        )

                # Descendente orientada desde el extremo superior derecho.
                if x_desc[0] >= x_max_global - 0.08 * rango_x:
                    xd_rect, yd_rect = x_desc, y_desc
                else:
                    xd_rect, yd_rect = x_desc[::-1], y_desc[::-1]
                rojo_rect_der = None
                if len(xd_rect):
                    objetivo_caida = yd_rect[0] - 0.10 * rango_y
                    candidatos = np.flatnonzero(
                        (xd_rect >= x_max_global - 0.04 * rango_x)
                        & (yd_rect <= objetivo_caida)
                    )
                    if len(candidatos):
                        indice = int(candidatos[0])
                        rojo_rect_der = (
                            float(xd_rect[indice]), float(yd_rect[indice])
                        )

                azul_rect_der = None
                desplazamiento_rect = (
                    x_max_global - xd_rect
                ) / rango_x
                for k in range(1, len(xd_rect) - 2):
                    if desplazamiento_rect[k] >= 0.020:
                        indice = k
                        azul_rect_der = (
                            float(xd_rect[indice]), float(yd_rect[indice])
                        )
                        break

                if all(punto is not None for punto in (
                    rojo_rect_izq, rojo_rect_der, azul_rect_der
                )):
                    superior_rect = 0.5 * (
                        rojo_rect_izq[1] + rojo_rect_der[1]
                    )
                    inferior_rect = 0.5 * (
                        azul_izquierda + azul_rect_der[1]
                    )
                    peso_actual = 0.5 * (
                        roja_izquierda + roja_derecha
                    ) - 0.5 * (azul_izquierda + azul_derecha)
                    peso_rect = superior_rect - inferior_rect
                    if 0.20 * rango_y <= peso_rect <= peso_actual + 1e-9:
                        x_roja_izq, roja_izquierda = rojo_rect_izq
                        x_roja_der, roja_derecha = rojo_rect_der
                        x_azul_der, azul_derecha = azul_rect_der
                        salida["Metodo_SAM_Seleccionado"] = (
                            "SAM_MODIFICADO_RECTANGULAR_RUIDO_LATERALES"
                        )
            salida["Metodo_SAM_Seleccionado"] = (
                salida.get("Metodo_SAM_Seleccionado")
                if salida.get("Metodo_SAM_Seleccionado")
                == "SAM_MODIFICADO_RECTANGULAR_RUIDO_LATERALES"
                else "SAM_MODIFICADO_MORFOLOGIA_COMPRESION_GAS"
            )
        elif morfologia_superior_arqueada:
            objetivo_rojos = y_min_global + 0.66 * rango_y
            x_laterales = np.concatenate([x_asc, x_desc])
            y_laterales = np.concatenate([y_asc, y_desc])
            indices_rojo_izq = np.flatnonzero(
                x_laterales <= x_min_global + 0.12 * rango_x
            )
            indices_rojo_der = np.flatnonzero(
                x_asc >= x_max_global - 0.25 * rango_x
            )
            if len(indices_rojo_izq) and len(indices_rojo_der):
                indice_rojo_izq = int(indices_rojo_izq[np.argmin(
                    abs(y_laterales[indices_rojo_izq] - objetivo_rojos)
                )])
                indice_rojo_der = int(indices_rojo_der[np.argmin(
                    abs(y_asc[indices_rojo_der] - objetivo_rojos)
                )])
                rojo_arqueado_izq = (
                    float(x_laterales[indice_rojo_izq]),
                    float(y_laterales[indice_rojo_izq]),
                )
                rojo_arqueado_der = (
                    float(x_asc[indice_rojo_der]),
                    float(y_asc[indice_rojo_der]),
                )
                media_roja_actual = 0.5 * (
                    roja_izquierda + roja_derecha
                )
                media_roja_arqueada = 0.5 * (
                    rojo_arqueado_izq[1] + rojo_arqueado_der[1]
                )
                if (
                    media_roja_actual
                    > media_roja_arqueada + 0.06 * rango_y
                ):
                    x_roja_izq, roja_izquierda = rojo_arqueado_izq
                    x_roja_der, roja_derecha = rojo_arqueado_der
                    salida["Metodo_SAM_Seleccionado"] = (
                        "SAM_MODIFICADO_LATERALES_SUPERIORES_ARQUEADOS"
                    )
                    ajuste_superior_arqueado_aplicado = True
        if (
            not morfologia_compresion_gas
            and not ajuste_superior_arqueado_aplicado
            and all(punto is not None for punto in geometria.values())
        ):
            rojo_geom_izq = geometria["rojo_izq"][:2]
            rojo_geom_der = geometria["rojo_der"][:2]
            azul_geom_izq = geometria["azul_izq"][:2]
            azul_geom_der = geometria["azul_der"][:2]
            # Rectangulares normales: si ambos laterales cubren gran parte de
            # la carga con poca variación de posición, se usan bandas internas
            # de esas verticales. Es independiente de la densidad de muestreo
            # y evita elegir el vértice de la horizontal o saltar demasiadas
            # muestras cuando el lateral está poco discretizado.
            x_todas = np.concatenate([x_asc, x_desc])
            y_todas = np.concatenate([y_asc, y_desc])

            def puntos_banda_lateral(lado):
                if lado == "izquierda":
                    mascara_lateral = (
                        x_todas <= x_min_global + 0.16 * rango_x
                    )
                else:
                    mascara_lateral = (
                        x_todas >= x_max_global - 0.16 * rango_x
                    )
                indices_lateral = np.flatnonzero(mascara_lateral)
                if (
                    len(indices_lateral) < 6
                    or np.ptp(y_todas[indices_lateral]) < 0.48 * rango_y
                    or np.ptp(x_todas[indices_lateral]) > 0.17 * rango_x
                ):
                    return None
                objetivo_azul = y_min_global + 0.28 * rango_y
                objetivo_rojo = y_min_global + 0.75 * rango_y
                distancia_exterior = (
                    (x_todas[indices_lateral] - x_min_global) / rango_x
                    if lado == "izquierda"
                    else (x_max_global - x_todas[indices_lateral]) / rango_x
                )
                score_azul_banda = (
                    abs(y_todas[indices_lateral] - objetivo_azul) / rango_y
                    + 0.30 * distancia_exterior
                )
                score_rojo_banda = (
                    abs(y_todas[indices_lateral] - objetivo_rojo) / rango_y
                    + 0.30 * distancia_exterior
                )
                indice_azul = int(indices_lateral[np.argmin(
                    score_azul_banda
                )])
                indice_rojo = int(indices_lateral[np.argmin(
                    score_rojo_banda
                )])
                return (
                    (float(x_todas[indice_azul]),
                     float(y_todas[indice_azul])),
                    (float(x_todas[indice_rojo]),
                     float(y_todas[indice_rojo])),
                )

            banda_izquierda = puntos_banda_lateral("izquierda")
            banda_derecha = puntos_banda_lateral("derecha")
            if banda_izquierda is not None and banda_derecha is not None:
                azul_geom_izq, _ = banda_izquierda
                azul_geom_der, rojo_geom_der = banda_derecha
            superior_geom = 0.5 * (
                rojo_geom_izq[1] + rojo_geom_der[1]
            )
            inferior_geom = 0.5 * (
                azul_geom_izq[1] + azul_geom_der[1]
            )
            peso_geom = superior_geom - inferior_geom
            coherencia_superior = (
                abs(rojo_geom_izq[1] - rojo_geom_der[1]) <= 0.22 * rango_y
            )
            coherencia_inferior = (
                abs(azul_geom_izq[1] - azul_geom_der[1]) <= 0.22 * rango_y
            )
            if (
                0.18 * rango_y <= peso_geom <= 0.92 * rango_y
                and coherencia_superior
                and coherencia_inferior
            ):
                x_roja_izq, roja_izquierda = rojo_geom_izq
                x_roja_der, roja_derecha = rojo_geom_der
                x_azul_izq, azul_izquierda = azul_geom_izq
                x_azul_der, azul_derecha = azul_geom_der
                salida["Metodo_SAM_Seleccionado"] = (
                    "SAM_MODIFICADO_TRANSICIONES_MORFOLOGICAS"
                )
        carga_superior = float(np.mean([roja_izquierda, roja_derecha]))
        carga_inferior = float(np.mean([azul_izquierda, azul_derecha]))
        peso = carga_superior - carga_inferior
        if not np.isfinite(peso) or peso <= 0:
            raise ValueError("CARGA_FLUIDO_MODIFICADA_NO_POSITIVA")

        profundidad = float(profundidad_bomba_m)
        diametro = float(diametro_piston_pulg)
        pt_kg = float(presion_tubing_kg_cm2)
        pc_kg = float(presion_casing_kg_cm2)
        sg = float(gravedad_especifica)
        gradiente = pd.to_numeric(gradiente_psi_m, errors="coerce")
        if not np.isfinite(gradiente) or gradiente <= 0:
            gradiente = PSI_PIE_AGUA * PIES_POR_METRO * sg
        if (
            not np.isfinite([profundidad, diametro, pt_kg, pc_kg, sg]).all()
            or profundidad <= 0
            or diametro <= 0
            or sg <= 0
        ):
            raise ValueError("DATOS_SAM_FISICAMENTE_INVALIDOS")

        area = float(np.pi * diametro ** 2 / 4.0)
        pt_psi = float(pt_kg * KG_CM2_A_PSI)
        pc_psi = float(pc_kg * KG_CM2_A_PSI)
        pd_psi = float(pt_psi + gradiente * profundidad)
        pip = float(pd_psi - peso / area)
        sumergencia = float((pip - pc_psi) / gradiente)
        salida.update({
            "Calculo_SAM_Modificado_Valido": True,
            "Carga_Roja_Izquierda_SAM_Modificado_lbf": roja_izquierda,
            "Carga_Roja_Derecha_SAM_Modificado_lbf": roja_derecha,
            "Carga_Azul_Izquierda_SAM_Modificado_lbf": azul_izquierda,
            "Carga_Azul_Derecha_SAM_Modificado_lbf": azul_derecha,
            "Posicion_Roja_Izquierda_SAM_Modificado_pulg": x_roja_izq,
            "Posicion_Roja_Derecha_SAM_Modificado_pulg": x_roja_der,
            "Posicion_Azul_Izquierda_SAM_Modificado_pulg": x_azul_izq,
            "Posicion_Azul_Derecha_SAM_Modificado_pulg": x_azul_der,
            "Carga_Superior_SAM_Seleccionada_lbf": carga_superior,
            "Carga_Inferior_SAM_Seleccionada_lbf": carga_inferior,
            "Peso_Fluido_SAM_Seleccionado_lbf": float(peso),
            "Area_Piston_SAM_pulg2": area,
            "Diferencial_Carga_SAM_psi": float(peso / area),
            "Presion_Tubing_SAM_kg_cm2": pt_kg,
            "Presion_Casing_SAM_kg_cm2": pc_kg,
            "Gravedad_Especifica_SAM": sg,
            "Gradiente_SAM_psi_m": float(gradiente),
            "Presion_Descarga_Bomba_SAM_psi": pd_psi,
            "PIP_SAM_Seleccionado_psi": pip,
            "Sumergencia_SAM_Seleccionada_m": sumergencia,
            "Sumergencia_Relativa_SAM_Seleccionada_pct": float(
                100.0 * sumergencia / profundidad
            ),
            "Nivel_Dinamico_SAM_Modificado_m": float(
                profundidad - sumergencia
            ),
        })
    except Exception as error:
        salida["Motivo_SAM_Modificado_No_Valido"] = str(error)
    return salida


def calcular_sumergencia_desde_horizontales(
    carga_superior_lbf,
    carga_inferior_lbf,
    profundidad_bomba_m,
    diametro_piston_pulg,
    horizontales_validas,
    sg_fluido=None,
    factor_carga_hidraulica=1.0,
):
    """Calcula sumergencia independiente usando horizontales finales."""
    salida = {
        "Calculo_Sumergencia_Propia_Valido": False,
        "Motivo_Sumergencia_Propia_No_Valida": "",
        "Peso_Fluido_Horizontales_lbf": np.nan,
        "Carga_Hidraulica_Efectiva_lbf": np.nan,
        "Factor_Carga_Hidraulica": np.nan,
        "Area_Piston_pulg2": np.nan,
        "Presion_Diferencial_Horizontales_psi": np.nan,
        "Presion_Diferencial_Horizontales_kg_cm2": np.nan,
        "Presion_Hidrostatica_Bomba_Sin_Sumergencia_kg_cm2": np.nan,
        "SG_Fluido_Asumido": np.nan,
        "Gradiente_Fluido_Asumido_psi_m": np.nan,
        "Altura_Columna_Equivalente_m": np.nan,
        "Sumergencia_Sobre_Bomba_m": np.nan,
        "Sumergencia_Relativa_Sobre_Bomba_pct": np.nan,
        "Nivel_Dinamico_Propio_m": np.nan,
        "Sumergencia_Propia_m": np.nan,
        "Sumergencia_Relativa_Propia_pct": np.nan,
    }

    if not bool(horizontales_validas):
        salida["Motivo_Sumergencia_Propia_No_Valida"] = (
            "HORIZONTALES_NO_VALIDAS"
        )
        return salida

    valores = np.asarray([
        carga_superior_lbf,
        carga_inferior_lbf,
        profundidad_bomba_m,
        diametro_piston_pulg,
        factor_carga_hidraulica,
    ], dtype=float)
    if not np.isfinite(valores).all():
        salida["Motivo_Sumergencia_Propia_No_Valida"] = (
            "DATOS_INCOMPLETOS"
        )
        return salida

    (
        carga_superior,
        carga_inferior,
        profundidad_m,
        diametro_pulg,
        factor_hidraulico,
    ) = valores
    peso_fluido_lbf = carga_superior - carga_inferior
    if (
        peso_fluido_lbf <= 0
        or profundidad_m <= 0
        or diametro_pulg <= 0
        or factor_hidraulico <= 0
    ):
        salida["Motivo_Sumergencia_Propia_No_Valida"] = (
            "DATOS_FISICOS_NO_VALIDOS"
        )
        return salida

    area_piston = np.pi * diametro_pulg ** 2 / 4.0
    sg_entrada = pd.to_numeric(sg_fluido, errors="coerce")
    if not np.isfinite(sg_entrada) or sg_entrada <= 0:
        sg_petroleo = 141.5 / (
            API_PETROLEO_SUMERGENCIA_PROPIA + 131.5
        )
        sg_entrada = (
            FRACCION_AGUA_SUMERGENCIA_PROPIA
            + FRACCION_PETROLEO_SUMERGENCIA_PROPIA * sg_petroleo
        )
    carga_hidraulica_lbf = peso_fluido_lbf * factor_hidraulico
    presion_diferencial_psi = carga_hidraulica_lbf / area_piston
    presion_kg_cm2 = presion_diferencial_psi / PSI_POR_KG_CM2
    # Convencion operativa usada por el cliente: una columna de fluido de
    # gravedad especifica SG aporta SG kg/cm2 cada 10 m.
    altura_columna_sobre_bomba_m = (
        presion_kg_cm2 * 10.0 / sg_entrada
    )
    gradiente_psi_m = presion_diferencial_psi / max(
        altura_columna_sobre_bomba_m,
        1e-9,
    )
    # La carga medida equivale a la altura de columna desde el nivel
    # dinamico hasta la bomba. Por lo tanto, la sumergencia sobre la bomba
    # es profundidad de bomba menos esa altura equivalente. El nivel
    # dinamico medido desde boca coincide con la altura equivalente.
    sumergencia_sobre_bomba_m = (
        profundidad_m - altura_columna_sobre_bomba_m
    )
    nivel_dinamico_m = altura_columna_sobre_bomba_m
    presion_sin_sumergencia_kg_cm2 = profundidad_m * sg_entrada / 10.0

    salida.update({
        "Calculo_Sumergencia_Propia_Valido": True,
        "Peso_Fluido_Horizontales_lbf": float(peso_fluido_lbf),
        "Carga_Hidraulica_Efectiva_lbf": float(carga_hidraulica_lbf),
        "Factor_Carga_Hidraulica": float(factor_hidraulico),
        "Area_Piston_pulg2": float(area_piston),
        "Presion_Diferencial_Horizontales_psi": float(
            presion_diferencial_psi
        ),
        "Presion_Diferencial_Horizontales_kg_cm2": float(
            presion_kg_cm2
        ),
        "Presion_Hidrostatica_Bomba_Sin_Sumergencia_kg_cm2": float(
            presion_sin_sumergencia_kg_cm2
        ),
        "SG_Fluido_Asumido": float(sg_entrada),
        "Gradiente_Fluido_Asumido_psi_m": float(gradiente_psi_m),
        "Altura_Columna_Equivalente_m": float(
            altura_columna_sobre_bomba_m
        ),
        "Sumergencia_Sobre_Bomba_m": float(sumergencia_sobre_bomba_m),
        "Sumergencia_Relativa_Sobre_Bomba_pct": float(
            100.0 * sumergencia_sobre_bomba_m / profundidad_m
        ),
        "Nivel_Dinamico_Propio_m": float(nivel_dinamico_m),
        # Alias historico: desde esta version representa sumergencia sobre
        # la bomba, no profundidad del nivel desde boca.
        "Sumergencia_Propia_m": float(sumergencia_sobre_bomba_m),
        "Sumergencia_Relativa_Propia_pct": float(
            100.0 * sumergencia_sobre_bomba_m / profundidad_m
        ),
    })
    return salida


def calcular_desplazamiento_bruto_efectivo(
    diametro_piston_pulg,
    carrera_inicio_pulg,
    carrera_fin_pulg,
    gpm,
):
    """Desplazamiento geometrico diario con la carrera efectiva de fondo."""
    valores = pd.to_numeric(
        pd.Series([
            diametro_piston_pulg,
            carrera_inicio_pulg,
            carrera_fin_pulg,
            gpm,
        ]),
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        return {
            "Carrera_Efectiva_Fondo_pulg": np.nan,
            "Desplazamiento_Bruto_Efectivo_m3_d": np.nan,
        }
    diametro, inicio, fin, golpes_minuto = valores
    carrera = fin - inicio
    if diametro <= 0 or carrera <= 0 or golpes_minuto <= 0:
        return {
            "Carrera_Efectiva_Fondo_pulg": np.nan,
            "Desplazamiento_Bruto_Efectivo_m3_d": np.nan,
        }
    area = np.pi * diametro ** 2 / 4.0
    desplazamiento = (
        area * carrera * golpes_minuto * 1440.0 * 1.6387064e-5
    )
    return {
        "Carrera_Efectiva_Fondo_pulg": float(carrera),
        "Desplazamiento_Bruto_Efectivo_m3_d": float(desplazamiento),
    }


def calcular_desplazamiento_desde_carrera(
    diametro_piston_pulg,
    carrera_pulg,
    gpm,
):
    """Desplazamiento diario a partir de una carrera ya determinada."""
    valores = pd.to_numeric(
        pd.Series([diametro_piston_pulg, carrera_pulg, gpm]),
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        return np.nan
    diametro, carrera, golpes_minuto = valores
    if diametro <= 0 or carrera <= 0 or golpes_minuto <= 0:
        return np.nan
    area = np.pi * diametro ** 2 / 4.0
    return float(
        area * carrera * golpes_minuto * 1440.0 * 1.6387064e-5
    )


def estimar_carrera_efectiva_en_horizontal_superior(
    posicion,
    carga,
    carga_horizontal_superior,
):
    """Mide la distancia entre cruces de la carta con el nivel superior.

    Esta es una medicion geometrica independiente de los campos de carrera
    efectiva de la API. Los cruces se interpolan linealmente sobre el
    contorno secuencial de la carta.
    """
    x = np.asarray(posicion, dtype=float)
    y = np.asarray(carga, dtype=float)
    nivel = pd.to_numeric(carga_horizontal_superior, errors="coerce")
    salida = {
        "Carrera_Efectiva_Fondo_Calculada_pulg": np.nan,
        "Posicion_Cruce_Superior_Izquierda_pulg": np.nan,
        "Posicion_Cruce_Superior_Derecha_pulg": np.nan,
        "Cantidad_Cruces_Horizontal_Superior": 0,
    }
    if (
        x.size < 3
        or x.size != y.size
        or not np.isfinite(nivel)
    ):
        return salida

    cruces = []
    for indice in range(x.size):
        siguiente = (indice + 1) % x.size
        x0, x1 = x[indice], x[siguiente]
        y0, y1 = y[indice], y[siguiente]
        if not np.isfinite([x0, x1, y0, y1]).all():
            continue
        d0, d1 = y0 - nivel, y1 - nivel
        if d0 == 0:
            cruces.append(float(x0))
        if d0 * d1 < 0 and y1 != y0:
            fraccion = (nivel - y0) / (y1 - y0)
            cruces.append(float(x0 + fraccion * (x1 - x0)))

    if not cruces:
        return salida
    cruces = np.asarray(sorted(cruces), dtype=float)
    tolerancia = max(float(np.ptp(x)) * 0.002, 1e-6)
    unicos = [float(cruces[0])]
    for valor in cruces[1:]:
        if abs(valor - unicos[-1]) > tolerancia:
            unicos.append(float(valor))
    salida["Cantidad_Cruces_Horizontal_Superior"] = len(unicos)
    if len(unicos) < 2:
        return salida

    izquierda = min(unicos)
    derecha = max(unicos)
    carrera = derecha - izquierda
    if carrera <= 0:
        return salida
    salida.update({
        "Carrera_Efectiva_Fondo_Calculada_pulg": float(carrera),
        "Posicion_Cruce_Superior_Izquierda_pulg": float(izquierda),
        "Posicion_Cruce_Superior_Derecha_pulg": float(derecha),
    })
    return salida


def calcular_sumergencia_relativa(
    sumergencia_m,
    profundidad_bomba_m,
):
    """Calcula sumergencia/profundidad en porcentaje, preservando NaN."""
    sumergencia = pd.to_numeric(
        pd.Series(sumergencia_m),
        errors="coerce",
    )
    profundidad = pd.to_numeric(
        pd.Series(profundidad_bomba_m),
        errors="coerce",
    )
    resultado = np.where(
        profundidad.abs() > 1e-9,
        100.0 * sumergencia / profundidad,
        np.nan,
    )
    if np.ndim(sumergencia_m) == 0 and np.ndim(profundidad_bomba_m) == 0:
        return float(resultado[0])
    return resultado


def _pendiente_theil_sen(dias, valores):
    """Pendiente robusta: mediana de pendientes entre pares."""
    pendientes = []
    for indice in range(len(valores) - 1):
        delta_dias = dias[indice + 1:] - dias[indice]
        delta_valores = valores[indice + 1:] - valores[indice]
        validos = delta_dias > 0
        if np.any(validos):
            pendientes.extend(
                (delta_valores[validos] / delta_dias[validos]).tolist()
            )
    return float(np.median(pendientes)) if pendientes else np.nan


def calcular_indicadores_moviles_15d(tabla):
    """
    Calcula indicadores robustos sobre los últimos 15 días calendario.

    La tendencia usa medianas diarias para que los días con más mediciones
    no tengan mayor peso estadístico.
    """
    if tabla is None or tabla.empty:
        return pd.DataFrame()

    trabajo = tabla.copy()
    trabajo["Fecha"] = pd.to_datetime(trabajo["Fecha"], errors="coerce")
    trabajo = trabajo.dropna(subset=["Fecha"])
    if trabajo.empty:
        return pd.DataFrame()

    if {
        "Carga_Maxima_Fondo_lbf",
        "Carga_Minima_Fondo_lbf",
    }.issubset(trabajo.columns):
        trabajo["Rango_Carga_Fondo_lbf"] = (
            pd.to_numeric(
                trabajo["Carga_Maxima_Fondo_lbf"],
                errors="coerce",
            )
            - pd.to_numeric(
                trabajo["Carga_Minima_Fondo_lbf"],
                errors="coerce",
            )
        )

    if {
        "Carrera_Fondo_Total_pulg",
        "Carrera_Superficie_pulg",
    }.issubset(trabajo.columns):
        superficie = pd.to_numeric(
            trabajo["Carrera_Superficie_pulg"],
            errors="coerce",
        )
        trabajo["Eficiencia_Carrera_pct"] = np.where(
            superficie.abs() > 1e-9,
            100.0
            * pd.to_numeric(
                trabajo["Carrera_Fondo_Total_pulg"],
                errors="coerce",
            )
            / superficie,
            np.nan,
        )

    if {
        "Sumergencia_API_m",
        "Profundidad_Bomba_m",
    }.issubset(trabajo.columns):
        trabajo["Sumergencia_Relativa_API_pct"] = (
            calcular_sumergencia_relativa(
                trabajo["Sumergencia_API_m"],
                trabajo["Profundidad_Bomba_m"],
            )
        )

    variables = [
        ("Llenado_Bomba_API_pct", "Llenado de bomba", "%"),
        (
            "Sumergencia_Relativa_API_pct",
            "Sumergencia relativa API",
            "%",
        ),
        ("Peso_Fluido_Promedio_lbf", "Peso de fluido promedio", "lbf"),
        ("Carga_Maxima_Fondo_lbf", "Carga máxima de fondo", "lbf"),
        ("Carga_Minima_Fondo_lbf", "Carga mínima de fondo", "lbf"),
        ("Rango_Carga_Fondo_lbf", "Apertura de cargas de fondo", "lbf"),
        ("Carrera_Fondo_Total_pulg", "Carrera de fondo", "pulg"),
        ("Carrera_Superficie_pulg", "Carrera de superficie", "pulg"),
        (
            "Eficiencia_Carrera_pct",
            "Eficiencia de carrera fondo/superficie",
            "%",
        ),
        ("Torque_Reductor_pct", "Torque reductor", "%"),
        ("Carga_Estructural_pct", "Carga estructural", "%"),
    ]

    fecha_final = trabajo["Fecha"].max()
    fecha_inicial = fecha_final.floor("D") - pd.Timedelta(days=14)
    ventana = trabajo.loc[trabajo["Fecha"] >= fecha_inicial].copy()
    ventana["Dia"] = ventana["Fecha"].dt.floor("D")

    filas = []
    for columna, etiqueta, unidad in variables:
        if (
            columna not in ventana.columns
            or not ventana[columna].notna().any()
        ):
            continue

        datos_crudos = (
            ventana[["Fecha", "Dia", columna]]
            .dropna()
            .sort_values("Fecha")
        )
        diaria = (
            datos_crudos
            .groupby("Dia", as_index=False)[columna]
            .median()
            .sort_values("Dia")
        )
        valores = diaria[columna].to_numpy(dtype=float)
        dias = (
            (diaria["Dia"] - diaria["Dia"].min())
            .dt.total_seconds()
            .to_numpy(dtype=float)
            / 86400.0
        )

        mediana = float(np.median(valores))
        mad = float(np.median(np.abs(valores - mediana)))
        volatilidad = (
            100.0 * 1.4826 * mad / abs(mediana)
            if abs(mediana) > 1e-9
            else np.nan
        )
        pendiente = _pendiente_theil_sen(dias, valores)
        pendiente_relativa = (
            100.0 * pendiente / abs(mediana)
            if np.isfinite(pendiente) and abs(mediana) > 1e-9
            else np.nan
        )

        corte_3d = fecha_final.floor("D") - pd.Timedelta(days=2)
        ultimos_3 = diaria.loc[diaria["Dia"] >= corte_3d, columna]
        anteriores_12 = diaria.loc[diaria["Dia"] < corte_3d, columna]
        cambio = (
            float(np.median(ultimos_3) - np.median(anteriores_12))
            if not ultimos_3.empty and not anteriores_12.empty
            else np.nan
        )
        base_12 = (
            float(np.median(anteriores_12))
            if not anteriores_12.empty
            else np.nan
        )
        cambio_pct = (
            100.0 * cambio / abs(base_12)
            if np.isfinite(cambio)
            and np.isfinite(base_12)
            and abs(base_12) > 1e-9
            else np.nan
        )

        dias_con_datos = int(len(diaria))
        calidad = (
            "Alta" if dias_con_datos >= 12
            else "Media" if dias_con_datos >= 8
            else "Baja" if dias_con_datos >= 5
            else "Insuficiente"
        )
        filas.append({
            "Variable": etiqueta,
            "Unidad": unidad,
            "Último": float(datos_crudos[columna].iloc[-1]),
            "Mediana_15d": mediana,
            "Pendiente_por_día": pendiente,
            "Pendiente_relativa_pct_día": pendiente_relativa,
            "Volatilidad_MAD_pct": volatilidad,
            "Cambio_3d_vs_12d": cambio,
            "Cambio_3d_vs_12d_pct": cambio_pct,
            "Días_con_datos": dias_con_datos,
            "Mediciones": int(len(datos_crudos)),
            "Calidad": calidad,
        })

    return pd.DataFrame(filas)


def analizar_subexplotacion_temporal(indicadores):
    """Califica el contexto temporal de un pozo robustamente subexplotado."""
    insuficiente = {
        "estado": "Evidencia temporal insuficiente",
        "color": "#e87918",
        "confianza": "Insuficiente",
        "evidencias": [],
    }
    if indicadores is None or indicadores.empty:
        insuficiente["evidencias"] = [
            "No se cargaron tendencias históricas válidas."
        ]
        return insuficiente

    por_variable = {
        fila["Variable"]: fila
        for _, fila in indicadores.iterrows()
    }
    nombres = {
        "peso": "Peso de fluido promedio",
        "apertura": "Apertura de cargas de fondo",
        "llenado": "Llenado de bomba",
        "sumergencia": "Sumergencia relativa API",
        "eficiencia": "Eficiencia de carrera fondo/superficie",
    }
    filas = {
        clave: por_variable.get(nombre)
        for clave, nombre in nombres.items()
    }
    suficientes = [
        fila for fila in filas.values()
        if fila is not None
        and pd.notna(fila.get("Días_con_datos"))
        and float(fila["Días_con_datos"]) >= 8
    ]
    if (
        len(suficientes) < 3
        or filas["peso"] is None
        or filas["llenado"] is None
    ):
        faltantes = [
            nombre for clave, nombre in nombres.items()
            if filas[clave] is None
            or pd.isna(filas[clave].get("Días_con_datos"))
            or float(filas[clave]["Días_con_datos"]) < 8
        ]
        insuficiente["evidencias"] = [
            "Se requieren al menos 8 días válidos en tres variables críticas.",
            "Cobertura insuficiente: " + ", ".join(faltantes),
        ]
        return insuficiente

    def numero(fila, columna):
        if fila is None:
            return np.nan
        valor = fila.get(columna, np.nan)
        return float(valor) if pd.notna(valor) else np.nan

    peso = filas["peso"]
    apertura = filas["apertura"]
    llenado = filas["llenado"]
    sumergencia = filas["sumergencia"]
    eficiencia = filas["eficiencia"]

    peso_pend = numero(peso, "Pendiente_relativa_pct_día")
    peso_cambio = numero(peso, "Cambio_3d_vs_12d_pct")
    peso_vol = numero(peso, "Volatilidad_MAD_pct")
    apertura_pend = numero(apertura, "Pendiente_relativa_pct_día")
    apertura_cambio = numero(apertura, "Cambio_3d_vs_12d_pct")
    apertura_vol = numero(apertura, "Volatilidad_MAD_pct")
    llenado_ultimo = numero(llenado, "Último")
    llenado_pend = numero(llenado, "Pendiente_por_día")
    llenado_cambio = numero(llenado, "Cambio_3d_vs_12d")
    llenado_vol = numero(llenado, "Volatilidad_MAD_pct")
    sum_ultimo = numero(sumergencia, "Último")
    sum_pend = numero(sumergencia, "Pendiente_por_día")
    sum_cambio = numero(sumergencia, "Cambio_3d_vs_12d")
    eficiencia_pend = numero(eficiencia, "Pendiente_relativa_pct_día")
    eficiencia_cambio = numero(eficiencia, "Cambio_3d_vs_12d_pct")

    peso_sube = (
        np.isfinite(peso_pend) and peso_pend > 0.15
    ) or (
        np.isfinite(peso_cambio) and peso_cambio > 2.0
    )
    peso_estable = (
        np.isfinite(peso_pend)
        and np.isfinite(peso_cambio)
        and abs(peso_pend) <= 0.15
        and abs(peso_cambio) <= 2.0
    )
    apertura_sube = (
        np.isfinite(apertura_pend) and apertura_pend > 0.20
    ) or (
        np.isfinite(apertura_cambio) and apertura_cambio > 3.0
    )
    apertura_estable = (
        np.isfinite(apertura_pend)
        and np.isfinite(apertura_cambio)
        and abs(apertura_pend) <= 0.20
        and abs(apertura_cambio) <= 3.0
    )
    llenado_baja = (
        np.isfinite(llenado_pend) and llenado_pend < -0.15
    ) or (
        np.isfinite(llenado_cambio) and llenado_cambio < -2.0
    )
    llenado_alto = (
        np.isfinite(llenado_ultimo)
        and 90.0 <= llenado_ultimo <= 102.0
    )
    sumergencia_baja = (
        np.isfinite(sum_ultimo) and sum_ultimo <= 10.0
    )
    sumergencia_desciende = (
        np.isfinite(sum_pend) and sum_pend < -0.10
    ) or (
        np.isfinite(sum_cambio) and sum_cambio < -1.0
    )
    eficiencia_baja = (
        np.isfinite(eficiencia_pend) and eficiencia_pend < -0.03
    ) or (
        np.isfinite(eficiencia_cambio) and eficiencia_cambio < -0.5
    )
    volatil = (
        (np.isfinite(peso_vol) and peso_vol > 6.0)
        or (np.isfinite(apertura_vol) and apertura_vol > 8.0)
        or (np.isfinite(llenado_vol) and llenado_vol > 8.0)
    )

    evidencias = [
        (
            f"Peso de fluido: {peso_pend:+.2f}%/día; "
            f"cambio reciente {peso_cambio:+.2f}%."
        ),
        (
            f"Apertura de cargas: {apertura_pend:+.2f}%/día; "
            f"cambio reciente {apertura_cambio:+.2f}%."
        ),
        (
            f"Llenado API actual: {llenado_ultimo:.1f}%; "
            f"tendencia {llenado_pend:+.2f} pp/día."
        ),
    ]
    if np.isfinite(sum_ultimo):
        evidencias.append(
            f"Sumergencia relativa API actual: {sum_ultimo:.1f}%."
        )
    if np.isfinite(eficiencia_pend):
        evidencias.append(
            "Eficiencia de carrera fondo/superficie: "
            f"{eficiencia_pend:+.2f}%/día."
        )

    if volatil:
        estado, color, confianza = (
            "Comportamiento temporal volátil",
            "#9333ea",
            "Media",
        )
    elif (
        peso_sube
        and apertura_sube
        and (
            (np.isfinite(llenado_ultimo) and llenado_ultimo < 90.0)
            or llenado_baja
        )
    ):
        estado, color, confianza = (
            "Condición de extracción deteriorándose",
            "#dc2626",
            "Alta",
        )
    elif (
        sumergencia_baja
        and llenado_alto
        and not llenado_baja
        and (peso_estable or not peso_sube)
    ):
        estado, color, confianza = (
            "Aproximándose al equilibrio operativo",
            "#0284c7",
            "Alta",
        )
    elif (
        peso_sube
        and (
            apertura_sube
            or sumergencia_desciende
            or eficiencia_baja
        )
    ):
        estado, color, confianza = (
            "Oportunidad debilitándose",
            "#e87918",
            "Media",
        )
    elif (
        peso_estable
        and apertura_estable
        and not llenado_baja
    ):
        estado, color, confianza = (
            "Condición estable",
            "#16833b",
            "Alta",
        )
    else:
        estado, color, confianza = (
            "Evolución temporal no concluyente",
            "#64748b",
            "Media",
        )

    return {
        "estado": estado,
        "color": color,
        "confianza": confianza,
        "evidencias": evidencias,
    }


def analizar_falta_aporte_temporal(
    indicadores,
    diagnostico_robusto="",
):
    """
    Califica la evolución temporal de pozos con golpe de fluido o gas.

    No reemplaza el diagnóstico de las cartas. Determina si la restricción
    de aporte se agrava, permanece estable o muestra recuperación.
    """
    insuficiente = {
        "estado": "Evidencia temporal insuficiente",
        "color": "#e87918",
        "confianza": "Insuficiente",
        "evidencias": [],
    }
    if indicadores is None or indicadores.empty:
        insuficiente["evidencias"] = [
            "No se cargaron tendencias históricas válidas."
        ]
        return insuficiente

    por_variable = {
        fila["Variable"]: fila
        for _, fila in indicadores.iterrows()
    }
    nombres = {
        "peso": "Peso de fluido promedio",
        "apertura": "Apertura de cargas de fondo",
        "llenado": "Llenado de bomba",
        "sumergencia": "Sumergencia relativa API",
        "eficiencia": "Eficiencia de carrera fondo/superficie",
    }
    filas = {
        clave: por_variable.get(nombre)
        for clave, nombre in nombres.items()
    }
    suficientes = [
        fila for fila in filas.values()
        if fila is not None
        and pd.notna(fila.get("Días_con_datos"))
        and float(fila["Días_con_datos"]) >= 8
    ]
    if (
        len(suficientes) < 3
        or filas["llenado"] is None
        or filas["sumergencia"] is None
    ):
        faltantes = [
            nombre for clave, nombre in nombres.items()
            if filas[clave] is None
            or pd.isna(filas[clave].get("Días_con_datos"))
            or float(filas[clave]["Días_con_datos"]) < 8
        ]
        insuficiente["evidencias"] = [
            "Se requieren al menos 8 días válidos en tres variables críticas.",
            "Deben estar disponibles llenado y sumergencia relativa.",
            "Cobertura insuficiente: " + ", ".join(faltantes),
        ]
        return insuficiente

    def numero(fila, columna):
        if fila is None:
            return np.nan
        valor = fila.get(columna, np.nan)
        return float(valor) if pd.notna(valor) else np.nan

    peso = filas["peso"]
    apertura = filas["apertura"]
    llenado = filas["llenado"]
    sumergencia = filas["sumergencia"]
    eficiencia = filas["eficiencia"]

    peso_pend = numero(peso, "Pendiente_relativa_pct_día")
    peso_cambio = numero(peso, "Cambio_3d_vs_12d_pct")
    peso_vol = numero(peso, "Volatilidad_MAD_pct")
    apertura_pend = numero(apertura, "Pendiente_relativa_pct_día")
    apertura_cambio = numero(apertura, "Cambio_3d_vs_12d_pct")
    apertura_vol = numero(apertura, "Volatilidad_MAD_pct")
    llenado_ultimo = numero(llenado, "Último")
    llenado_pend = numero(llenado, "Pendiente_por_día")
    llenado_cambio = numero(llenado, "Cambio_3d_vs_12d")
    llenado_vol = numero(llenado, "Volatilidad_MAD_pct")
    sum_ultimo = numero(sumergencia, "Último")
    sum_pend = numero(sumergencia, "Pendiente_por_día")
    sum_cambio = numero(sumergencia, "Cambio_3d_vs_12d")
    eficiencia_pend = numero(eficiencia, "Pendiente_relativa_pct_día")
    eficiencia_cambio = numero(eficiencia, "Cambio_3d_vs_12d_pct")

    peso_sube = (
        np.isfinite(peso_pend) and peso_pend > 0.15
    ) or (
        np.isfinite(peso_cambio) and peso_cambio > 2.0
    )
    peso_baja = (
        np.isfinite(peso_pend) and peso_pend < -0.15
    ) or (
        np.isfinite(peso_cambio) and peso_cambio < -2.0
    )
    apertura_sube = (
        np.isfinite(apertura_pend) and apertura_pend > 0.20
    ) or (
        np.isfinite(apertura_cambio) and apertura_cambio > 3.0
    )
    apertura_baja = (
        np.isfinite(apertura_pend) and apertura_pend < -0.20
    ) or (
        np.isfinite(apertura_cambio) and apertura_cambio < -3.0
    )
    llenado_baja = (
        np.isfinite(llenado_pend) and llenado_pend < -0.15
    ) or (
        np.isfinite(llenado_cambio) and llenado_cambio < -2.0
    )
    llenado_sube = (
        np.isfinite(llenado_pend) and llenado_pend > 0.15
    ) or (
        np.isfinite(llenado_cambio) and llenado_cambio > 2.0
    )
    sumergencia_baja = (
        np.isfinite(sum_ultimo) and sum_ultimo <= 10.0
    )
    sumergencia_desciende = (
        np.isfinite(sum_pend) and sum_pend < -0.10
    ) or (
        np.isfinite(sum_cambio) and sum_cambio < -1.0
    )
    sumergencia_sube = (
        np.isfinite(sum_pend) and sum_pend > 0.10
    ) or (
        np.isfinite(sum_cambio) and sum_cambio > 1.0
    )
    eficiencia_baja = (
        np.isfinite(eficiencia_pend) and eficiencia_pend < -0.03
    ) or (
        np.isfinite(eficiencia_cambio) and eficiencia_cambio < -0.5
    )
    eficiencia_sube = (
        np.isfinite(eficiencia_pend) and eficiencia_pend > 0.03
    ) or (
        np.isfinite(eficiencia_cambio) and eficiencia_cambio > 0.5
    )
    llenado_restringido = (
        np.isfinite(llenado_ultimo) and llenado_ultimo < 90.0
    )
    llenado_critico = (
        np.isfinite(llenado_ultimo) and llenado_ultimo < 70.0
    )
    volatil = (
        (np.isfinite(peso_vol) and peso_vol > 6.0)
        or (np.isfinite(apertura_vol) and apertura_vol > 8.0)
        or (np.isfinite(llenado_vol) and llenado_vol > 8.0)
    )

    signos_deterioro = sum([
        llenado_baja,
        sumergencia_desciende,
        peso_sube,
        apertura_sube,
        eficiencia_baja,
    ])
    signos_recuperacion = sum([
        llenado_sube,
        sumergencia_sube,
        peso_baja,
        apertura_baja,
        eficiencia_sube,
    ])

    evidencias = [
        (
            f"Llenado API actual: {llenado_ultimo:.1f}%; "
            f"tendencia {llenado_pend:+.2f} pp/día."
        ),
        (
            f"Sumergencia relativa API actual: {sum_ultimo:.1f}%; "
            f"tendencia {sum_pend:+.2f} pp/día."
        ),
    ]
    if np.isfinite(peso_pend):
        evidencias.append(
            f"Peso de fluido: {peso_pend:+.2f}%/día; "
            f"cambio reciente {peso_cambio:+.2f}%."
        )
    if np.isfinite(apertura_pend):
        evidencias.append(
            f"Apertura de cargas: {apertura_pend:+.2f}%/día; "
            f"cambio reciente {apertura_cambio:+.2f}%."
        )
    if np.isfinite(eficiencia_pend):
        evidencias.append(
            "Eficiencia de carrera fondo/superficie: "
            f"{eficiencia_pend:+.2f}%/día."
        )

    # Primero se determina si existe una dirección temporal coherente.
    # La volatilidad se aplica después como calificativo secundario para
    # evitar que oculte un deterioro o una recuperación bien sustentados.
    if (
        signos_deterioro >= 3
        and llenado_restringido
        and (
            sumergencia_baja
            or llenado_critico
            or sumergencia_desciende
        )
    ):
        estado, color, confianza = (
            "Restricción de aporte agravándose",
            "#dc2626",
            "Alta",
        )
    elif (
        signos_recuperacion >= 3
        and llenado_sube
        and sumergencia_sube
    ):
        estado, color, confianza = (
            "Evidencia temporal de recuperación",
            "#16833b",
            "Alta",
        )
    elif (
        llenado_restringido
        and signos_deterioro <= 1
        and signos_recuperacion <= 1
    ):
        estado, color, confianza = (
            "Restricción de aporte estable",
            "#e87918",
            "Media",
        )
    elif signos_deterioro >= 2:
        estado, color, confianza = (
            "Posible deterioro de la admisión",
            "#f97316",
            "Media",
        )
    elif signos_recuperacion >= 2:
        estado, color, confianza = (
            "Posible recuperación de la admisión",
            "#0284c7",
            "Media",
        )
    elif volatil:
        estado, color, confianza = (
            "Comportamiento temporal volátil",
            "#9333ea",
            "Media",
        )
    else:
        estado, color, confianza = (
            "Evolución temporal no concluyente",
            "#64748b",
            "Media",
        )

    if volatil and estado != "Comportamiento temporal volátil":
        estado += " — con comportamiento volátil"
        evidencias.append(
            "La dispersión es relevante, pero no invalida la dirección "
            "temporal predominante."
        )

    if diagnostico_robusto:
        evidencias.insert(
            0,
            f"Diagnóstico robusto de cartas: {diagnostico_robusto}.",
        )
    return {
        "estado": estado,
        "color": color,
        "confianza": confianza,
        "evidencias": evidencias,
    }


def analizar_bloqueo_temporal(tabla):
    """
    Analiza persistencia, inicio y alternancia de una condición sin trabajo.

    La función está pensada como complemento de un diagnóstico robusto
    actual de ``Posible sin trabajo de bomba``. Compara cada pozo contra
    sus propios niveles altos de los últimos 30 días y no reemplaza el
    diagnóstico geométrico de las cartas.
    """
    insuficiente = {
        "estado": "Evidencia temporal insuficiente",
        "color": "#e87918",
        "confianza": "Insuficiente",
        "evidencias": [],
        "fecha_inicio": None,
        "transiciones_15d": 0,
        "dias_bloqueado_15d": 0,
        "dias_validos_15d": 0,
        "porcentaje_bloqueado_15d": np.nan,
    }
    if tabla is None or tabla.empty or "Fecha" not in tabla:
        insuficiente["evidencias"] = [
            "No se cargaron tendencias históricas válidas."
        ]
        return insuficiente

    trabajo = tabla.copy()
    trabajo["Fecha"] = pd.to_datetime(
        trabajo["Fecha"],
        errors="coerce",
    )
    trabajo = trabajo.dropna(subset=["Fecha"])
    if trabajo.empty:
        insuficiente["evidencias"] = [
            "No existen fechas válidas en el histórico."
        ]
        return insuficiente

    columnas_numericas = [
        "Llenado_Bomba_API_pct",
        "Peso_Fluido_Promedio_lbf",
        "Carga_Maxima_Fondo_lbf",
        "Carga_Minima_Fondo_lbf",
        "Carrera_Fondo_Total_pulg",
        "Carrera_Superficie_pulg",
    ]
    for columna in columnas_numericas:
        if columna in trabajo:
            trabajo[columna] = pd.to_numeric(
                trabajo[columna],
                errors="coerce",
            )

    if {
        "Carga_Maxima_Fondo_lbf",
        "Carga_Minima_Fondo_lbf",
    }.issubset(trabajo.columns):
        trabajo["Apertura_Cargas_lbf"] = (
            trabajo["Carga_Maxima_Fondo_lbf"]
            - trabajo["Carga_Minima_Fondo_lbf"]
        )

    if {
        "Carrera_Fondo_Total_pulg",
        "Carrera_Superficie_pulg",
    }.issubset(trabajo.columns):
        superficie = trabajo["Carrera_Superficie_pulg"]
        trabajo["Eficiencia_Carrera_pct"] = np.where(
            superficie.abs() > 1e-9,
            100.0
            * trabajo["Carrera_Fondo_Total_pulg"]
            / superficie,
            np.nan,
        )

    variables = [
        columna
        for columna in [
            "Llenado_Bomba_API_pct",
            "Peso_Fluido_Promedio_lbf",
            "Apertura_Cargas_lbf",
            "Eficiencia_Carrera_pct",
        ]
        if columna in trabajo
    ]
    if len(variables) < 3:
        insuficiente["evidencias"] = [
            "Se necesitan al menos tres variables entre llenado, "
            "peso de fluido, apertura de cargas y eficiencia de carrera."
        ]
        return insuficiente

    trabajo["Dia"] = trabajo["Fecha"].dt.floor("D")
    diaria = (
        trabajo[["Dia", *variables]]
        .groupby("Dia", as_index=False)
        .median(numeric_only=True)
        .sort_values("Dia")
    )
    fecha_final = diaria["Dia"].max()
    diaria = diaria.loc[
        diaria["Dia"]
        >= fecha_final - pd.Timedelta(days=29)
    ].copy()
    if len(diaria) < 8:
        insuficiente["evidencias"] = [
            "Se requieren al menos 8 días con datos para analizar el bloqueo."
        ]
        return insuficiente

    # El percentil alto representa la condición de mayor trabajo observada
    # en el propio pozo. Es más robusto que aplicar una carga absoluta común.
    referencias = {}
    for columna in [
        "Llenado_Bomba_API_pct",
        "Peso_Fluido_Promedio_lbf",
        "Apertura_Cargas_lbf",
    ]:
        referencias[columna] = (
            float(diaria[columna].quantile(0.80))
            if columna in diaria and diaria[columna].notna().any()
            else np.nan
        )

    def puntos_bajos(valor, referencia, limite_absoluto=None):
        if not np.isfinite(valor):
            return 0
        if limite_absoluto is not None and valor <= limite_absoluto:
            return 2
        if not np.isfinite(referencia) or referencia <= 1e-9:
            return 0
        relacion = valor / referencia
        if relacion <= 0.50:
            return 2
        if relacion <= 0.70:
            return 1
        return 0

    puntajes = []
    variables_validas = []
    for _, fila in diaria.iterrows():
        componentes = []
        if "Llenado_Bomba_API_pct" in diaria:
            llenado_dia = fila["Llenado_Bomba_API_pct"]
            componentes.append(
                3
                if np.isfinite(llenado_dia) and llenado_dia <= 25.0
                else puntos_bajos(
                    llenado_dia,
                    referencias["Llenado_Bomba_API_pct"],
                )
            )
        if "Peso_Fluido_Promedio_lbf" in diaria:
            componentes.append(
                puntos_bajos(
                    fila["Peso_Fluido_Promedio_lbf"],
                    referencias["Peso_Fluido_Promedio_lbf"],
                )
            )
        if "Apertura_Cargas_lbf" in diaria:
            componentes.append(
                puntos_bajos(
                    fila["Apertura_Cargas_lbf"],
                    referencias["Apertura_Cargas_lbf"],
                )
            )

        validas = sum(
            pd.notna(fila.get(columna, np.nan))
            for columna in [
                "Llenado_Bomba_API_pct",
                "Peso_Fluido_Promedio_lbf",
                "Apertura_Cargas_lbf",
            ]
            if columna in diaria
        )
        eficiencia = fila.get("Eficiencia_Carrera_pct", np.nan)
        punto_eficiencia = int(
            np.isfinite(eficiencia)
            and 90.0 <= eficiencia <= 115.0
        )
        puntajes.append(
            float(sum(componentes) + punto_eficiencia)
            if validas >= 2
            else np.nan
        )
        variables_validas.append(validas)

    diaria["Score_Bloqueo"] = puntajes
    diaria["Variables_Validas"] = variables_validas
    diaria["Bloqueado"] = np.where(
        diaria["Score_Bloqueo"].notna(),
        diaria["Score_Bloqueo"] >= 4.0,
        np.nan,
    )
    valida = diaria.dropna(subset=["Bloqueado"]).copy()
    if len(valida) < 8:
        insuficiente["evidencias"] = [
            "Menos de 8 días poseen suficientes variables para puntuar."
        ]
        return insuficiente

    # Elimina inversiones aisladas de un solo día para no interpretar ruido
    # como un bloqueo o desbloqueo operativo.
    estados = valida["Bloqueado"].astype(bool).to_numpy()
    estados_filtrados = estados.copy()
    for indice in range(1, len(estados) - 1):
        if estados[indice - 1] == estados[indice + 1]:
            estados_filtrados[indice] = estados[indice - 1]
    valida["Bloqueado_Filtrado"] = estados_filtrados

    ventana_15 = valida.loc[
        valida["Dia"]
        >= fecha_final - pd.Timedelta(days=14)
    ].copy()
    dias_validos = int(len(ventana_15))
    dias_bloqueado = int(ventana_15["Bloqueado_Filtrado"].sum())
    fraccion_bloqueado = (
        dias_bloqueado / dias_validos
        if dias_validos
        else np.nan
    )
    estados_15 = ventana_15["Bloqueado_Filtrado"].to_numpy(dtype=bool)
    transiciones = int(
        np.sum(estados_15[1:] != estados_15[:-1])
        if len(estados_15) > 1
        else 0
    )
    actual_bloqueado = bool(estados_filtrados[-1])

    # Inicio de la corrida bloqueada vigente.
    fecha_inicio = None
    if actual_bloqueado:
        inicio = len(estados_filtrados) - 1
        while inicio > 0 and estados_filtrados[inicio - 1]:
            inicio -= 1
        fecha_inicio = pd.Timestamp(valida.iloc[inicio]["Dia"])

    # Para reconocer desbloqueo exigimos dos estados recientes sin bloqueo.
    desbloqueo_reciente = bool(
        len(estados_filtrados) >= 3
        and not estados_filtrados[-1]
        and not estados_filtrados[-2]
        and np.any(estados_filtrados[:-2])
    )
    alternante = bool(
        dias_validos >= 8
        and transiciones >= 3
        and dias_bloqueado >= 3
        and (dias_validos - dias_bloqueado) >= 3
    )
    persistente = bool(
        dias_validos >= 10
        and fraccion_bloqueado >= 0.80
    )
    bloqueo_reciente = bool(
        actual_bloqueado
        and fecha_inicio is not None
        and fecha_inicio
        >= fecha_final - pd.Timedelta(days=6)
        and np.any(
            valida.loc[
                valida["Dia"] < fecha_inicio,
                "Bloqueado_Filtrado",
            ].eq(False)
        )
    )

    if desbloqueo_reciente:
        estado, color, confianza = (
            "Posible desbloqueo reciente",
            "#16833b",
            "Media",
        )
    elif alternante:
        estado, color, confianza = (
            "Bloqueo intermitente",
            "#9333ea",
            "Alta",
        )
    elif persistente:
        estado, color, confianza = (
            "Bloqueo persistente en la ventana de 15 días",
            "#dc2626",
            "Alta",
        )
    elif bloqueo_reciente:
        estado, color, confianza = (
            "Bloqueo reciente detectado",
            "#e87918",
            "Alta",
        )
    elif actual_bloqueado:
        estado, color, confianza = (
            "Condición bloqueada sin inicio concluyente",
            "#f97316",
            "Media",
        )
    else:
        estado, color, confianza = (
            "Firma temporal de bloqueo no confirmada",
            "#64748b",
            "Media",
        )

    ultimo = valida.iloc[-1]
    evidencias = [
        (
            f"Días compatibles con bloqueo: {dias_bloqueado} de "
            f"{dias_validos} ({100 * fraccion_bloqueado:.0f}%)."
        ),
        f"Transiciones bloqueado/desbloqueado en 15 días: {transiciones}.",
        f"Score de bloqueo más reciente: {ultimo['Score_Bloqueo']:.1f}/8.",
    ]
    if fecha_inicio is not None:
        evidencias.append(
            "Inicio estimado de la condición vigente: "
            f"{fecha_inicio.strftime('%d/%m/%Y')}."
        )
    if "Llenado_Bomba_API_pct" in ultimo and pd.notna(
        ultimo["Llenado_Bomba_API_pct"]
    ):
        evidencias.append(
            "Llenado API diario más reciente: "
            f"{ultimo['Llenado_Bomba_API_pct']:.1f}%."
        )
    if "Eficiencia_Carrera_pct" in ultimo and pd.notna(
        ultimo["Eficiencia_Carrera_pct"]
    ):
        evidencias.append(
            "Carrera fondo/superficie más reciente: "
            f"{ultimo['Eficiencia_Carrera_pct']:.1f}%."
        )

    return {
        "estado": estado,
        "color": color,
        "confianza": confianza,
        "evidencias": evidencias,
        "fecha_inicio": fecha_inicio,
        "transiciones_15d": transiciones,
        "dias_bloqueado_15d": dias_bloqueado,
        "dias_validos_15d": dias_validos,
        "porcentaje_bloqueado_15d": (
            100.0 * fraccion_bloqueado
        ),
    }


def a_array(valor):
    """Convierte listas del JSON o listas serializadas a arrays float."""
    if valor is None:
        return np.array([], dtype=float)
    if isinstance(valor, str):
        if not valor.strip():
            return np.array([], dtype=float)
        valor = json.loads(valor)
    return np.asarray(valor, dtype=float)


def carta_valida(fila):
    fp = a_array(fila["Fondo_Posiciones"])
    fc = a_array(fila["Fondo_Cargas"])
    sp = a_array(fila["Superficie_Posiciones"])
    sc = a_array(fila["Superficie_Cargas"])
    return (
        len(fp) == len(fc) == 80
        and len(sp) == len(sc) == 80
        and np.isfinite(fp).all()
        and np.isfinite(fc).all()
        and np.isfinite(sp).all()
        and np.isfinite(sc).all()
    )


def cargar_respuesta_json(origen: Any):
    if isinstance(origen, (str, Path)):
        with open(origen, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    if isinstance(origen, bytes):
        return json.loads(origen.decode("utf-8-sig"))
    if hasattr(origen, "read"):
        contenido = origen.read()
        if isinstance(contenido, bytes):
            contenido = contenido.decode("utf-8-sig")
        return json.loads(contenido)
    if isinstance(origen, (dict, list)):
        return origen
    raise TypeError("Origen JSON no reconocido")


def preparar_datos(origen):
    respuesta = cargar_respuesta_json(origen)
    if isinstance(respuesta, dict) and "items" in respuesta:
        items = respuesta["items"]
        total_declarado = respuesta.get("totalRecords")
    else:
        items = respuesta
        total_declarado = None

    if not isinstance(items, list) or not items:
        raise ValueError("El JSON no contiene una colección de cartas válida")

    datos = pd.json_normalize(items).rename(columns={
        "IdCarta": "CartaId",
        "PosicionesFondo": "Fondo_Posiciones",
        "CargasFondo": "Fondo_Cargas",
        "PosicionesSuperficie": "Superficie_Posiciones",
        "CargasSuperficie": "Superficie_Cargas",
    })

    if "PorcentajeTorqueReductorExistente" not in datos.columns:
        datos["PorcentajeTorqueReductorExistente"] = np.nan
    if "PorcentajeCargaEstructural" not in datos.columns:
        datos["PorcentajeCargaEstructural"] = np.nan

    datos["Torque_Reductor_pct"] = pd.to_numeric(
        datos["PorcentajeTorqueReductorExistente"],
        errors="coerce",
    )
    datos["Carga_Estructural_pct"] = pd.to_numeric(
        datos["PorcentajeCargaEstructural"],
        errors="coerce",
    )

    obligatorias = [
        "CartaId", "Pozo", "Fecha",
        "Fondo_Posiciones", "Fondo_Cargas",
        "Superficie_Posiciones", "Superficie_Cargas",
        "ProfundidadBomba", "DiametroPistonBomba", "GPM",
    ]
    faltantes = [c for c in obligatorias if c not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")

    datos["Carta_Valida"] = datos.apply(carta_valida, axis=1)
    invalidas = datos.loc[~datos["Carta_Valida"]].copy()
    muestra = datos.loc[datos["Carta_Valida"]].copy().reset_index(drop=True)
    return datos, muestra, invalidas, total_declarado


def procesar_json(
    origen,
    silencioso=True,
    presion_tubing_kg_cm2=10.0,
    presion_casing_kg_cm2=10.0,
    gravedad_especifica_sam=0.994,
    gradiente_sam_psi_m=None,
):
    datos, muestra, invalidas, total_declarado = preparar_datos(origen)

    # Los display del notebook son auditorías, no forman parte de la salida.
    def display(*_args, **_kwargs):
        return None


    # ===== CELDA ORIGINAL 4 =====
    import numpy as np
    import pandas as pd


    def tramo_circular(inicio, fin, n):
        """
        Devuelve los índices recorridos desde inicio hasta fin,
        permitiendo que la secuencia continúe desde el final al principio.
        """
        if inicio <= fin:
            return np.arange(inicio, fin + 1)

        return np.concatenate([
            np.arange(inicio, n),
            np.arange(0, fin + 1),
        ])


    def separar_carreras(posicion, carga):
        """
        Separa la curva cerrada en:
          - carrera ascendente: posición mínima -> posición máxima
          - carrera descendente: posición máxima -> posición mínima

        Respeta el orden original de los puntos.
        """
        posicion = np.asarray(posicion, dtype=float)
        carga = np.asarray(carga, dtype=float)

        validos = np.isfinite(posicion) & np.isfinite(carga)
        posicion = posicion[validos]
        carga = carga[validos]

        if len(posicion) < 10:
            raise ValueError("La carta tiene muy pocos puntos válidos.")

        indice_min = int(np.argmin(posicion))
        indice_max = int(np.argmax(posicion))

        idx_asc = tramo_circular(indice_min, indice_max, len(posicion))
        idx_desc = tramo_circular(indice_max, indice_min, len(posicion))

        ascendente = {
            "posicion": posicion[idx_asc],
            "carga": carga[idx_asc],
        }

        descendente = {
            "posicion": posicion[idx_desc],
            "carga": carga[idx_desc],
        }

        return ascendente, descendente


    # ===== CELDA ORIGINAL 5 =====
    def tramo_consecutivo_mas_largo(mascara):
        """Devuelve los índices del tramo consecutivo True más largo."""
        mascara = np.asarray(mascara, dtype=bool)

        mejor_inicio = None
        mejor_fin = None
        inicio_actual = None

        for i, valor in enumerate(mascara):
            if valor and inicio_actual is None:
                inicio_actual = i

            termina = inicio_actual is not None and (
                not valor or i == len(mascara) - 1
            )

            if termina:
                fin_actual = i if valor else i - 1

                if (
                    mejor_inicio is None
                    or fin_actual - inicio_actual > mejor_fin - mejor_inicio
                ):
                    mejor_inicio = inicio_actual
                    mejor_fin = fin_actual

                inicio_actual = None

        if mejor_inicio is None:
            return np.array([], dtype=int)

        return np.arange(mejor_inicio, mejor_fin + 1)


    def estimar_linea_horizontal(
        posicion,
        carga,
        recorte_extremos=0.10,
        ventana=7,
        fraccion_pendiente=0.50,
        tolerancia_mad=2.5,
        minimo_puntos=5,
    ):
        """
        Identifica un tramo aproximadamente horizontal.

        La carga representativa es la mediana robusta del tramo elegido.
        """
        x = np.asarray(posicion, dtype=float)
        y = np.asarray(carga, dtype=float)

        if len(x) < minimo_puntos:
            raise ValueError("No hay suficientes puntos para estimar una línea.")

        # Suavizado robusto para evitar que un pico aislado domine la pendiente.
        y_suave = (
            pd.Series(y)
            .rolling(
                window=ventana,
                center=True,
                min_periods=max(3, ventana // 2),
            )
            .median()
            .to_numpy()
        )

        rango_x = np.nanmax(x) - np.nanmin(x)

        if rango_x <= 0:
            raise ValueError("La posición no presenta recorrido.")

        # Descartar los extremos del recorrido.
        x_normalizada = (x - np.nanmin(x)) / rango_x

        zona_central = (
            (x_normalizada >= recorte_extremos)
            & (x_normalizada <= 1 - recorte_extremos)
        )

        # Pendiente local aproximada dCarga/dPosición.
        dx = np.gradient(x)
        dy = np.gradient(y_suave)

        pendiente = np.full(len(x), np.nan)

        movimiento_valido = np.abs(dx) > max(rango_x * 1e-6, 1e-9)
        pendiente[movimiento_valido] = np.abs(
            dy[movimiento_valido] / dx[movimiento_valido]
        )

        pendientes_centrales = pendiente[
            zona_central & np.isfinite(pendiente)
        ]

        if len(pendientes_centrales) == 0:
            raise ValueError("No se pudo calcular la pendiente local.")

        # Conservamos la mitad de los puntos con menor pendiente.
        limite_pendiente = np.quantile(
            pendientes_centrales,
            fraccion_pendiente,
        )

        baja_pendiente = (
            zona_central
            & np.isfinite(pendiente)
            & (pendiente <= limite_pendiente)
        )

        # Primera estimación robusta del nivel de carga.
        nivel_inicial = np.median(y_suave[baja_pendiente])

        desviaciones = np.abs(y_suave[baja_pendiente] - nivel_inicial)
        mad = np.median(desviaciones)

        # Escala robusta equivalente aproximadamente al desvío estándar.
        escala_robusta = 1.4826 * mad

        # Evita una tolerancia igual a cero en curvas muy planas.
        rango_carga = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
        tolerancia_minima = max(0.03 * rango_carga, 1e-9)

        tolerancia = max(
            tolerancia_mad * escala_robusta,
            tolerancia_minima,
        )

        candidatos = (
            baja_pendiente
            & (np.abs(y_suave - nivel_inicial) <= tolerancia)
        )

        indices_tramo = tramo_consecutivo_mas_largo(candidatos)

        # Respaldo si el filtro fue demasiado estricto.
        if len(indices_tramo) < minimo_puntos:
            indices_tramo = tramo_consecutivo_mas_largo(baja_pendiente)

        if len(indices_tramo) < minimo_puntos:
            # Último respaldo: usar los puntos centrales de baja pendiente,
            # aunque no formen un tramo largo.
            indices_tramo = np.flatnonzero(baja_pendiente)

        if len(indices_tramo) == 0:
            raise ValueError("No se encontró un tramo horizontal adecuado.")

        nivel_carga = float(np.median(y[indices_tramo]))

        return {
            "carga_representativa": nivel_carga,
            "indices": indices_tramo,
            "posicion_inicio": float(np.min(x[indices_tramo])),
            "posicion_fin": float(np.max(x[indices_tramo])),
            "cantidad_puntos": len(indices_tramo),
            "limite_pendiente": float(limite_pendiente),
            "tolerancia_carga": float(tolerancia),
        }


    # ===== CELDA ORIGINAL 6 =====
    def estimar_descendente_menor_carga(
        posicion,
        carga,
        recorte_extremos=0.08,
        ventana_suavizado=5,
        fraccion_ventana=0.16,
        cambio_relativo_max=0.18,
        rugosidad_relativa_max=0.10,
        minimo_puntos=5,
    ):
        """
        Busca en la carrera descendente el tramo pseudo-horizontal
        de menor carga.

        Prioridades:
          1. El tramo no debe ser una transición vertical.
          2. Debe tener una longitud mínima.
          3. Entre los tramos aceptables, se elige el de menor carga.
        """
        x = np.asarray(posicion, dtype=float)
        y = np.asarray(carga, dtype=float)

        validos = np.isfinite(x) & np.isfinite(y)
        x = x[validos]
        y = y[validos]

        if len(x) < minimo_puntos:
            raise ValueError("No hay suficientes puntos en la carrera descendente.")

        # Suavizado para reducir el efecto de oscilaciones puntuales.
        y_suave = (
            pd.Series(y)
            .rolling(
                window=ventana_suavizado,
                center=True,
                min_periods=2,
            )
            .median()
            .bfill()
            .ffill()
            .to_numpy()
        )

        rango_x = np.ptp(x)
        rango_y = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)

        if rango_x <= 0:
            raise ValueError("La posición no presenta recorrido.")

        rango_y = max(rango_y, 1e-9)

        # Descartar los extremos de posición, donde ocurren los cambios de carrera.
        x_normalizada = (x - np.min(x)) / rango_x

        zona_central = (
            (x_normalizada >= recorte_extremos)
            & (x_normalizada <= 1 - recorte_extremos)
        )

        indices_centrales = np.flatnonzero(zona_central)

        if len(indices_centrales) < minimo_puntos:
            indices_centrales = np.arange(len(x))

        # Longitud de las ventanas examinadas.
        longitud_ventana = max(
            minimo_puntos,
            int(round(fraccion_ventana * len(indices_centrales))),
        )

        longitud_ventana = min(
            longitud_ventana,
            len(indices_centrales),
        )

        candidatos = []

        for inicio in range(0, len(x) - longitud_ventana + 1):
            indices = np.arange(inicio, inicio + longitud_ventana)

            # La ventana debe encontrarse dentro de la zona central.
            if not np.all(zona_central[indices]):
                continue

            x_tramo = x[indices]
            y_tramo = y_suave[indices]

            amplitud_x = np.ptp(x_tramo)

            # Evitar ventanas concentradas prácticamente en una posición.
            if amplitud_x < 0.06 * rango_x:
                continue

            # Recta local utilizada únicamente para medir la inclinación.
            pendiente, intercepto = np.polyfit(
                x_tramo,
                y_tramo,
                deg=1,
            )

            tendencia = pendiente * x_tramo + intercepto
            residuos = y_tramo - tendencia

            # Cambio de carga del extremo inicial al final del tramo,
            # expresado respecto del rango total de carga.
            cambio_relativo = (
                abs(pendiente) * amplitud_x / rango_y
            )

            # Variabilidad alrededor de la tendencia local.
            rugosidad = np.median(
                np.abs(residuos - np.median(residuos))
            )
            rugosidad_relativa = rugosidad / rango_y

            es_pseudo_horizontal = (
                cambio_relativo <= cambio_relativo_max
                and rugosidad_relativa <= rugosidad_relativa_max
            )

            if es_pseudo_horizontal:
                progreso_medio = float(
                    np.mean(indices) / max(len(x) - 1, 1)
                )

                persistencia = float(amplitud_x / rango_x)

                candidatos.append({
                    "indices": indices,
                    "carga_mediana": float(np.median(y[indices])),
                    "cambio_relativo": float(cambio_relativo),
                    "rugosidad_relativa": float(rugosidad_relativa),
                    "pendiente_local": float(pendiente),

                    # 0 = comienzo de la carrera descendente
                    # 1 = final de la carrera descendente
                    "progreso_medio": progreso_medio,

                    # Fracción del recorrido de posición cubierta por el tramo
                    "persistencia": persistencia,
                })

        # Si el filtro inicial fue demasiado estricto, se permite algo más
        # de inclinación, pero todavía se rechazan las transiciones verticales.
        if not candidatos:
            return estimar_descendente_menor_carga(
                posicion=x,
                carga=y,
                recorte_extremos=recorte_extremos,
                ventana_suavizado=ventana_suavizado,
                fraccion_ventana=fraccion_ventana,
                cambio_relativo_max=cambio_relativo_max * 1.5,
                rugosidad_relativa_max=rugosidad_relativa_max * 1.5,
                minimo_puntos=minimo_puntos,
            )

        # Criterio principal: menor carga.
        # En caso de cargas similares, se prefiere el tramo más horizontal.

        # Normalización de las cargas de los candidatos.
        # Priorizamos candidatos ubicados después del 55 % de la
        # carrera descendente. Esto evita interpretar una depresión
        # transitoria inicial como carga representativa.
        candidatos_tardios = [
            c for c in candidatos
            if c["progreso_medio"] >= 0.55
        ]

        # Si no encontramos ninguno, ampliamos la búsqueda al 45 % final.
        if not candidatos_tardios:
            candidatos_tardios = [
                c for c in candidatos
                if c["progreso_medio"] >= 0.45
            ]

        # Solo si tampoco existen candidatos allí, usamos todos.
        if candidatos_tardios:
            candidatos_evaluados = candidatos_tardios
        else:
            candidatos_evaluados = candidatos


        # Normalizar las cargas dentro del grupo evaluado.
        cargas_candidatas = np.array([
            c["carga_mediana"]
            for c in candidatos_evaluados
        ])

        carga_min = np.min(cargas_candidatas)
        carga_max = np.max(cargas_candidatas)
        rango_candidatos = max(carga_max - carga_min, 1e-9)


        for candidato in candidatos_evaluados:
            carga_normalizada = (
                candidato["carga_mediana"] - carga_min
            ) / rango_candidatos

            penalizacion_temporal = (
                1 - candidato["progreso_medio"]
            )

            persistencia_normalizada = min(
                candidato["persistencia"] / 0.25,
                1.0,
            )
            penalizacion_persistencia = (
                1 - persistencia_normalizada
            )

            penalizacion_pendiente = min(
                candidato["cambio_relativo"]
                / max(cambio_relativo_max, 1e-9),
                1.0,
            )

            penalizacion_rugosidad = min(
                candidato["rugosidad_relativa"]
                / max(rugosidad_relativa_max, 1e-9),
                1.0,
            )

            candidato["puntaje"] = (
                0.15 * carga_normalizada
                + 0.40 * penalizacion_temporal
                + 0.30 * penalizacion_persistencia
                + 0.10 * penalizacion_pendiente
                + 0.05 * penalizacion_rugosidad
            )


        mejor = min(
            candidatos_evaluados,
            key=lambda c: c["puntaje"],
        )

        indices = mejor["indices"]
        carga_representativa = float(np.median(y[indices]))

        return {
            "carga_representativa": carga_representativa,
            "indices": indices,
            "posicion_inicio": float(np.min(x[indices])),
            "posicion_fin": float(np.max(x[indices])),
            "cantidad_puntos": len(indices),
            "pendiente_local": mejor["pendiente_local"],
            "cambio_relativo": mejor["cambio_relativo"],
            "rugosidad_relativa": mejor["rugosidad_relativa"],
            "progreso_medio": mejor["progreso_medio"],
            "persistencia": mejor["persistencia"],
            "puntaje": mejor["puntaje"],
        }


    # ===== CELDA ORIGINAL 8 =====
    def area_poligono(x, y):
        """
        Calcula el área de un polígono mediante la fórmula shoelace.
        Los puntos deben estar ordenados siguiendo el contorno.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        validos = np.isfinite(x) & np.isfinite(y)
        x = x[validos]
        y = y[validos]

        if len(x) < 3:
            return np.nan

        area = 0.5 * abs(
            np.sum(
                x * np.roll(y, -1)
                - y * np.roll(x, -1)
            )
        )

        return float(area)


    def extremos_en_banda_horizontal(
        posicion,
        carga,
        nivel,
        tolerancia,
        minimo_puntos=2,
    ):
        """
        Busca los puntos reales próximos a una horizontal y devuelve
        las posiciones más izquierda y más derecha entre esos puntos.

        Si la banda inicial contiene pocos puntos, la amplía
        progresivamente.
        """
        x = np.asarray(posicion, dtype=float)
        y = np.asarray(carga, dtype=float)

        validos = np.isfinite(x) & np.isfinite(y)
        x = x[validos]
        y = y[validos]

        if len(x) == 0:
            raise ValueError(
                "La carrera no contiene puntos válidos."
            )

        multiplicadores = [
            1.0,
            1.5,
            2.0,
            3.0,
        ]

        indices = np.array([], dtype=int)
        multiplicador_usado = np.nan

        for multiplicador in multiplicadores:
            indices = np.flatnonzero(
                np.abs(y - nivel)
                <= tolerancia * multiplicador
            )

            if len(indices) >= minimo_puntos:
                multiplicador_usado = multiplicador
                break

        # Respaldo: tomar los puntos reales más próximos
        # si no hubo suficientes dentro de la banda.
        if len(indices) < minimo_puntos:
            cantidad = min(
                max(
                    minimo_puntos,
                    int(np.ceil(0.15 * len(x))),
                ),
                len(x),
            )

            indices = np.argsort(
                np.abs(y - nivel)
            )[:cantidad]

            multiplicador_usado = np.nan

        return {
            "x_izquierdo": float(
                np.min(x[indices])
            ),
            "x_derecho": float(
                np.max(x[indices])
            ),
            "indices": indices,
            "multiplicador_usado":
                multiplicador_usado,
            "tolerancia_real": float(
                np.max(
                    np.abs(y[indices] - nivel)
                )
            ),
        }


    def construir_carta_ideal(
        ascendente,
        descendente,
        linea_asc,
        linea_desc,
        fraccion_banda=0.35,
    ):
        """
        Construye la carta ideal.

        El lateral izquierdo utiliza puntos reales próximos a las
        horizontales.

        El lateral derecho se proyecta con la misma inclinación,
        pero nunca puede terminar antes del extremo real de la
        carrera descendente.
        """
        x_asc = np.asarray(
            ascendente["posicion"],
            dtype=float,
        )

        y_asc = np.asarray(
            ascendente["carga"],
            dtype=float,
        )

        x_desc = np.asarray(
            descendente["posicion"],
            dtype=float,
        )

        y_desc = np.asarray(
            descendente["carga"],
            dtype=float,
        )

        carga_asc = float(
            linea_asc["carga_representativa"]
        )

        carga_desc = float(
            linea_desc["carga_representativa"]
        )

        diferencia_cargas = abs(
            carga_asc - carga_desc
        )

        tolerancia = max(
            fraccion_banda
            * diferencia_cargas,
            1e-9,
        )

        # Extremos próximos a la horizontal superior.
        extremos_asc = extremos_en_banda_horizontal(
            posicion=x_asc,
            carga=y_asc,
            nivel=carga_asc,
            tolerancia=tolerancia,
        )

        # Extremos próximos a la horizontal inferior.
        extremos_desc = extremos_en_banda_horizontal(
            posicion=x_desc,
            carga=y_desc,
            nivel=carga_desc,
            tolerancia=tolerancia,
        )

        # Vértice superior izquierdo.
        x_superior_izquierdo = float(
            extremos_asc["x_izquierdo"]
        )

        # Vértice inferior izquierdo.
        x_inferior_izquierdo = float(
            extremos_desc["x_izquierdo"]
        )

        # Vértice superior derecho.
        x_superior_derecho = float(
            extremos_asc["x_derecho"]
        )

        # Inclinación del lateral izquierdo.
        desplazamiento_lateral = (
            x_inferior_izquierdo
            - x_superior_izquierdo
        )

        # Posición inferior derecha según el paralelismo.
        x_inferior_derecho_paralelo = (
            x_superior_derecho
            + desplazamiento_lateral
        )

        # Primer punto real de la carrera descendente.
        x_inferior_derecho_real = float(
            np.max(x_desc)
        )

        # Mantener la proyección, pero no permitir que recorte
        # la carrera descendente real.
        x_inferior_derecho = max(
            x_inferior_derecho_paralelo,
            x_inferior_derecho_real,
        )

        vertices = np.array([
            # Superior izquierdo
            [
                x_superior_izquierdo,
                carga_asc,
            ],

            # Superior derecho
            [
                x_superior_derecho,
                carga_asc,
            ],

            # Inferior derecho
            [
                x_inferior_derecho,
                carga_desc,
            ],

            # Inferior izquierdo
            [
                x_inferior_izquierdo,
                carga_desc,
            ],
        ])

        diagnostico = {
            "fraccion_banda": float(
                fraccion_banda
            ),
            "tolerancia": float(
                tolerancia
            ),
            "asc_tolerancia_real": float(
                extremos_asc["tolerancia_real"]
            ),
            "desc_tolerancia_real": float(
                extremos_desc["tolerancia_real"]
            ),
            "desplazamiento_lateral": float(
                desplazamiento_lateral
            ),
            "x_inferior_derecho_paralelo": float(
                x_inferior_derecho_paralelo
            ),
            "x_inferior_derecho_real": float(
                x_inferior_derecho_real
            ),
            "x_inferior_derecho_usado": float(
                x_inferior_derecho
            ),
        }

        return vertices, diagnostico


    def calcular_llenado_bomba(
        posicion,
        carga,
        ascendente,
        descendente,
        linea_asc,
        linea_desc,
        fraccion_banda=0.35,
    ):
        """
        Calcula el área real, el área ideal y el porcentaje
        estimado de llenado.
        """
        area_real = area_poligono(
            posicion,
            carga,
        )

        vertices_ideal, diagnostico = construir_carta_ideal(
            ascendente=ascendente,
            descendente=descendente,
            linea_asc=linea_asc,
            linea_desc=linea_desc,
            fraccion_banda=fraccion_banda,
        )

        area_ideal = area_poligono(
            vertices_ideal[:, 0],
            vertices_ideal[:, 1],
        )

        if (
            not np.isfinite(area_ideal)
            or area_ideal <= 0
        ):
            llenado = np.nan
        else:
            llenado = (
                100
                * area_real
                / area_ideal
            )

        return {
            "area_real": float(
                area_real
            ),
            "area_ideal": float(
                area_ideal
            ),
            "llenado_porcentaje": float(
                llenado
            ),
            "vertices_ideal":
                vertices_ideal,
            "diagnostico":
                diagnostico,
        }


    # ===== CELDA ORIGINAL 10 =====
    def corregir_horizontal_inferior_por_persistencia(
        desc,
        linea_desc,
        rango_carga_total,
        fraccion_debajo_min=0.35,
        desviacion_min_pct=0.08,
        cuantiles_banda=(0.15, 0.45),
    ):
        """
        Corrige horizontales inferiores colocadas demasiado arriba.

        Se activa solamente cuando una proporción importante de la carrera
        descendente permanece claramente por debajo de la horizontal actual.
        No reacciona ante una cola breve, para no confundir golpe de bomba
        con una meseta inferior.
        """
        x = np.asarray(desc["posicion"], dtype=float)
        y = np.asarray(desc["carga"], dtype=float)

        resultado = dict(linea_desc)
        resultado["horizontal_inferior_corregida"] = False
        resultado["fraccion_persistente_debajo"] = 0.0

        if len(y) < 8 or not np.isfinite(rango_carga_total):
            return resultado

        rango_carga_total = max(float(rango_carga_total), 1e-9)
        carga_actual = float(linea_desc["carga_representativa"])

        margen = desviacion_min_pct * rango_carga_total
        claramente_debajo = y < (carga_actual - margen)

        fraccion_debajo = float(np.mean(claramente_debajo))
        resultado["fraccion_persistente_debajo"] = fraccion_debajo

        # Una cola corta no debe mover la horizontal.
        if fraccion_debajo < fraccion_debajo_min:
            return resultado

        # Excluir extremos geométricos donde aparecen impactos y retornos.
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        rango_x = max(x_max - x_min, 1e-9)

        zona_central = (
            (x >= x_min + 0.12 * rango_x)
            & (x <= x_max - 0.08 * rango_x)
        )

        y_central = y[zona_central]

        if len(y_central) < 5:
            return resultado

        q_bajo, q_alto = np.nanquantile(
            y_central,
            cuantiles_banda,
        )

        banda_baja = (
            zona_central
            & (y >= q_bajo)
            & (y <= q_alto)
        )

        indices = np.flatnonzero(banda_baja)

        if len(indices) < 3:
            return resultado

        nueva_carga = float(np.nanmedian(y[indices]))

        # Solo corregir hacia abajo.
        if nueva_carga >= carga_actual - margen:
            return resultado

        resultado.update({
            "carga_representativa": nueva_carga,
            "indices": indices,
            "cantidad_puntos": int(len(indices)),
            "posicion_inicio": float(np.nanmin(x[indices])),
            "posicion_fin": float(np.nanmax(x[indices])),
            "horizontal_inferior_corregida": True,
        })

        return resultado


    # ============================================================
    # ANALIZAR CARTAS CON COMPUERTA DE CALIDAD DE HORIZONTALES
    # ============================================================

    def calidad_horizontal_tramo(rama, linea, rango_y_total):
        x = np.asarray(rama["posicion"], float)[linea["indices"]]
        y = np.asarray(rama["carga"], float)[linea["indices"]]
        if len(x) < 2 or np.ptp(x) <= 0:
            return np.nan
        pendiente = np.polyfit(x, y, 1)[0]
        return float(abs(pendiente) * np.ptp(x) / max(rango_y_total, 1e-9))


    def _segmentos_se_intersectan(a, b, c, d, tolerancia=1e-10):
        """Detecta un cruce propio entre dos segmentos no adyacentes."""
        def orientacion(p, q, r):
            return (
                (q[0] - p[0]) * (r[1] - p[1])
                - (q[1] - p[1]) * (r[0] - p[0])
            )

        o1 = orientacion(a, b, c)
        o2 = orientacion(a, b, d)
        o3 = orientacion(c, d, a)
        o4 = orientacion(c, d, b)

        # Solo contamos cruces claros. Los contactos colineales o en
        # extremos pueden aparecer legítimamente al cerrar la carta.
        return bool(
            o1 * o2 < -tolerancia
            and o3 * o4 < -tolerancia
        )


    def evaluar_integridad_geometrica(posicion, carga, peso_api=np.nan):
        """
        Valida que el orden de adquisición represente un ciclo físico.

        La compacidad o el área pequeña NO invalidan una carta: una bomba
        bloqueada puede producir un contorno muy fino pero perfectamente
        continuo. Aquí se buscan únicamente desorden, discontinuidades y
        cruces incompatibles con un recorrido físico.
        """
        x = np.asarray(posicion, dtype=float)
        y = np.asarray(carga, dtype=float)

        resultado_base = {
            "valida": True,
            "evidencias": [],
            "saltos_grandes": 0,
            "reversiones_posicion": 0,
            "cruces_propios": 0,
            "cruces_extremo_izquierdo": 0,
            "cruces_fuera_extremo_izquierdo": 0,
            "rango_carga_sobre_peso_api": np.nan,
        }

        if (
            len(x) < 10
            or len(x) != len(y)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            resultado_base.update({
                "valida": False,
                "evidencias": ["SERIE_INCOMPLETA_O_NO_FINITA"],
            })
            return resultado_base

        rango_x = float(np.ptp(x))
        rango_y = float(np.ptp(y))
        if rango_x <= 0 or rango_y <= 0:
            resultado_base.update({
                "valida": False,
                "evidencias": ["RECORRIDO_O_CARGA_SIN_RANGO"],
            })
            return resultado_base

        # Saltos bidimensionales normalizados. Una carta normal puede
        # presentar uno o dos cambios rápidos en sus extremos; exigimos
        # varios saltos grandes para considerarla corrupta.
        pasos = np.hypot(
            np.diff(x) / rango_x,
            np.diff(y) / rango_y,
        )
        saltos_grandes = int(np.sum(pasos > 0.45))

        # Inversiones significativas del sentido de posición después de
        # un suavizado corto. Un ciclo normal tiene esencialmente una ida
        # y una vuelta; se toleran pequeñas oscilaciones y puntos repetidos.
        ventana = 5 if len(x) >= 15 else 3
        nucleo = np.ones(ventana, dtype=float) / ventana
        x_suave = np.convolve(x, nucleo, mode="same")
        margen = 0.025 * rango_x
        dx = np.diff(x_suave)
        signos = np.sign(dx[np.abs(dx) > margen])
        reversiones = int(
            np.sum(signos[1:] != signos[:-1])
        ) if len(signos) > 1 else 0

        # Cruces propios del contorno. Se excluyen segmentos vecinos y
        # el par formado por el primero y el último.
        puntos = np.column_stack([
            (x - np.nanmin(x)) / rango_x,
            (y - np.nanmin(y)) / rango_y,
        ])
        cruces = 0
        posiciones_x_cruces = []
        cantidad_segmentos = len(puntos) - 1
        for i in range(cantidad_segmentos):
            for j in range(i + 2, cantidad_segmentos):
                if i == 0 and j == cantidad_segmentos - 1:
                    continue
                if _segmentos_se_intersectan(
                    puntos[i],
                    puntos[i + 1],
                    puntos[j],
                    puntos[j + 1],
                ):
                    cruces += 1
                    # Coordenada normalizada del cruce. Permite distinguir
                    # cruces físicos localizados en el extremo izquierdo
                    # (por ejemplo, un golpe de bomba) de un contorno
                    # desordenado con cruces distribuidos.
                    p = puntos[i]
                    r = puntos[i + 1] - puntos[i]
                    q = puntos[j]
                    s = puntos[j + 1] - puntos[j]
                    denominador = (
                        r[0] * s[1]
                        - r[1] * s[0]
                    )
                    if abs(denominador) > 1e-12:
                        qp = q - p
                        parametro = (
                            qp[0] * s[1]
                            - qp[1] * s[0]
                        ) / denominador
                        x_cruce = float(
                            p[0] + parametro * r[0]
                        )
                    else:
                        x_cruce = float(np.mean([
                            p[0],
                            puntos[i + 1, 0],
                            q[0],
                            puntos[j + 1, 0],
                        ]))
                    posiciones_x_cruces.append(x_cruce)

        # Los cruces dentro del primer 12 % del recorrido se auditan,
        # pero no invalidan por sí solos la carta: pueden formar parte
        # de la firma de un golpe de bomba localizado.
        cruces_extremo_izquierdo = int(np.sum(
            np.asarray(posiciones_x_cruces) <= 0.12
        ))
        cruces_fuera_extremo_izquierdo = int(
            cruces - cruces_extremo_izquierdo
        )

        ratio_escala = (
            rango_y / float(peso_api)
            if np.isfinite(peso_api) and peso_api > 0
            else np.nan
        )

        evidencias = []
        indicadores_moderados = 0

        if saltos_grandes >= 4:
            evidencias.append("MULTIPLES_SALTOS_ENTRE_PUNTOS")
            indicadores_moderados += 1
        if reversiones >= 5:
            evidencias.append("DEMASIADAS_INVERSIONES_DE_POSICION")
            indicadores_moderados += 1
        if cruces_fuera_extremo_izquierdo >= 4:
            evidencias.append("MULTIPLES_CRUCES_DEL_CONTORNO")
            indicadores_moderados += 1
        elif cruces_extremo_izquierdo >= 4:
            evidencias.append(
                "CRUCES_LOCALIZADOS_EXTREMO_IZQUIERDO"
            )
        if np.isfinite(ratio_escala) and ratio_escala > 8.0:
            evidencias.append("RANGO_CARGA_INCOMPATIBLE_CON_PESO_API")
            # Se conserva para auditoría, pero no invalida la carta:
            # el propio peso informado por la API puede ser anómalo.

        escala_extrema_y_recorrido_inconsistente = bool(
            np.isfinite(ratio_escala)
            and ratio_escala > 10.0
            and reversiones >= 3
        )
        if escala_extrema_y_recorrido_inconsistente:
            evidencias.append(
                "ESCALA_EXTREMA_Y_RECORRIDO_INCONSISTENTE"
            )

        # Un indicador extremadamente marcado alcanza por sí solo.
        # Para señales moderadas exigimos concurrencia, reduciendo falsos
        # positivos sobre cartas reales complejas pero físicamente válidas.
        invalida = bool(
            saltos_grandes >= 6
            or reversiones >= 8
            or cruces_fuera_extremo_izquierdo >= 8
            or indicadores_moderados >= 2
            or escala_extrema_y_recorrido_inconsistente
        )

        return {
            "valida": not invalida,
            "evidencias": evidencias,
            "saltos_grandes": saltos_grandes,
            "reversiones_posicion": reversiones,
            "cruces_propios": cruces,
            "cruces_extremo_izquierdo":
                cruces_extremo_izquierdo,
            "cruces_fuera_extremo_izquierdo":
                cruces_fuera_extremo_izquierdo,
            "rango_carga_sobre_peso_api": ratio_escala,
        }


    def corregir_horizontales_por_friccion_redondeada(
        posicion,
        carga,
        ascendente,
        descendente,
        linea_asc,
        linea_desc,
    ):
        """
        Acerca ambas horizontales cuando las ramas presentan el arqueo
        opuesto y sostenido característico de fricción.

        La corrección es deliberadamente conservadora: no se activa por
        una loma puntual ni por un vacío derecho aislado. Exige que la
        meseta superior esté arqueada hacia abajo, la inferior hacia
        arriba y que los niveles robustos interiores reduzcan el gap en
        proporciones acotadas. Devuelve además métricas auditables.
        """
        asc_corregida = dict(linea_asc)
        desc_corregida = dict(linea_desc)
        carga_sup_original = float(
            linea_asc["carga_representativa"]
        )
        carga_inf_original = float(
            linea_desc["carga_representativa"]
        )
        salida = {
            "detectada": False,
            "aplicada": False,
            "carga_sup_original": carga_sup_original,
            "carga_inf_original": carga_inf_original,
            "carga_sup_corregida": carga_sup_original,
            "carga_inf_corregida": carga_inf_original,
            "arqueo_superior_pct_gap": np.nan,
            "arqueo_inferior_pct_gap": np.nan,
            "curvatura_superior": np.nan,
            "curvatura_inferior": np.nan,
            "desvio_inferior_u85_pct_gap": np.nan,
            "reduccion_gap_pct": 0.0,
        }

        try:
            x_total = np.asarray(posicion, dtype=float)
            y_total = np.asarray(carga, dtype=float)
            rango_x = float(np.ptp(x_total))
            rango_y = float(np.ptp(y_total))
            gap_original = carga_sup_original - carga_inf_original
            if (
                rango_x <= 1e-9
                or rango_y <= 1e-9
                or gap_original <= 1e-9
            ):
                return asc_corregida, desc_corregida, salida

            x_min = float(np.nanmin(x_total))
            grilla_u = np.linspace(0.12, 0.88, 61)
            grilla_x = x_min + rango_x * grilla_u

            def interpolar_rama_robusta(rama):
                tabla = pd.DataFrame({
                    "x": np.asarray(rama["posicion"], dtype=float),
                    "y": np.asarray(rama["carga"], dtype=float),
                })
                tabla = tabla.replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                tabla = (
                    tabla.groupby("x", as_index=False)["y"]
                    .median()
                    .sort_values("x")
                )
                if len(tabla) < 4:
                    return None
                return np.interp(
                    grilla_x,
                    tabla["x"].to_numpy(dtype=float),
                    tabla["y"].to_numpy(dtype=float),
                )

            y_sup = interpolar_rama_robusta(ascendente)
            y_inf = interpolar_rama_robusta(descendente)
            if y_sup is None or y_inf is None:
                return asc_corregida, desc_corregida, salida

            # Niveles interiores robustos. No se usan extremos para evitar
            # que un pico, impacto o loma puntual mueva las horizontales.
            nivel_sup_interior = float(np.nanquantile(y_sup, 0.20))
            nivel_inf_interior = float(np.nanquantile(y_inf, 0.80))
            arqueo_sup = (
                carga_sup_original - nivel_sup_interior
            ) / gap_original
            arqueo_inf = (
                nivel_inf_interior - carga_inf_original
            ) / gap_original

            coef_sup = np.polyfit(
                grilla_u - 0.50,
                y_sup / rango_y,
                2,
            )
            coef_inf = np.polyfit(
                grilla_u - 0.50,
                y_inf / rango_y,
                2,
            )
            curvatura_sup = float(coef_sup[0])
            curvatura_inf = float(coef_inf[0])

            indice_u85 = int(np.argmin(np.abs(grilla_u - 0.85)))
            desvio_inf_u85 = (
                y_inf[indice_u85] - nivel_inf_interior
            ) / gap_original

            salida.update({
                "arqueo_superior_pct_gap": float(100.0 * arqueo_sup),
                "arqueo_inferior_pct_gap": float(100.0 * arqueo_inf),
                "curvatura_superior": curvatura_sup,
                "curvatura_inferior": curvatura_inf,
                "desvio_inferior_u85_pct_gap": float(
                    100.0 * desvio_inf_u85
                ),
            })

            # Ventana estrecha para el patrón de fricción redondeada.
            # Las cotas superiores evitan corregir cartas de admisión
            # incompleta, donde el gran vacío derecho es el fenómeno real.
            patron_redondeado_ambas_ramas = bool(
                # La rama superior puede quedar casi plana aun cuando la
                # fricción redondee de forma clara y sostenida la inferior.
                # Se admite un arqueo superior prácticamente nulo, pero se
                # conservan la curvatura opuesta, el arqueo inferior y el
                # control del extremo derecho como condiciones conjuntas.
                0.00 <= arqueo_sup <= 0.10
                and 0.08 <= arqueo_inf <= 0.20
                and -0.70 <= curvatura_sup <= -0.08
                and 0.30 <= curvatura_inf <= 1.50
                and desvio_inf_u85 <= 0.15
            )
            patron_asimetrico_rama_inferior = bool(
                # Variante en la que la fricción se expresa principalmente
                # como una cubeta inferior, mientras la rama superior queda
                # prácticamente plana. Se mantiene una ventana estrecha para
                # no confundirla con una transferencia de admisión derecha.
                -0.01 <= arqueo_sup <= 0.08
                and 0.06 <= arqueo_inf <= 0.20
                and abs(curvatura_sup) <= 0.08
                and 0.30 <= curvatura_inf <= 1.50
                and desvio_inf_u85 <= 0.15
            )
            detectada = bool(
                patron_redondeado_ambas_ramas
                or patron_asimetrico_rama_inferior
            )
            salida["detectada"] = detectada
            if not detectada:
                return asc_corregida, desc_corregida, salida

            nuevo_gap = nivel_sup_interior - nivel_inf_interior
            proporcion_gap = nuevo_gap / gap_original
            if not (0.70 <= proporcion_gap <= 0.90):
                return asc_corregida, desc_corregida, salida

            asc_corregida["carga_representativa"] = (
                nivel_sup_interior
            )
            desc_corregida["carga_representativa"] = (
                nivel_inf_interior
            )
            asc_corregida["horizontal_corregida_friccion"] = True
            desc_corregida["horizontal_corregida_friccion"] = True
            salida.update({
                "aplicada": True,
                "carga_sup_corregida": nivel_sup_interior,
                "carga_inf_corregida": nivel_inf_interior,
                "reduccion_gap_pct": float(
                    100.0 * (1.0 - proporcion_gap)
                ),
            })
        except Exception:
            return dict(linea_asc), dict(linea_desc), salida

        return asc_corregida, desc_corregida, salida


    def evaluar_horizontales(
        posicion, carga, asc, desc, linea_asc, linea_desc,
        peso_api, llenado_api,
    ):
        """Devuelve un estado auditable; no fuerza una respuesta."""
        x = np.asarray(posicion, float)
        y = np.asarray(carga, float)
        gap = float(linea_asc["carga_representativa"] - linea_desc["carga_representativa"])
        rango_x = max(np.ptp(x), 1e-9)
        rango_y = max(np.ptp(y), 1e-9)
        compacidad = float(area_poligono(x, y) / (rango_x * rango_y))
        ratio_gap_api = gap / peso_api if np.isfinite(peso_api) and peso_api > 0 else np.nan
        pendiente_sup = calidad_horizontal_tramo(asc, linea_asc, rango_y)
        pendiente_inf = calidad_horizontal_tramo(desc, linea_desc, rango_y)

        # Persistencia geométrica de los tramos respecto del recorrido
        # completo. Los umbrales son deliberadamente bajos porque, ante
        # compresión fuerte, la referencia inferior real puede ser corta.
        extension_sup = abs(
            float(linea_asc["posicion_fin"])
            - float(linea_asc["posicion_inicio"])
        ) / rango_x
        extension_inf = abs(
            float(linea_desc["posicion_fin"])
            - float(linea_desc["posicion_inicio"])
        ) / rango_x

        evidencias = []
        if gap <= 0:
            evidencias.append("HORIZONTALES_INVERTIDAS")
        if np.isfinite(ratio_gap_api) and ratio_gap_api < 0.50:
            evidencias.append("SEPARACION_MENOR_50PCT_PESO_API")
        if compacidad < 0.20:
            evidencias.append("CARTA_DIAGONAL_ANGOSTA")
        if np.isfinite(llenado_api) and llenado_api <= 15:
            evidencias.append("LLENADO_API_MUY_BAJO")
        if ((np.isfinite(pendiente_sup) and pendiente_sup > 0.15)
                or (np.isfinite(pendiente_inf) and pendiente_inf > 0.15)):
            evidencias.append("TRAMOS_NO_HORIZONTALES")

        # Inversión es suficiente. Sin inversión exigimos tres evidencias concurrentes.
        # Las evidencias de contexto (llenado API bajo, carta angosta o
        # diferencia contra el peso API) describen una condición operativa;
        # no deben anular por sí solas dos horizontales geométricamente
        # válidas. Si alguno de los tramos es dudoso, se conserva la
        # protección histórica de tres evidencias concurrentes.
        superior_geometricamente_valida = bool(
            np.isfinite(pendiente_sup)
            and pendiente_sup <= 0.15
            and np.isfinite(extension_sup)
            and extension_sup >= 0.08
        )
        inferior_geometricamente_valida = bool(
            np.isfinite(pendiente_inf)
            and pendiente_inf <= 0.18
            and np.isfinite(extension_inf)
            and extension_inf >= 0.06
        )
        ambas_geometricamente_validas = bool(
            superior_geometricamente_valida
            and inferior_geometricamente_valida
        )

        if gap <= 0:
            confiables = False
        elif ambas_geometricamente_validas:
            confiables = True
        else:
            confiables = len(evidencias) < 3
        return {
            "confiables": bool(confiables),
            "estado": "HORIZONTALES_OK" if confiables else "HORIZONTALES_NO_ENCONTRADAS",
            "evidencias": evidencias,
            "compacidad_carta": compacidad,
            "ratio_gap_api": ratio_gap_api,
            "pendiente_relativa_superior": pendiente_sup,
            "pendiente_relativa_inferior": pendiente_inf,
            "extension_horizontal_superior_pct": float(
                100.0 * extension_sup
            ),
            "extension_horizontal_inferior_pct": float(
                100.0 * extension_inf
            ),
            "horizontal_superior_geometricamente_valida": (
                superior_geometricamente_valida
            ),
            "horizontal_inferior_geometricamente_valida": (
                inferior_geometricamente_valida
            ),
        }


    resultados, errores = [], []

    for _, carta in muestra.iterrows():
        carta_id = int(carta["CartaId"])
        posicion = a_array(carta["Fondo_Posiciones"])
        carga = a_array(carta["Fondo_Cargas"])
        peso_api = pd.to_numeric(carta.get("PesoFluidoPromedio"), errors="coerce")
        llenado_api = pd.to_numeric(carta.get("LlenadoBomba"), errors="coerce")

        try:
            asc, desc = separar_carreras(posicion, carga)
            linea_asc = estimar_linea_horizontal(asc["posicion"], asc["carga"])
            linea_desc = estimar_descendente_menor_carga(desc["posicion"], desc["carga"])

            # Corregir una horizontal inferior persistentemente ubicada
            # por encima de la rama descendente real.
            linea_desc = corregir_horizontal_inferior_por_persistencia(
                desc=desc,
                linea_desc=linea_desc,
                rango_carga_total=np.ptp(carga),
            )

            linea_asc, linea_desc, correccion_friccion = (
                corregir_horizontales_por_friccion_redondeada(
                    posicion=posicion,
                    carga=carga,
                    ascendente=asc,
                    descendente=desc,
                    linea_asc=linea_asc,
                    linea_desc=linea_desc,
                )
            )

            calidad = evaluar_horizontales(
                posicion, carga, asc, desc, linea_asc, linea_desc,
                peso_api, llenado_api,
            )
            integridad = evaluar_integridad_geometrica(
                posicion,
                carga,
                peso_api,
            )

            if calidad["confiables"] and integridad["valida"]:
                calculo = calcular_llenado_bomba(
                    posicion, carga, asc, desc, linea_asc, linea_desc,
                    fraccion_banda=0.35,
                )
                vertices, area_ideal = calculo["vertices_ideal"], calculo["area_ideal"]
                llenado_calculado = calculo["llenado_porcentaje"]
            else:
                vertices, area_ideal, llenado_calculado = None, np.nan, np.nan

            carga_asc = float(linea_asc["carga_representativa"])
            carga_desc = float(linea_desc["carga_representativa"])
            profundidad_bomba_m = pd.to_numeric(
                carta.get("ProfundidadBomba"),
                errors="coerce",
            )
            diametro_piston_pulg = pd.to_numeric(
                carta.get("DiametroPistonBomba"),
                errors="coerce",
            )
            gravedad_especifica = pd.to_numeric(
                carta.get("GravedadEspecifica"),
                errors="coerce",
            )
            desplazamiento_desde_carrera_api = (
                calcular_desplazamiento_bruto_efectivo(
                    diametro_piston_pulg=diametro_piston_pulg,
                    carrera_inicio_pulg=pd.to_numeric(
                        carta.get("CarreraEfectivaBombaInicio"),
                        errors="coerce",
                    ),
                    carrera_fin_pulg=pd.to_numeric(
                        carta.get("CarreraEfectivaBombaFin"),
                        errors="coerce",
                    ),
                    gpm=pd.to_numeric(carta.get("GPM"), errors="coerce"),
                )
            )
            sumergencia_propia = calcular_sumergencia_desde_horizontales(
                carga_superior_lbf=carga_asc,
                carga_inferior_lbf=carga_desc,
                profundidad_bomba_m=profundidad_bomba_m,
                diametro_piston_pulg=diametro_piston_pulg,
                sg_fluido=gravedad_especifica,
                horizontales_validas=(
                    calidad["confiables"]
                    and integridad["valida"]
                ),
            )

            # Modulo independiente SAM Modificado. No participa de la carta
            # patrones, del llenado ni de las reglas diagnosticas.
            sam_modificado = calcular_sam_modificado(
                ascendente=asc,
                descendente=desc,
                profundidad_bomba_m=profundidad_bomba_m,
                diametro_piston_pulg=diametro_piston_pulg,
                presion_tubing_kg_cm2=presion_tubing_kg_cm2,
                presion_casing_kg_cm2=presion_casing_kg_cm2,
                gravedad_especifica=gravedad_especifica_sam,
                gradiente_psi_m=gradiente_sam_psi_m,
            )

            # La distancia entre cruces de la horizontal superior oculta se
            # conserva como metrica geometrica auxiliar, pero no representa
            # la carrera completa de fondo.
            carrera_entre_cruces_horizontal_peso = (
                estimar_carrera_efectiva_en_horizontal_superior(
                    posicion=posicion,
                    carga=carga,
                    carga_horizontal_superior=(
                        sam_modificado[
                            "Carga_Superior_SAM_Seleccionada_lbf"
                        ]
                    ),
                )
            )
            carrera_entre_cruces_horizontal_peso_pulg = (
                carrera_entre_cruces_horizontal_peso[
                    "Carrera_Efectiva_Fondo_Calculada_pulg"
                ]
            )

            # Carrera geometrica de fondo: recorrido total medido por la
            # posicion de la carta. No incluye ningun factor de llenado.
            carrera_geometrica_calculada_pulg = float(np.ptp(posicion))
            desplazamiento_geometrico_calculado_m3_d = (
                calcular_desplazamiento_desde_carrera(
                    diametro_piston_pulg=diametro_piston_pulg,
                    carrera_pulg=carrera_geometrica_calculada_pulg,
                    gpm=pd.to_numeric(carta.get("GPM"), errors="coerce"),
                )
            )
            carrera_total_fondo_calculada_pulg = (
                carrera_geometrica_calculada_pulg
            )
            desplazamiento_total_calculado_m3_d = (
                calcular_desplazamiento_desde_carrera(
                    diametro_piston_pulg=diametro_piston_pulg,
                    carrera_pulg=carrera_total_fondo_calculada_pulg,
                    gpm=pd.to_numeric(carta.get("GPM"), errors="coerce"),
                )
            )
            desplazamiento_api_m3_d = pd.to_numeric(
                carta.get("DesplazamientoEfectivo"), errors="coerce"
            )
            desplazamiento_total_api_m3_d = pd.to_numeric(
                carta.get("DesplazamientoTotal"), errors="coerce"
            )
            escurrimiento_api_m3_d = pd.to_numeric(
                carta.get("Escurrimiento"), errors="coerce"
            )
            resultados.append({
                "CartaId": carta_id, "Pozo": carta["Pozo"], "Fecha": carta["Fecha"],
                "GPM": pd.to_numeric(carta.get("GPM"), errors="coerce"),
                "Profundidad_Bomba_m": profundidad_bomba_m,
                "Diametro_Piston_pulg": diametro_piston_pulg,
                "Gravedad_Especifica_API": gravedad_especifica,
                # Alias historicos, conservados para compatibilidad.
                "Carrera_Efectiva_Fondo_pulg": (
                    desplazamiento_desde_carrera_api[
                        "Carrera_Efectiva_Fondo_pulg"
                    ]
                ),
                "Desplazamiento_Bruto_Efectivo_m3_d": (
                    desplazamiento_geometrico_calculado_m3_d
                ),
                "Carrera_Entre_Cruces_Horizontal_Peso_pulg": (
                    carrera_entre_cruces_horizontal_peso_pulg
                ),
                "Posicion_Cruce_Superior_Izquierda_pulg": (
                    carrera_entre_cruces_horizontal_peso.get(
                        "Posicion_Cruce_Superior_Izquierda_pulg",
                        np.nan,
                    )
                ),
                "Posicion_Cruce_Superior_Derecha_pulg": (
                    carrera_entre_cruces_horizontal_peso.get(
                        "Posicion_Cruce_Superior_Derecha_pulg",
                        np.nan,
                    )
                ),
                "Cantidad_Cruces_Horizontal_Superior": (
                    carrera_entre_cruces_horizontal_peso.get(
                        "Cantidad_Cruces_Horizontal_Superior",
                        0,
                    )
                ),
                "Carrera_Geometrica_Fondo_Calculada_pulg": (
                    carrera_geometrica_calculada_pulg
                ),
                "Desplazamiento_Bruto_Geometrico_Calculado_m3_d": (
                    desplazamiento_geometrico_calculado_m3_d
                ),
                "Carrera_Efectiva_Fondo_API_pulg": (
                    desplazamiento_desde_carrera_api[
                        "Carrera_Efectiva_Fondo_pulg"
                    ]
                ),
                "Desplazamiento_Desde_Carrera_Efectiva_API_m3_d": (
                    desplazamiento_desde_carrera_api[
                        "Desplazamiento_Bruto_Efectivo_m3_d"
                    ]
                ),
                "Desplazamiento_Bruto_Efectivo_Calculado_m3_d": (
                    desplazamiento_geometrico_calculado_m3_d
                ),
                "Carrera_Total_Fondo_Calculada_pulg": (
                    carrera_total_fondo_calculada_pulg
                ),
                "Desplazamiento_Bruto_Total_Calculado_m3_d": (
                    desplazamiento_total_calculado_m3_d
                ),
                "Escurrimiento_Calculado_m3_d": (
                    np.nan
                ),
                "Llenado_Implicito_Carrera_Efectiva_pct": (
                    np.nan
                ),
                "Desplazamiento_Bruto_Efectivo_API_m3_d": (
                    desplazamiento_api_m3_d
                ),
                "Desplazamiento_Bruto_Total_API_m3_d": (
                    desplazamiento_total_api_m3_d
                ),
                "Escurrimiento_API_m3_d": escurrimiento_api_m3_d,
                "Delta_Desplazamiento_Calculado_vs_API_m3_d": (
                    np.nan
                ),
                "Delta_Desplazamiento_Total_Calculado_vs_API_m3_d": (
                    desplazamiento_total_calculado_m3_d
                    - desplazamiento_total_api_m3_d
                ),
                "Delta_Escurrimiento_Calculado_vs_API_m3_d": (
                    np.nan
                ),
                "Carrera_Fondo_pulg": float(np.ptp(posicion)),
                "Estado_Horizontales": calidad["estado"],
                "Evidencias_Horizontales": calidad["evidencias"],
                "Posible_Sin_Trabajo_Bomba": (
                    not calidad["confiables"]
                    and integridad["valida"]
                ),
                "Carta_Geometricamente_Valida": integridad["valida"],
                "Posible_Carta_No_Valida": not integridad["valida"],
                "Evidencias_Integridad": integridad["evidencias"],
                "Saltos_Grandes_Carta": integridad["saltos_grandes"],
                "Reversiones_Posicion_Carta": integridad["reversiones_posicion"],
                "Cruces_Propios_Carta": integridad["cruces_propios"],
                "Cruces_Extremo_Izquierdo_Carta": (
                    integridad["cruces_extremo_izquierdo"]
                ),
                "Cruces_Fuera_Extremo_Izquierdo_Carta": (
                    integridad["cruces_fuera_extremo_izquierdo"]
                ),
                "Rango_Carga_Sobre_Peso_API": (
                    integridad["rango_carga_sobre_peso_api"]
                ),
                "Compacidad_Carta": calidad["compacidad_carta"],
                "Separacion_Sobre_Peso_API": calidad["ratio_gap_api"],
                "Pendiente_Relativa_Superior": calidad["pendiente_relativa_superior"],
                "Pendiente_Relativa_Inferior": calidad["pendiente_relativa_inferior"],
                "Extension_Horizontal_Superior_pct": (
                    calidad["extension_horizontal_superior_pct"]
                ),
                "Extension_Horizontal_Inferior_pct": (
                    calidad["extension_horizontal_inferior_pct"]
                ),
                "Horizontal_Superior_Geometricamente_Valida": (
                    calidad[
                        "horizontal_superior_geometricamente_valida"
                    ]
                ),
                "Horizontal_Inferior_Geometricamente_Valida": (
                    calidad[
                        "horizontal_inferior_geometricamente_valida"
                    ]
                ),
                "Carga_Asc_Geometrica": carga_asc, "Carga_Desc_Geometrica": carga_desc,
                "Friccion_Elevada_Geometrica": (
                    correccion_friccion["detectada"]
                ),
                "Correccion_Friccion_Aplicada": (
                    correccion_friccion["aplicada"]
                ),
                "Carga_Asc_Antes_Friccion": (
                    correccion_friccion["carga_sup_original"]
                ),
                "Carga_Desc_Antes_Friccion": (
                    correccion_friccion["carga_inf_original"]
                ),
                "Arqueo_Superior_Friccion_pct_gap": (
                    correccion_friccion["arqueo_superior_pct_gap"]
                ),
                "Arqueo_Inferior_Friccion_pct_gap": (
                    correccion_friccion["arqueo_inferior_pct_gap"]
                ),
                "Curvatura_Superior_Friccion": (
                    correccion_friccion["curvatura_superior"]
                ),
                "Curvatura_Inferior_Friccion": (
                    correccion_friccion["curvatura_inferior"]
                ),
                "Reduccion_Gap_Friccion_pct": (
                    correccion_friccion["reduccion_gap_pct"]
                ),
                "Separacion_Horizontales": carga_asc - carga_desc,
                **sumergencia_propia,
                **sam_modificado,
                "Area_Real": area_poligono(posicion, carga), "Area_Ideal": area_ideal,
                "Llenado_Calculado_pct": llenado_calculado,
                "Llenado_API_pct": llenado_api,
                "Peso_Fluido_API_lbf": peso_api,
                "Peso_Fluido_Max_API_lbf": pd.to_numeric(carta.get("PesoFluidoMax"), errors="coerce"),
                "Peso_Fluido_Teorico_API_lbf": pd.to_numeric(carta.get("Foc"), errors="coerce"),
                "Sumergencia_API_m": pd.to_numeric(carta.get("Sumergencia"), errors="coerce"),
                "Nivel_API_m": pd.to_numeric(carta.get("Nivel"), errors="coerce"),
                "Presion_Fondo_API": pd.to_numeric(carta.get("PresionDeFondo"), errors="coerce"),
                "Valvula_Fija_API": pd.to_numeric(carta.get("ValvulaFija"), errors="coerce"),
                "Valvula_Movil_API": pd.to_numeric(carta.get("ValvulaMovil"), errors="coerce"),
                "Carga_Maxima_Bomba_API": pd.to_numeric(carta.get("CargaMaximaBomba"), errors="coerce"),
                "Carga_Minima_Bomba_API": pd.to_numeric(carta.get("CargaMinimaBomba"), errors="coerce"),
                "Carga_Maxima_Superficie_API": pd.to_numeric(carta.get("CargaMaximaSuperficie"), errors="coerce"),
                "Carga_Minima_Superficie_API": pd.to_numeric(carta.get("CargaMinimaSuperficie"), errors="coerce"),
                "Vertices_Ideal": vertices, "Linea_Asc": linea_asc, "Linea_Desc": linea_desc,
                "Ascendente": asc, "Descendente": desc,
            })
        except Exception as error:
            errores.append({"CartaId": carta_id, "Pozo": carta["Pozo"], "Error": str(error)})

    resultados_cartas = pd.DataFrame(resultados)
    errores_cartas = pd.DataFrame(errores)
    print(f"Cartas analizadas: {len(resultados_cartas)}")
    print(f"Horizontales no encontradas: {(resultados_cartas['Estado_Horizontales'] != 'HORIZONTALES_OK').sum()}")
    print(f"Errores técnicos: {len(errores_cartas)}")
    display(resultados_cartas[[
        "CartaId", "Pozo", "Estado_Horizontales", "Evidencias_Horizontales",
        "Posible_Sin_Trabajo_Bomba", "Compacidad_Carta", "Separacion_Sobre_Peso_API",
        "Pendiente_Relativa_Superior", "Pendiente_Relativa_Inferior",
        "Extension_Horizontal_Superior_pct",
        "Extension_Horizontal_Inferior_pct",
        "Llenado_Calculado_pct", "Llenado_API_pct", "Peso_Fluido_API_lbf",
        "Sumergencia_API_m",
    ]].round(3))


    # ===== CELDA ORIGINAL 11 =====
    # ============================================================
    # CORRECCIONES GEOMÉTRICAS PREVIAS A LA CARTA IDEAL
    # ============================================================

    # Este bloque:
    # 1. Rechaza cartas extremadamente angostas.
    # 2. Busca una meseta baja tardía en cartas con indicios
    #    de compresión/golpe de fluido y horizontal inferior
    #    seleccionada demasiado arriba.


    def tramo_consecutivo_mas_largo_indices(
        indices,
    ):
        """
        Dentro de una lista de índices ordenados,
        devuelve el grupo consecutivo más largo.
        """
        indices = np.asarray(
            indices,
            dtype=int,
        )

        if len(indices) == 0:
            return np.array(
                [],
                dtype=int,
            )

        cortes = np.flatnonzero(
            np.diff(indices) != 1
        ) + 1

        grupos = np.split(
            indices,
            cortes,
        )

        return max(
            grupos,
            key=len,
        )


    def buscar_meseta_baja_tardia(
        descendente,
        carga_total,
        fraccion_inicio=0.55,
        cuantil_bajo=0.25,
        tolerancia_relativa=0.12,
        minimo_puntos=4,
    ):
        """
        Busca una zona baja relativamente estable en la
        parte final de la carrera descendente.

        Está pensada para cartas donde una meseta alta de
        compresión fue confundida con la horizontal inferior.
        """
        x = np.asarray(
            descendente["posicion"],
            dtype=float,
        )

        y = np.asarray(
            descendente["carga"],
            dtype=float,
        )

        n = len(y)

        if n < minimo_puntos:
            return None

        inicio_final = int(
            np.floor(
                fraccion_inicio * n
            )
        )

        indices_finales = np.arange(
            inicio_final,
            n,
        )

        if len(indices_finales) < minimo_puntos:
            return None

        y_final = y[
            indices_finales
        ]

        nivel_bajo_inicial = np.quantile(
            y_final,
            cuantil_bajo,
        )

        rango_carga_total = max(
            np.ptp(
                np.asarray(
                    carga_total,
                    dtype=float,
                )
            ),
            1e-9,
        )

        tolerancia = (
            tolerancia_relativa
            * rango_carga_total
        )

        candidatos = indices_finales[
            np.abs(
                y[indices_finales]
                - nivel_bajo_inicial
            )
            <= tolerancia
        ]

        tramo = (
            tramo_consecutivo_mas_largo_indices(
                candidatos
            )
        )

        if len(tramo) < minimo_puntos:
            return None

        amplitud_x = np.ptp(
            x[tramo]
        )

        rango_x = max(
            np.ptp(x),
            1e-9,
        )

        # Evitar tomar únicamente puntos verticales
        # concentrados en una sola posición.
        if amplitud_x < 0.06 * rango_x:
            return None

        nivel = float(
            np.median(
                y[tramo]
            )
        )

        return {
            "carga_representativa":
                nivel,
            "indices":
                tramo,
            "posicion_inicio":
                float(
                    np.min(
                        x[tramo]
                    )
                ),
            "posicion_fin":
                float(
                    np.max(
                        x[tramo]
                    )
                ),
            "cantidad_puntos":
                len(tramo),
            "tolerancia":
                float(tolerancia),
        }


    # ============================================================
    # PREPARAR COLUMNAS DE DIAGNÓSTICO
    # ============================================================

    if (
        "Metodo_Horizontal_Inferior"
        not in resultados_cartas.columns
    ):
        resultados_cartas[
            "Metodo_Horizontal_Inferior"
        ] = "ESTANDAR"


    resultados_cartas[
        "Horizontal_Inferior_Corregida"
    ] = False


    # ============================================================
    # 1. RECHAZAR CARTAS EXTREMADAMENTE ANGOSTAS
    # ============================================================

    mascara_angosta_extrema = (
        resultados_cartas[
            "Estado_Horizontales"
        ].eq(
            "HORIZONTALES_OK"
        )
        & resultados_cartas[
            "Carta_Geometricamente_Valida"
        ].fillna(False)
        & (
            resultados_cartas[
                "Compacidad_Carta"
            ] < 0.05
        )
    )


    for indice in resultados_cartas.index[
        mascara_angosta_extrema
    ]:
        evidencias = list(
            resultados_cartas.at[
                indice,
                "Evidencias_Horizontales",
            ]
        )

        if (
            "COMPACIDAD_EXTREMA_MENOR_5PCT"
            not in evidencias
        ):
            evidencias.append(
                "COMPACIDAD_EXTREMA_MENOR_5PCT"
            )

        resultados_cartas.at[
            indice,
            "Estado_Horizontales",
        ] = "HORIZONTALES_NO_ENCONTRADAS"

        resultados_cartas.at[
            indice,
            "Evidencias_Horizontales",
        ] = evidencias

        resultados_cartas.at[
            indice,
            "Posible_Sin_Trabajo_Bomba",
        ] = True

        resultados_cartas.at[
            indice,
            "Area_Ideal",
        ] = np.nan

        resultados_cartas.at[
            indice,
            "Llenado_Calculado_pct",
        ] = np.nan

        resultados_cartas.at[
            indice,
            "Vertices_Ideal",
        ] = None


    # ============================================================
    # 2. CORREGIR HORIZONTAL INFERIOR DEMASIADO ALTA
    # ============================================================

    cartas_corregidas = []


    for indice, resultado in resultados_cartas.iterrows():
        if (
            resultado[
                "Estado_Horizontales"
            ]
            != "HORIZONTALES_OK"
        ):
            continue

        peso_api = resultado[
            "Peso_Fluido_API_lbf"
        ]

        llenado_api = resultado[
            "Llenado_API_pct"
        ]

        separacion_actual = resultado[
            "Separacion_Horizontales"
        ]

        llenado_actual = resultado[
            "Llenado_Calculado_pct"
        ]

        if (
            not np.isfinite(peso_api)
            or peso_api <= 0
        ):
            continue

        relacion_actual = (
            separacion_actual
            / peso_api
        )

        # Activar solamente en cartas con señales fuertes:
        # - llenado API extremadamente bajo,
        # - separación geométrica demasiado pequeña,
        # - llenado propio físicamente sospechoso.
        requiere_revision_inferior = (
            np.isfinite(llenado_api)
            and llenado_api <= 15
            and relacion_actual < 0.50
            and np.isfinite(llenado_actual)
            and llenado_actual > 100
        )

        if not requiere_revision_inferior:
            continue

        carta_id = int(
            resultado["CartaId"]
        )

        carta = muestra.loc[
            muestra["CartaId"]
            .astype(int)
            == carta_id
        ].iloc[0]

        posicion = a_array(
            carta["Fondo_Posiciones"]
        )

        carga = a_array(
            carta["Fondo_Cargas"]
        )

        descendente = resultado[
            "Descendente"
        ]

        nueva_linea_desc = (
            buscar_meseta_baja_tardia(
                descendente=descendente,
                carga_total=carga,
            )
        )

        if nueva_linea_desc is None:
            continue

        linea_asc = resultado[
            "Linea_Asc"
        ]

        ascendente = resultado[
            "Ascendente"
        ]

        calculo_nuevo = (
            calcular_llenado_bomba(
                posicion=posicion,
                carga=carga,
                ascendente=ascendente,
                descendente=descendente,
                linea_asc=linea_asc,
                linea_desc=nueva_linea_desc,
                fraccion_banda=0.35,
            )
        )

        llenado_nuevo = calculo_nuevo[
            "llenado_porcentaje"
        ]

        separacion_nueva = (
            linea_asc[
                "carga_representativa"
            ]
            - nueva_linea_desc[
                "carga_representativa"
            ]
        )

        relacion_nueva = (
            separacion_nueva
            / peso_api
        )

        # Aceptar solamente una corrección físicamente razonable.
        correccion_aceptable = (
            np.isfinite(
                llenado_nuevo
            )
            and 0 < llenado_nuevo <= 120
            and relacion_nueva >= 0.50
        )

        if not correccion_aceptable:
            continue

        resultados_cartas.at[
            indice,
            "Linea_Desc",
        ] = nueva_linea_desc

        resultados_cartas.at[
            indice,
            "Carga_Desc_Geometrica",
        ] = nueva_linea_desc[
            "carga_representativa"
        ]

        resultados_cartas.at[
            indice,
            "Separacion_Horizontales",
        ] = separacion_nueva

        resultados_cartas.at[
            indice,
            "Separacion_Sobre_Peso_API",
        ] = relacion_nueva

        resultados_cartas.at[
            indice,
            "Area_Ideal",
        ] = calculo_nuevo[
            "area_ideal"
        ]

        resultados_cartas.at[
            indice,
            "Llenado_Calculado_pct",
        ] = llenado_nuevo

        resultados_cartas.at[
            indice,
            "Vertices_Ideal",
        ] = calculo_nuevo[
            "vertices_ideal"
        ]

        resultados_cartas.at[
            indice,
            "Metodo_Horizontal_Inferior",
        ] = "MESETA_BAJA_TARDIA"

        resultados_cartas.at[
            indice,
            "Horizontal_Inferior_Corregida",
        ] = True

        evidencias = list(
            resultados_cartas.at[
                indice,
                "Evidencias_Horizontales",
            ]
        )

        evidencias.append(
            "HORIZONTAL_INFERIOR_CORREGIDA"
        )

        resultados_cartas.at[
            indice,
            "Evidencias_Horizontales",
        ] = evidencias

        cartas_corregidas.append({
            "CartaId": carta_id,
            "Pozo": resultado["Pozo"],
            "Horizontal_Inferior_Anterior":
                resultado[
                    "Carga_Desc_Geometrica"
                ],
            "Horizontal_Inferior_Nueva":
                nueva_linea_desc[
                    "carga_representativa"
                ],
            "Llenado_Anterior_pct":
                llenado_actual,
            "Llenado_Nuevo_pct":
                llenado_nuevo,
        })

    # ============================================================
    # 3. DESPEGUE INFERIOR DERECHO Y AJUSTE DE LA REFERENCIA
    # ============================================================

    def medir_despegue_inferior_derecho(
        descendente,
        linea_asc,
        linea_desc,
    ):
        """
        Mide cuánto se separa sostenidamente la rama inferior de su
        horizontal antes del cierre derecho de la carta.
        """
        x = np.asarray(descendente["posicion"], dtype=float)
        y = np.asarray(descendente["carga"], dtype=float)

        rango_x = float(np.ptp(x))
        gap = float(
            linea_asc["carga_representativa"]
            - linea_desc["carga_representativa"]
        )

        if (
            len(x) < 6
            or rango_x <= 0
            or gap <= 0
        ):
            return {
                "despegue_robusto_pct": np.nan,
                "extension_despegue_pct": np.nan,
                "sostenido": False,
            }

        u = (x - np.nanmin(x)) / rango_x

        # Zona derecha previa al cierre. Se elimina el último 5 %,
        # donde toda carta debe transferir carga para cerrar.
        zona = (
            (u >= 0.55)
            & (u <= 0.95)
        )

        if np.sum(zona) < 4:
            return {
                "despegue_robusto_pct": np.nan,
                "extension_despegue_pct": np.nan,
                "sostenido": False,
            }

        separacion_pct = (
            100.0
            * np.maximum(
                y - linea_desc["carga_representativa"],
                0.0,
            )
            / gap
        )

        despegue_robusto = float(
            np.nanpercentile(
                separacion_pct[zona],
                90,
            )
        )

        puntos_separados = (
            zona
            & (separacion_pct >= 20.0)
        )

        if np.sum(puntos_separados) >= 2:
            extension = float(
                100.0
                * np.ptp(x[puntos_separados])
                / rango_x
            )
        else:
            extension = 0.0

        sostenido = bool(
            despegue_robusto >= 25.0
            and extension >= 4.0
        )

        return {
            "despegue_robusto_pct": despegue_robusto,
            "extension_despegue_pct": extension,
            "sostenido": sostenido,
        }


    def proponer_horizontal_inferior_operativa(
        descendente,
        linea_asc,
        linea_desc,
    ):
        """
        Propone una referencia inferior robusta excluyendo el golpe
        izquierdo y la transferencia derecha.
        """
        x = np.asarray(descendente["posicion"], dtype=float)
        y = np.asarray(descendente["carga"], dtype=float)

        rango_x = float(np.ptp(x))
        gap = float(
            linea_asc["carga_representativa"]
            - linea_desc["carga_representativa"]
        )

        if len(x) < 8 or rango_x <= 0 or gap <= 0:
            return None

        u = (x - np.nanmin(x)) / rango_x
        zona_central = (
            (u >= 0.20)
            & (u <= 0.75)
        )

        if np.sum(zona_central) < 6:
            return None

        carga_central = y[zona_central]
        nivel_nuevo = float(
            np.nanmedian(carga_central)
        )
        dispersion_relativa = float(
            (
                np.nanpercentile(carga_central, 90)
                - np.nanpercentile(carga_central, 10)
            )
            / gap
        )
        elevacion_relativa = float(
            (
                nivel_nuevo
                - linea_desc["carga_representativa"]
            )
            / gap
        )

        # Exigimos una meseta central compacta y una diferencia
        # material respecto de la referencia anterior.
        if (
            dispersion_relativa > 0.15
            or elevacion_relativa < 0.08
            or nivel_nuevo
                >= linea_asc["carga_representativa"] - 0.30 * gap
        ):
            return None

        tolerancia = max(
            0.06 * gap,
            1e-9,
        )
        indices = np.flatnonzero(
            zona_central
            & (np.abs(y - nivel_nuevo) <= tolerancia)
        )

        if len(indices) < 3:
            return None

        nueva_linea = dict(linea_desc)
        nueva_linea.update({
            "carga_representativa": nivel_nuevo,
            "indices": indices,
            "cantidad_puntos": int(len(indices)),
            "posicion_inicio": float(np.nanmin(x[indices])),
            "posicion_fin": float(np.nanmax(x[indices])),
            "horizontal_inferior_corregida": True,
        })
        return nueva_linea


    for indice, resultado in resultados_cartas.iterrows():
        if (
            resultado["Estado_Horizontales"]
            != "HORIZONTALES_OK"
            or not bool(
                resultado.get(
                    "Carta_Geometricamente_Valida",
                    True,
                )
            )
        ):
            resultados_cartas.at[
                indice,
                "Despegue_Inferior_Derecho_pct",
            ] = np.nan
            resultados_cartas.at[
                indice,
                "Extension_Despegue_Inferior_Derecho_pct",
            ] = np.nan
            resultados_cartas.at[
                indice,
                "Transferencia_Inferior_Sostenida",
            ] = False
            continue

        linea_asc = resultado["Linea_Asc"]
        linea_desc = resultado["Linea_Desc"]
        descendente = resultado["Descendente"]

        despegue = medir_despegue_inferior_derecho(
            descendente,
            linea_asc,
            linea_desc,
        )

        # Si la rama no se despega sostenidamente, una cola izquierda
        # no debe arrastrar hacia abajo toda la horizontal inferior.
        if not despegue["sostenido"]:
            nueva_linea = proponer_horizontal_inferior_operativa(
                descendente,
                linea_asc,
                linea_desc,
            )

            if nueva_linea is not None:
                carta_id = int(resultado["CartaId"])
                carta = muestra.loc[
                    muestra["CartaId"].astype(int)
                    == carta_id
                ].iloc[0]
                posicion = a_array(carta["Fondo_Posiciones"])
                carga = a_array(carta["Fondo_Cargas"])

                calculo_nuevo = calcular_llenado_bomba(
                    posicion=posicion,
                    carga=carga,
                    ascendente=resultado["Ascendente"],
                    descendente=descendente,
                    linea_asc=linea_asc,
                    linea_desc=nueva_linea,
                    fraccion_banda=0.35,
                )

                llenado_nuevo = calculo_nuevo[
                    "llenado_porcentaje"
                ]

                if (
                    np.isfinite(llenado_nuevo)
                    and 0 < llenado_nuevo <= 120
                ):
                    linea_desc = nueva_linea
                    separacion_nueva = float(
                        linea_asc["carga_representativa"]
                        - linea_desc["carga_representativa"]
                    )
                    peso_api = resultado["Peso_Fluido_API_lbf"]

                    resultados_cartas.at[
                        indice,
                        "Linea_Desc",
                    ] = linea_desc
                    resultados_cartas.at[
                        indice,
                        "Carga_Desc_Geometrica",
                    ] = linea_desc["carga_representativa"]
                    resultados_cartas.at[
                        indice,
                        "Separacion_Horizontales",
                    ] = separacion_nueva
                    resultados_cartas.at[
                        indice,
                        "Separacion_Sobre_Peso_API",
                    ] = (
                        separacion_nueva / peso_api
                        if np.isfinite(peso_api) and peso_api > 0
                        else np.nan
                    )
                    resultados_cartas.at[
                        indice,
                        "Area_Ideal",
                    ] = calculo_nuevo["area_ideal"]
                    resultados_cartas.at[
                        indice,
                        "Llenado_Calculado_pct",
                    ] = llenado_nuevo
                    resultados_cartas.at[
                        indice,
                        "Vertices_Ideal",
                    ] = calculo_nuevo["vertices_ideal"]
                    resultados_cartas.at[
                        indice,
                        "Metodo_Horizontal_Inferior",
                    ] = "MESETA_CENTRAL_SIN_DESPEGUE"
                    resultados_cartas.at[
                        indice,
                        "Horizontal_Inferior_Corregida",
                    ] = True

                    despegue = medir_despegue_inferior_derecho(
                        descendente,
                        linea_asc,
                        linea_desc,
                    )

        resultados_cartas.at[
            indice,
            "Despegue_Inferior_Derecho_pct",
        ] = despegue["despegue_robusto_pct"]
        resultados_cartas.at[
            indice,
            "Extension_Despegue_Inferior_Derecho_pct",
        ] = despegue["extension_despegue_pct"]
        resultados_cartas.at[
            indice,
            "Transferencia_Inferior_Sostenida",
        ] = despegue["sostenido"]


    cartas_corregidas = pd.DataFrame(
        cartas_corregidas
    )


    # ------------------------------------------------------------
    print(
        "Cartas extremadamente angostas rechazadas:",
        int(
            mascara_angosta_extrema.sum()
        ),
    )

    print(
        "Horizontales inferiores corregidas:",
        len(
            cartas_corregidas
        ),
    )

    display(
        cartas_corregidas.round(2)
    )


    # ===== CELDA ORIGINAL 12 =====
    # ============================================================
    # BASE OPERATIVA LIMPIA PARA DIAGNÓSTICOS
    # ============================================================

    base_diagnosticos = resultados_cartas.copy()

    # Fuente canónica para diagnóstico. El llenado es el calculado a partir
    # de la carta y las variables hidráulicas provienen exclusivamente del
    # SAM Modificado: peso = horizontal superior nueva - horizontal inferior
    # nueva, y sumergencia recalculada desde ese peso. Los valores API se
    # conservan sólo como referencia y comparación.
    base_diagnosticos["Llenado_Usado_pct"] = (
        base_diagnosticos["Llenado_Calculado_pct"]
    )
    base_diagnosticos["Peso_Fluido_Usado_lbf"] = pd.to_numeric(
        base_diagnosticos["Peso_Fluido_SAM_Seleccionado_lbf"],
        errors="coerce",
    )
    base_diagnosticos["Sumergencia_Usada_m"] = pd.to_numeric(
        base_diagnosticos["Sumergencia_SAM_Seleccionada_m"],
        errors="coerce",
    )

    base_diagnosticos["Carga_Real_vs_Teorica_pct"] = (
        100 * base_diagnosticos["Peso_Fluido_Usado_lbf"]
        / base_diagnosticos["Peso_Fluido_Teorico_API_lbf"]
    )
    base_diagnosticos["Diferencia_Llenado_pp"] = (
        base_diagnosticos["Llenado_Usado_pct"] - base_diagnosticos["Llenado_API_pct"]
    )

    # Desplazamiento bruto y ajustado por nuestro llenado.
    COEF_DESPLAZAMIENTO_BPD = (np.pi / 4) * 1440 / 9702
    base_diagnosticos["Desplazamiento_Bruto_bpd"] = (
        COEF_DESPLAZAMIENTO_BPD
        * base_diagnosticos["Diametro_Piston_pulg"] ** 2
        * base_diagnosticos["Carrera_Fondo_pulg"]
        * base_diagnosticos["GPM"]
    )
    base_diagnosticos["Desplazamiento_Segun_Llenado_bpd"] = (
        base_diagnosticos["Desplazamiento_Bruto_bpd"]
        * base_diagnosticos["Llenado_Usado_pct"] / 100
    )

    # Controles: no se borran anomalías, solamente se marcan.
    base_diagnosticos["Peso_SAM_Valido"] = (
        np.isfinite(base_diagnosticos["Peso_Fluido_Usado_lbf"])
        & (base_diagnosticos["Peso_Fluido_Usado_lbf"] > 0)
    )
    base_diagnosticos["Sumergencia_SAM_Valida"] = (
        np.isfinite(base_diagnosticos["Sumergencia_Usada_m"])
        & (base_diagnosticos["Sumergencia_Usada_m"] >= 0)
        & (base_diagnosticos["Sumergencia_Usada_m"] <= base_diagnosticos["Profundidad_Bomba_m"])
    )
    base_diagnosticos["Llenado_Valido"] = (
        np.isfinite(base_diagnosticos["Llenado_Usado_pct"])
        & (base_diagnosticos["Llenado_Usado_pct"] >= 0)
        & (base_diagnosticos["Llenado_Usado_pct"] <= 140)
    )
    base_diagnosticos["Datos_Operativos_Validos"] = (
        base_diagnosticos["Peso_SAM_Valido"]
        & base_diagnosticos["Sumergencia_SAM_Valida"]
        & base_diagnosticos["Llenado_Valido"]
    )

    columnas_base = [
        "CartaId", "Pozo", "Fecha", "GPM", "Profundidad_Bomba_m",
        "Diametro_Piston_pulg", "Carrera_Fondo_pulg", "Llenado_Usado_pct",
        "Llenado_API_pct", "Peso_Fluido_Usado_lbf",
        "Peso_Fluido_Teorico_API_lbf", "Carga_Real_vs_Teorica_pct",
        "Sumergencia_Usada_m", "Desplazamiento_Bruto_bpd",
        "Desplazamiento_Segun_Llenado_bpd", "Datos_Operativos_Validos",
    ]

    print("Cartas disponibles:", len(base_diagnosticos))
    print("Con datos operativos válidos:", int(base_diagnosticos["Datos_Operativos_Validos"].sum()))
    display(base_diagnosticos[columnas_base].round(2))


    # ===== CELDA ORIGINAL 14 =====
    # ============================================================
    # PRIMERAS MÉTRICAS ROBUSTAS PARA DIAGNÓSTICO
    # ============================================================

    def mad_robusto(valores):
        valores = np.asarray(valores, dtype=float)
        valores = valores[np.isfinite(valores)]
        if len(valores) == 0:
            return np.nan
        mediana = np.median(valores)
        return float(np.median(np.abs(valores - mediana)))


    filas_metricas = []
    for _, r in base_diagnosticos.iterrows():
        horizontales_ok = r["Estado_Horizontales"] == "HORIZONTALES_OK"
        if horizontales_ok:
            la, ld = r["Linea_Asc"], r["Linea_Desc"]
            asc, desc = r["Ascendente"], r["Descendente"]
            gap = max(abs(r["Separacion_Horizontales"]), 1e-9)
            ya = np.asarray(asc["carga"], float)[la["indices"]]
            yd = np.asarray(desc["carga"], float)[ld["indices"]]
            variabilidad_sup = 100 * 1.4826 * mad_robusto(ya) / gap
            variabilidad_inf = 100 * 1.4826 * mad_robusto(yd) / gap
        else:
            variabilidad_sup = variabilidad_inf = np.nan

        filas_metricas.append({
            "CartaId": r["CartaId"], "Pozo": r["Pozo"], "Fecha": r["Fecha"],
            "Estado_Horizontales": r["Estado_Horizontales"],
            "Posible_Sin_Trabajo_Bomba": r["Posible_Sin_Trabajo_Bomba"],
            "Llenado_Calculado_pct": r["Llenado_Usado_pct"],
            "Sumergencia_m": r["Sumergencia_Usada_m"],
            "Peso_Fluido_lbf": r["Peso_Fluido_Usado_lbf"],
            "Carga_Real_vs_Teorica_pct": r["Carga_Real_vs_Teorica_pct"],
            "Variabilidad_Horizontal_Superior_pct": variabilidad_sup,
            "Variabilidad_Horizontal_Inferior_pct": variabilidad_inf,
            "Datos_Operativos_Validos": r["Datos_Operativos_Validos"],
            "Despegue_Inferior_Derecho_pct":
                r.get("Despegue_Inferior_Derecho_pct", np.nan),
            "Extension_Despegue_Inferior_Derecho_pct":
                r.get(
                    "Extension_Despegue_Inferior_Derecho_pct",
                    np.nan,
                ),
            "Transferencia_Inferior_Sostenida":
                bool(
                    r.get(
                        "Transferencia_Inferior_Sostenida",
                        False,
                    )
                ),
            "Vacio_Superior_Izquierdo_pct": np.nan, "Vacio_Superior_Derecho_pct": np.nan,
            "Vacio_Inferior_Derecho_pct": np.nan, "Profundidad_Golpe_Inferior_pct": np.nan,
            "Ancho_Golpe_Inferior_pct": np.nan, "Pendiente_Transferencia_Derecha": np.nan,
            "Curvatura_Transferencia_Derecha": np.nan, "Area_Fuera_Carta_Ideal_pct": np.nan,
            "Apertura_Media_Carta_pct": np.nan,
        })

    metricas_cartas = pd.DataFrame(filas_metricas)
    display(metricas_cartas.round(2))


    # ===== CELDA ORIGINAL 19 =====
    # ============================================================
    # PASO 1 — ÁREAS FALTANTES POR CUADRANTE
    # ============================================================

    import numpy as np
    import pandas as pd


    def area_poligono_normalizada(poligono):
        """
        Área de un polígono definido como array Nx2.
        """
        poligono = np.asarray(
            poligono,
            dtype=float,
        )

        if len(poligono) < 3:
            return 0.0

        x = poligono[:, 0]
        y = poligono[:, 1]

        return float(
            0.5
            * abs(
                np.sum(
                    x * np.roll(y, -1)
                    - y * np.roll(x, -1)
                )
            )
        )


    def recortar_poligono_borde(
        poligono,
        esta_adentro,
        calcular_interseccion,
    ):
        """
        Recorta un polígono contra un único borde.
        Implementación del algoritmo Sutherland-Hodgman.
        """
        poligono = np.asarray(
            poligono,
            dtype=float,
        )

        if len(poligono) == 0:
            return np.empty(
                (0, 2),
                dtype=float,
            )

        salida = []

        punto_anterior = poligono[-1]
        anterior_adentro = esta_adentro(
            punto_anterior
        )

        for punto_actual in poligono:
            actual_adentro = esta_adentro(
                punto_actual
            )

            if actual_adentro:
                if not anterior_adentro:
                    salida.append(
                        calcular_interseccion(
                            punto_anterior,
                            punto_actual,
                        )
                    )

                salida.append(
                    punto_actual
                )

            elif anterior_adentro:
                salida.append(
                    calcular_interseccion(
                        punto_anterior,
                        punto_actual,
                    )
                )

            punto_anterior = punto_actual
            anterior_adentro = actual_adentro

        if not salida:
            return np.empty(
                (0, 2),
                dtype=float,
            )

        return np.asarray(
            salida,
            dtype=float,
        )


    def interseccion_vertical(
        punto_1,
        punto_2,
        x_borde,
    ):
        """
        Intersección del segmento con x = x_borde.
        """
        x1, y1 = punto_1
        x2, y2 = punto_2

        denominador = x2 - x1

        if abs(denominador) < 1e-12:
            return np.array([
                x_borde,
                y1,
            ])

        proporcion = (
            (x_borde - x1)
            / denominador
        )

        return np.array([
            x_borde,
            y1
            + proporcion
            * (y2 - y1),
        ])


    def interseccion_horizontal(
        punto_1,
        punto_2,
        y_borde,
    ):
        """
        Intersección del segmento con y = y_borde.
        """
        x1, y1 = punto_1
        x2, y2 = punto_2

        denominador = y2 - y1

        if abs(denominador) < 1e-12:
            return np.array([
                x1,
                y_borde,
            ])

        proporcion = (
            (y_borde - y1)
            / denominador
        )

        return np.array([
            x1
            + proporcion
            * (x2 - x1),
            y_borde,
        ])


    def recortar_a_rectangulo(
        poligono,
        xmin,
        xmax,
        ymin,
        ymax,
    ):
        """
        Devuelve la parte del polígono contenida
        dentro de un rectángulo.
        """
        resultado = np.asarray(
            poligono,
            dtype=float,
        )

        # Borde izquierdo.
        resultado = recortar_poligono_borde(
            resultado,
            esta_adentro=lambda p:
                p[0] >= xmin,
            calcular_interseccion=lambda p1, p2:
                interseccion_vertical(
                    p1,
                    p2,
                    xmin,
                ),
        )

        # Borde derecho.
        resultado = recortar_poligono_borde(
            resultado,
            esta_adentro=lambda p:
                p[0] <= xmax,
            calcular_interseccion=lambda p1, p2:
                interseccion_vertical(
                    p1,
                    p2,
                    xmax,
                ),
        )

        # Borde superior.
        resultado = recortar_poligono_borde(
            resultado,
            esta_adentro=lambda p:
                p[1] >= ymin,
            calcular_interseccion=lambda p1, p2:
                interseccion_horizontal(
                    p1,
                    p2,
                    ymin,
                ),
        )

        # Borde inferior.
        resultado = recortar_poligono_borde(
            resultado,
            esta_adentro=lambda p:
                p[1] <= ymax,
            calcular_interseccion=lambda p1, p2:
                interseccion_horizontal(
                    p1,
                    p2,
                    ymax,
                ),
        )

        return resultado


    def transformar_a_carta_normalizada(
        posicion,
        carga,
        vertices_ideal,
    ):
        """
        Transforma el paralelogramo ideal en el cuadrado [0,1] x [0,1].

        Vértices esperados:
          0: superior izquierdo
          1: superior derecho
          2: inferior derecho
          3: inferior izquierdo
        """
        puntos = np.column_stack([
            np.asarray(
                posicion,
                dtype=float,
            ),
            np.asarray(
                carga,
                dtype=float,
            ),
        ])

        vertices = np.asarray(
            vertices_ideal,
            dtype=float,
        )

        superior_izquierdo = (
            vertices[0]
        )

        vector_horizontal = (
            vertices[1]
            - vertices[0]
        )

        vector_vertical = (
            vertices[3]
            - vertices[0]
        )

        matriz = np.column_stack([
            vector_horizontal,
            vector_vertical,
        ])

        determinante = np.linalg.det(
            matriz
        )

        if abs(determinante) < 1e-12:
            raise ValueError(
                "La carta ideal no permite "
                "una transformación estable."
            )

        matriz_inversa = np.linalg.inv(
            matriz
        )

        desplazados = (
            puntos
            - superior_izquierdo
        )

        normalizados = (
            desplazados
            @ matriz_inversa.T
        )

        return normalizados


    def calcular_areas_cuadrantes(
        posicion,
        carga,
        vertices_ideal,
    ):
        """
        Calcula cuánto de cada cuadrante ideal está
        ocupado por el área real de la carta.
        """
        poligono_uv = (
            transformar_a_carta_normalizada(
                posicion,
                carga,
                vertices_ideal,
            )
        )

        cuadrantes = {
            "Superior_Izquierdo": (
                0.0,
                0.5,
                0.0,
                0.5,
            ),
            "Superior_Derecho": (
                0.5,
                1.0,
                0.0,
                0.5,
            ),
            "Inferior_Izquierdo": (
                0.0,
                0.5,
                0.5,
                1.0,
            ),
            "Inferior_Derecho": (
                0.5,
                1.0,
                0.5,
                1.0,
            ),
        }

        # Cada cuadrante representa el 25 %
        # del paralelogramo ideal.
        area_ideal_cuadrante = 0.25

        resultado = {
            "Area_Real_Normalizada":
                area_poligono_normalizada(
                    poligono_uv
                )
        }

        area_total_dentro = 0.0

        for nombre, limites in cuadrantes.items():
            xmin, xmax, ymin, ymax = limites

            recortado = (
                recortar_a_rectangulo(
                    poligono_uv,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                )
            )

            area_dentro = (
                area_poligono_normalizada(
                    recortado
                )
            )

            area_dentro = float(
                np.clip(
                    area_dentro,
                    0.0,
                    area_ideal_cuadrante,
                )
            )

            ocupacion_pct = (
                100
                * area_dentro
                / area_ideal_cuadrante
            )

            faltante_pct = (
                100
                - ocupacion_pct
            )

            resultado[
                f"Area_Ocupada_{nombre}_pct"
            ] = ocupacion_pct

            resultado[
                f"Area_Faltante_{nombre}_pct"
            ] = faltante_pct

            area_total_dentro += area_dentro

        resultado[
            "Area_Dentro_Carta_Ideal_pct"
        ] = (
            100
            * area_total_dentro
        )

        area_fuera = max(
            resultado[
                "Area_Real_Normalizada"
            ]
            - area_total_dentro,
            0.0,
        )

        resultado[
            "Area_Fuera_Carta_Ideal_pct"
        ] = (
            100
            * area_fuera
        )

        resultado[
            "Area_Real_Sobre_Ideal_pct"
        ] = (
            100
            * resultado[
                "Area_Real_Normalizada"
            ]
        )

        return resultado


    # ============================================================
    # CALCULAR LAS MÉTRICAS PARA TODAS LAS CARTAS VÁLIDAS
    # ============================================================

    resultados_cuadrantes = []
    errores_cuadrantes = []


    for _, resultado in base_diagnosticos.iterrows():
        carta_id = int(
            resultado["CartaId"]
        )

        # Las cartas sin horizontales no deben recibir
        # áreas por cuadrante.
        if (
            resultado["Estado_Horizontales"]
            != "HORIZONTALES_OK"
        ):
            resultados_cuadrantes.append({
                "CartaId": carta_id,
                "Estado_Areas_Cuadrantes":
                    "NO_APLICA_SIN_HORIZONTALES",
            })

            continue

        carta = muestra.loc[
            muestra["CartaId"].astype(int)
            == carta_id
        ].iloc[0]

        posicion = a_array(
            carta["Fondo_Posiciones"]
        )

        carga = a_array(
            carta["Fondo_Cargas"]
        )

        try:
            areas = calcular_areas_cuadrantes(
                posicion=posicion,
                carga=carga,
                vertices_ideal=resultado[
                    "Vertices_Ideal"
                ],
            )

            areas.update({
                "CartaId": carta_id,
                "Estado_Areas_Cuadrantes":
                    "OK",
            })

            resultados_cuadrantes.append(
                areas
            )

        except Exception as error:
            errores_cuadrantes.append({
                "CartaId": carta_id,
                "Pozo": resultado["Pozo"],
                "Error": str(error),
            })

            resultados_cuadrantes.append({
                "CartaId": carta_id,
                "Estado_Areas_Cuadrantes":
                    "ERROR",
            })


    areas_cuadrantes = pd.DataFrame(
        resultados_cuadrantes
    )

    errores_areas_cuadrantes = pd.DataFrame(
        errores_cuadrantes
    )


    # ============================================================
    # AGREGAR LAS ÁREAS A metricas_cartas
    # ============================================================

    columnas_areas_previas = [
        columna
        for columna in metricas_cartas.columns
        if (
            columna.startswith(
                "Area_Ocupada_"
            )
            or columna.startswith(
                "Area_Faltante_"
            )
            or columna in [
                "Area_Dentro_Carta_Ideal_pct",
                "Area_Fuera_Carta_Ideal_pct",
                "Area_Real_Sobre_Ideal_pct",
                "Area_Real_Normalizada",
                "Estado_Areas_Cuadrantes",
            ]
        )
    ]


    metricas_cartas = (
        metricas_cartas
        .drop(
            columns=columnas_areas_previas,
            errors="ignore",
        )
        .merge(
            areas_cuadrantes,
            on="CartaId",
            how="left",
        )
    )


    # Primeras equivalencias con los vacíos
    # que usaremos en las reglas.
    metricas_cartas[
        "Vacio_Superior_Izquierdo_pct"
    ] = metricas_cartas[
        "Area_Faltante_Superior_Izquierdo_pct"
    ]

    metricas_cartas[
        "Vacio_Superior_Derecho_pct"
    ] = metricas_cartas[
        "Area_Faltante_Superior_Derecho_pct"
    ]

    metricas_cartas[
        "Vacio_Inferior_Derecho_pct"
    ] = metricas_cartas[
        "Area_Faltante_Inferior_Derecho_pct"
    ]


    print(
        "Cartas con áreas calculadas:",
        (
            areas_cuadrantes[
                "Estado_Areas_Cuadrantes"
            ] == "OK"
        ).sum(),
    )

    print(
        "Cartas sin horizontales:",
        (
            areas_cuadrantes[
                "Estado_Areas_Cuadrantes"
            ]
            == "NO_APLICA_SIN_HORIZONTALES"
        ).sum(),
    )

    print(
        "Errores de cálculo:",
        len(
            errores_areas_cuadrantes
        ),
    )


    columnas_mostrar = [
        "CartaId",
        "Pozo",
        "Llenado_Calculado_pct",
        "Area_Faltante_Superior_Izquierdo_pct",
        "Area_Faltante_Superior_Derecho_pct",
        "Area_Faltante_Inferior_Izquierdo_pct",
        "Area_Faltante_Inferior_Derecho_pct",
        "Area_Dentro_Carta_Ideal_pct",
        "Area_Fuera_Carta_Ideal_pct",
        "Estado_Areas_Cuadrantes",
    ]


    display(
        metricas_cartas[
            columnas_mostrar
        ].round(2)
    )


    # ===== CELDA ORIGINAL 21 =====
    # ============================================================
    # PASO 3 — MÉTRICAS AVANZADAS Y REGLAS INICIALES
    # ============================================================

    import numpy as np
    import pandas as pd


    # ============================================================
    # UMBRALES PROVISIONALES
    # ============================================================

    # Pérdida de válvula viajera.
    UMBRAL_VACIO_SUP_IZQ_VALVULA = 15.0
    UMBRAL_VACIO_SUP_DER_VALVULA = 3.0
    UMBRAL_VACIO_INF_DER_VALVULA = 35.0
    UMBRAL_ANGULO_LATERAL_VALVULA = 87.0

    # Golpe de fluido o compresión de gas.
    UMBRAL_VACIO_SUP_DER_FLUIDO = 20.0
    UMBRAL_VACIO_INF_DER_FLUIDO = 30.0
    UMBRAL_VACIO_SUP_DER_ADMISION_SUAVE = 3.0
    UMBRAL_LLENADO_INCOMPLETO = 90.0
    UMBRAL_PENDIENTE_GOLPE_FLUIDO = 4.0
    UMBRAL_CURVATURA_GOLPE_FLUIDO = 18.0

    # Golpe de bomba.
    UMBRAL_PROFUNDIDAD_GOLPE_BOMBA = 12.0
    UMBRAL_ANCHO_MAX_GOLPE_BOMBA = 35.0

    # Pozo subexplotado.
    UMBRAL_LLENADO_SUBEXPLOTADO = 90.0
    UMBRAL_SUMERGENCIA_RELATIVA = 20.0


    # ============================================================
    # FUNCIONES GEOMÉTRICAS
    # ============================================================

    def angulo_principal_puntos(
        puntos,
    ):
        """
        Calcula la orientación principal de una nube de puntos.

        Devuelve un ángulo entre 0 y 90 grados:
          90° = lateral aproximadamente vertical.
           0° = lateral aproximadamente horizontal.
        """
        puntos = np.asarray(
            puntos,
            dtype=float,
        )

        validos = np.all(
            np.isfinite(puntos),
            axis=1,
        )

        puntos = puntos[validos]

        if len(puntos) < 3:
            return np.nan

        centrados = (
            puntos
            - np.mean(
                puntos,
                axis=0,
            )
        )

        covarianza = np.cov(
            centrados.T
        )

        valores, vectores = np.linalg.eigh(
            covarianza
        )

        vector = vectores[
            :,
            np.argmax(valores),
        ]

        angulo = np.degrees(
            np.arctan2(
                abs(vector[1]),
                abs(vector[0]),
            )
        )

        return float(
            np.clip(
                angulo,
                0.0,
                90.0,
            )
        )


    def angulos_interiores_laterales(
        puntos_izquierdos,
        puntos_derechos,
    ):
        """
        Calcula los ángulos interiores firmados de ambos laterales.

        Para laterales inclinados hacia la derecha:
        - el ángulo interior izquierdo resulta menor que 90°;
        - el ángulo interior derecho resulta mayor que 90°.
        """

        def orientar(puntos):
            """
            Calcula la orientación del eje de un lateral.
            """
            puntos = np.asarray(
                puntos,
                dtype=float,
            )

            validos = np.all(
                np.isfinite(puntos),
                axis=1,
            )

            puntos = puntos[validos]

            if len(puntos) < 2:
                return np.nan

            # Evitar que las horizontales superior e inferior
            # dominen el ajuste.
            mascara_central = (
                (puntos[:, 1] >= 0.08)
                & (puntos[:, 1] <= 0.92)
            )

            centrales = puntos[
                mascara_central
            ]

            if len(centrales) >= 2:
                puntos = centrales

            cobertura_vertical = np.ptp(
                puntos[:, 1]
            )

            if cobertura_vertical < 0.12:
                return np.nan

            # Posición horizontal en función de la carga.
            pendiente, _ = np.polyfit(
                puntos[:, 1],
                puntos[:, 0],
                deg=1,
            )

            orientacion = np.degrees(
                np.arctan2(
                    1.0,
                    pendiente,
                )
            )

            return float(
                np.clip(
                    orientacion,
                    0.0,
                    180.0,
                )
            )

        # Estos cálculos están fuera de orientar().
        eje_izquierdo = orientar(
            puntos_izquierdos
        )

        eje_derecho = orientar(
            puntos_derechos
        )

        if np.isfinite(eje_izquierdo):
            interior_izquierdo = float(
                eje_izquierdo
            )
        else:
            interior_izquierdo = np.nan

        if np.isfinite(eje_derecho):
            interior_derecho = float(
                180.0 - eje_derecho
            )
        else:
            interior_derecho = np.nan

        if (
            np.isfinite(eje_izquierdo)
            and np.isfinite(eje_derecho)
        ):
            diferencia_paralelismo = float(
                abs(
                    eje_izquierdo
                    - eje_derecho
                )
            )
        else:
            diferencia_paralelismo = np.nan

        # Este return pertenece a angulos_interiores_laterales(),
        # no a la función interna orientar().
        return {
            "angulo_interior_izquierdo":
                interior_izquierdo,

            "angulo_interior_derecho":
                interior_derecho,

            "diferencia_paralelismo":
                diferencia_paralelismo,
        }

    def metricas_transferencia_derecha(
        descendente_uv,
    ):
        """
        Mide la transferencia derecha entre el 20 % y el 80 %
        de la carga normalizada. Una transferencia corta y separada
        del extremo derecho es compatible con golpe de fluido; una
        transferencia extensa es compatible con compresiÃ³n de gas.
        """
        puntos = np.asarray(descendente_uv, dtype=float)
        puntos = puntos[np.all(np.isfinite(puntos), axis=1)]

        resultado_vacio = {
            "pendiente": np.nan,
            "curvatura": np.nan,
            "extension_horizontal_pct": np.nan,
            "ancho_20_80_pct": np.nan,
            "inicio_transferencia_u_pct": np.nan,
        }

        if len(puntos) < 4:
            return resultado_vacio

        if np.nanmedian(puntos[:3, 1]) > np.nanmedian(puntos[-3:, 1]):
            puntos = puntos[::-1]

        def cruces_ascendentes(nivel):
            cruces = []
            for i in range(len(puntos) - 1):
                u0, v0 = puntos[i]
                u1, v1 = puntos[i + 1]
                if v0 <= nivel <= v1 and v1 > v0:
                    fraccion = (nivel - v0) / max(v1 - v0, 1e-12)
                    cruces.append({
                        "indice": i,
                        "u": float(u0 + fraccion * (u1 - u0)),
                    })
            return cruces

        candidatos = []
        for cruce_20 in cruces_ascendentes(0.20):
            for cruce_80 in cruces_ascendentes(0.80):
                if cruce_80["indice"] <= cruce_20["indice"]:
                    continue
                if max(cruce_20["u"], cruce_80["u"]) < 0.45:
                    continue
                candidatos.append({
                    "cruce_20": cruce_20,
                    "cruce_80": cruce_80,
                    "ancho": abs(cruce_80["u"] - cruce_20["u"]),
                    "u_medio": (cruce_20["u"] + cruce_80["u"]) / 2,
                })

        if not candidatos:
            return resultado_vacio

        candidato = max(candidatos, key=lambda item: item["u_medio"])
        cruce_20 = candidato["cruce_20"]
        cruce_80 = candidato["cruce_80"]
        tramo = puntos[cruce_20["indice"]:cruce_80["indice"] + 2]

        if len(tramo) < 3:
            return resultado_vacio

        du = np.diff(
            tramo[:, 0]
        )

        dv = np.diff(
            tramo[:, 1]
        )

        movimientos = (
            np.abs(du)
            + np.abs(dv)
        ) > 1e-9

        du = du[movimientos]
        dv = dv[movimientos]

        if len(du) < 2:
            return resultado_vacio

        # Evitar divisiones enormes por pequeños errores numéricos.
        pendiente_local = (
            np.abs(dv)
            / np.maximum(
                np.abs(du),
                0.01,
            )
        )

        pendiente = float(
            np.median(
                pendiente_local
            )
        )

        angulos = np.unwrap(
            np.arctan2(
                dv,
                du,
            )
        )

        cambios_angulares = np.abs(
            np.diff(
                angulos
            )
        )

        curvatura = (
            float(
                np.degrees(
                    np.median(
                        cambios_angulares
                    )
                )
            )
            if len(cambios_angulares)
            else 0.0
        )

        extension_horizontal = (
            100
            * np.ptp(
                tramo[:, 0]
            )
        )

        return {
            "pendiente": pendiente,
            "curvatura": curvatura,
            "extension_horizontal_pct":
                extension_horizontal,
            "ancho_20_80_pct": float(100 * candidato["ancho"]),
            "inicio_transferencia_u_pct": float(100 * cruce_20["u"]),
        }


    def extraer_metricas_avanzadas(
        resultado,
        carta,
    ):
        """
        Calcula ángulos laterales, transferencia derecha
        y cola inferior izquierda.
        """
        posicion = a_array(
            carta["Fondo_Posiciones"]
        )

        carga = a_array(
            carta["Fondo_Cargas"]
        )

        vertices = resultado[
            "Vertices_Ideal"
        ]

        if (
            resultado["Estado_Horizontales"]
            != "HORIZONTALES_OK"
            or vertices is None
        ):
            return {
                "Angulo_Lateral_Izquierdo_deg": np.nan,
                "Angulo_Lateral_Derecho_deg": np.nan,
                "Angulo_Interior_Izquierdo_deg": np.nan,
                "Angulo_Interior_Derecho_deg": np.nan,
                "Diferencia_Paralelismo_Laterales_deg": np.nan,
                "Pendiente_Transferencia_Derecha": np.nan,
                "Curvatura_Transferencia_Derecha": np.nan,
                "Extension_Transferencia_Derecha_pct": np.nan,
                "Ancho_Transferencia_20_80_pct": np.nan,
                "Inicio_Transferencia_Derecha_pct": np.nan,
                "Profundidad_Golpe_Inferior_pct": np.nan,
                "Ancho_Golpe_Inferior_pct": np.nan,
            }

        # Transformar la carta completa al cuadrado ideal.
        puntos_uv = (
            transformar_a_carta_normalizada(
                posicion=posicion,
                carga=carga,
                vertices_ideal=vertices,
            )
        )

        u = puntos_uv[:, 0]
        v = puntos_uv[:, 1]

        # Laterales: utilizar los extremos de la distribución horizontal.
        limite_izquierdo = np.nanquantile(
            u,
            0.15,
        )

        limite_derecho = np.nanquantile(
            u,
            0.85,
        )

        puntos_izquierdos = puntos_uv[
            u <= limite_izquierdo
        ]

        puntos_derechos = puntos_uv[
            u >= limite_derecho
        ]

        angulo_izquierdo = (
            angulo_principal_puntos(
                puntos_izquierdos
            )
        )

        angulo_derecho = (
            angulo_principal_puntos(
                puntos_derechos
            )
        )

        # ========================================================
        # ÁNGULOS REALES DE LOS LATERALES
        # ========================================================
        # Para tubing libre no usamos la transformación afín de
        # la carta ideal, porque esa transformación puede enderezar
        # artificialmente los laterales.

        rango_posicion = max(
            np.ptp(posicion),
            1e-9,
        )

        rango_carga = max(
            np.ptp(carga),
            1e-9,
        )

        posicion_normalizada = (
            posicion - np.nanmin(posicion)
        ) / rango_posicion

        carga_normalizada = (
            carga - np.nanmin(carga)
        ) / rango_carga

        puntos_reales_normalizados = np.column_stack([
            posicion_normalizada,
            carga_normalizada,
        ])

        # Seleccionar los extremos reales izquierdo y derecho.
        limite_real_izquierdo = np.nanquantile(
            posicion_normalizada,
            0.18,
        )

        limite_real_derecho = np.nanquantile(
            posicion_normalizada,
            0.82,
        )

        puntos_reales_izquierdos = (
            puntos_reales_normalizados[
                posicion_normalizada
                <= limite_real_izquierdo
            ]
        )

        puntos_reales_derechos = (
            puntos_reales_normalizados[
                posicion_normalizada
                >= limite_real_derecho
            ]
        )

        angulos_firmados = angulos_interiores_laterales(
            puntos_izquierdos=puntos_reales_izquierdos,
            puntos_derechos=puntos_reales_derechos,
        )

        # Carrera descendente en coordenadas normalizadas.
        descendente = resultado[
            "Descendente"
        ]

        descendente_uv = (
            transformar_a_carta_normalizada(
                posicion=descendente["posicion"],
                carga=descendente["carga"],
                vertices_ideal=vertices,
            )
        )

        transferencia = (
            metricas_transferencia_derecha(
                descendente_uv
            )
        )

        # Cola bajo la ideal en el sector inferior izquierdo.
        mascara_golpe = (
            np.isfinite(u)
            & np.isfinite(v)
            & (u <= 0.55)
            & (v > 1.02)
        )

        if np.any(mascara_golpe):
            profundidad_golpe = (
                100
                * np.nanmax(
                    v[mascara_golpe]
                    - 1.0
                )
            )

            ancho_golpe = (
                100
                * np.ptp(
                    u[mascara_golpe]
                )
            )

        else:
            profundidad_golpe = 0.0
            ancho_golpe = 0.0

        return {
            "Angulo_Lateral_Izquierdo_deg":
                angulo_izquierdo,
            "Angulo_Lateral_Derecho_deg":
                angulo_derecho,
            "Angulo_Interior_Izquierdo_deg":
                angulos_firmados["angulo_interior_izquierdo"],
            "Angulo_Interior_Derecho_deg":
                angulos_firmados["angulo_interior_derecho"],
            "Diferencia_Paralelismo_Laterales_deg":
                angulos_firmados["diferencia_paralelismo"],
            "Pendiente_Transferencia_Derecha":
                transferencia["pendiente"],
            "Curvatura_Transferencia_Derecha":
                transferencia["curvatura"],
            "Extension_Transferencia_Derecha_pct":
                transferencia[
                    "extension_horizontal_pct"
                ],
            "Ancho_Transferencia_20_80_pct":
                transferencia["ancho_20_80_pct"],
            "Inicio_Transferencia_Derecha_pct":
                transferencia["inicio_transferencia_u_pct"],
            "Profundidad_Golpe_Inferior_pct":
                profundidad_golpe,
            "Ancho_Golpe_Inferior_pct":
                ancho_golpe,
        }


    # ============================================================
    # CALCULAR MÉTRICAS AVANZADAS
    # ============================================================

    filas_avanzadas = []


    for _, resultado in base_diagnosticos.iterrows():
        carta_id = int(
            resultado["CartaId"]
        )

        carta = muestra.loc[
            muestra["CartaId"]
            .astype(int)
            == carta_id
        ].iloc[0]

        avanzadas = (
            extraer_metricas_avanzadas(
                resultado=resultado,
                carta=carta,
            )
        )

        avanzadas[
            "CartaId"
        ] = carta_id

        filas_avanzadas.append(
            avanzadas
        )


    metricas_avanzadas = pd.DataFrame(
        filas_avanzadas
    )


    columnas_avanzadas_previas = [
        columna
        for columna in metricas_avanzadas.columns
        if columna != "CartaId"
    ]


    metricas_cartas = (
        metricas_cartas
        .drop(
            columns=columnas_avanzadas_previas,
            errors="ignore",
        )
        .merge(
            metricas_avanzadas,
            on="CartaId",
            how="left",
        )
    )


    # ============================================================
    # VARIABLES DERIVADAS
    # ============================================================

    metricas_cartas[
        "Sumergencia_Relativa_pct"
    ] = (
        100
        * metricas_cartas[
            "Sumergencia_m"
        ]
        / base_diagnosticos.set_index(
            "CartaId"
        ).loc[
            metricas_cartas["CartaId"],
            "Profundidad_Bomba_m",
        ].to_numpy()
    )


    def calcular_angulos_carta_ideal(
        vertices_ideal,
    ):
        """
        Calcula los ángulos interiores inferiores de los
        laterales izquierdo y derecho de la carta ideal.

        Orden esperado:
            0 = superior izquierdo
            1 = superior derecho
            2 = inferior derecho
            3 = inferior izquierdo
        """
        resultado_vacio = {
            "izquierdo": np.nan,
            "derecho": np.nan,
        }

        if vertices_ideal is None:
            return resultado_vacio

        vertices = np.asarray(
            vertices_ideal,
            dtype=float,
        )

        if (
            vertices.shape != (4, 2)
            or not np.all(np.isfinite(vertices))
        ):
            return resultado_vacio

        sup_izq = vertices[0]
        sup_der = vertices[1]
        inf_der = vertices[2]
        inf_izq = vertices[3]

        ancho = abs(
            sup_der[0] - sup_izq[0]
        )

        altura = abs(
            sup_izq[1] - inf_izq[1]
        )

        if ancho <= 1e-9 or altura <= 1e-9:
            return resultado_vacio

        # Lateral izquierdo, desde el vértice inferior
        # hacia el superior.
        dx_izq = (
            sup_izq[0] - inf_izq[0]
        ) / ancho

        dy_izq = (
            sup_izq[1] - inf_izq[1]
        ) / altura

        angulo_izquierdo = np.degrees(
            np.arctan2(
                dy_izq,
                dx_izq,
            )
        )

        # Lateral derecho, desde el vértice inferior
        # hacia el superior.
        dx_der = (
            sup_der[0] - inf_der[0]
        ) / ancho

        dy_der = (
            sup_der[1] - inf_der[1]
        ) / altura

        orientacion_derecha = np.degrees(
            np.arctan2(
                dy_der,
                dx_der,
            )
        )

        # Ángulo interior medido respecto de la horizontal
        # inferior orientada hacia la izquierda.
        angulo_derecho = (
            180.0 - orientacion_derecha
        )

        return {
            "izquierdo": float(
                angulo_izquierdo
            ),
            "derecho": float(
                angulo_derecho
            ),
        }

    # ============================================================
    # APLICAR REGLAS
    # ============================================================

    def medir_golpe_bomba_izquierdo(
        posicion,
        carga,
        carga_inferior,
        fraccion_extremo_izquierdo=0.18,
        profundidad_min_pct=0.10,
    ):
        """
        Busca una excursión breve bajo la horizontal inferior,
        exclusivamente en el extremo izquierdo de la carta.

        El golpe queda caracterizado por:
        - profundidad respecto de la altura útil de la carta;
        - ancho respecto de la carrera;
        - ubicación del mínimo;
        - cantidad de puntos involucrados.
        """
        x = np.asarray(posicion, dtype=float)
        y = np.asarray(carga, dtype=float)

        validos = np.isfinite(x) & np.isfinite(y)
        x = x[validos]
        y = y[validos]

        if len(x) < 8:
            return {
                "Profundidad_Golpe_Inferior_pct": 0.0,
                "Ancho_Golpe_Inferior_pct": 0.0,
                "Posicion_Minimo_Golpe_pct": np.nan,
                "Puntos_Golpe_Inferior": 0,
                "Golpe_Localizado_Izquierda": False,
            }

        x_min = np.min(x)
        x_max = np.max(x)
        rango_x = max(x_max - x_min, 1e-9)

        altura_carta = max(
            np.nanpercentile(y, 95)
            - np.nanpercentile(y, 5),
            1e-9,
        )

        posicion_relativa = (x - x_min) / rango_x

        # El golpe solamente puede aparecer muy a la izquierda.
        zona_izquierda = (
            posicion_relativa <= fraccion_extremo_izquierdo
        )

        profundidad = (
            carga_inferior - y
        ) / altura_carta

        puntos_golpe = (
            zona_izquierda
            & (profundidad >= profundidad_min_pct)
        )

        indices = np.flatnonzero(puntos_golpe)

        if len(indices) == 0:
            return {
                "Profundidad_Golpe_Inferior_pct": 0.0,
                "Ancho_Golpe_Inferior_pct": 0.0,
                "Posicion_Minimo_Golpe_pct": np.nan,
                "Puntos_Golpe_Inferior": 0,
                "Golpe_Localizado_Izquierda": False,
            }

        indice_minimo = indices[
            np.argmax(profundidad[indices])
        ]

        profundidad_max_pct = float(
            100 * profundidad[indice_minimo]
        )

        ancho_pct = float(
            100
            * (
                np.max(x[indices])
                - np.min(x[indices])
            )
            / rango_x
        )

        posicion_minimo_pct = float(
            100 * posicion_relativa[indice_minimo]
        )

        # Debe ser profundo, estrecho y estar realmente en el extremo.
        localizado = bool(
            profundidad_max_pct >= 10.5
            and ancho_pct <= 18
            and posicion_minimo_pct <= 15
            and len(indices) <= max(8, int(0.25 * len(x)))
        )

        return {
            "Profundidad_Golpe_Inferior_pct":
                profundidad_max_pct,

            "Ancho_Golpe_Inferior_pct":
                ancho_pct,

            "Posicion_Minimo_Golpe_pct":
                posicion_minimo_pct,

            "Puntos_Golpe_Inferior":
                int(len(indices)),

            "Golpe_Localizado_Izquierda":
                localizado,
        }

    def detectar_rulo_golpe_bomba_izquierdo(posicion, carga):
        """Detecta un rulo angosto en el extremo inferior izquierdo.

        Complementa el criterio de profundidad: algunos impactos vuelven
        sobre sí mismos sin caer suficientemente bajo la horizontal de
        referencia. Se exige una reversión lateral real, angosta, localizada
        y con transferencia apreciable de carga para no confundir el ruido
        casi vertical de cartas normales.
        """
        x = np.asarray(posicion, dtype=float)
        y = np.asarray(carga, dtype=float)
        validos = np.isfinite(x) & np.isfinite(y)
        x, y = x[validos], y[validos]
        if len(x) < 7:
            return False
        rango_x_local = max(float(np.ptp(x)), 1e-9)
        rango_y_local = max(float(np.ptp(y)), 1e-9)
        x_min_local = float(np.min(x))
        xn = (x - x_min_local) / rango_x_local
        yn = (y - float(np.min(y))) / rango_y_local
        for inicio in range(len(x) - 4):
            for fin in range(inicio + 4, min(len(x), inicio + 10)):
                xx = xn[inicio:fin + 1]
                yy = yn[inicio:fin + 1]
                if np.max(xx) > 0.14 or np.ptp(xx) > 0.075:
                    continue
                dx = np.diff(xx)
                dx_significativo = dx[np.abs(dx) >= 0.004]
                if len(dx_significativo) < 2:
                    continue
                reversion = bool(
                    np.any(dx_significativo > 0)
                    and np.any(dx_significativo < 0)
                )
                retorno = abs(float(xx[-1] - xx[0])) <= 0.035
                transferencia = float(np.ptp(yy)) >= 0.17
                zona_inferior = float(np.max(yy)) <= 0.58
                if reversion and retorno and transferencia and zona_inferior:
                    return True
        return False


    variables_operativas = (
        muestra[
            [
                "CartaId",
                "Torque_Reductor_pct",
                "Carga_Estructural_pct",
            ]
        ]
        .drop_duplicates(subset=["CartaId"])
    )

    metricas_cartas = metricas_cartas.drop(
        columns=["Torque_Reductor_pct", "Carga_Estructural_pct"],
        errors="ignore",
    ).merge(
        variables_operativas,
        on="CartaId",
        how="left",
    )

    filas_diagnosticos = []

    # Área encerrada respecto del rectángulo envolvente de la carta.
    # Por debajo de este valor, ambas carreras recorren prácticamente
    # la misma trayectoria y se considera posible falta de trabajo.
    UMBRAL_COMPACIDAD_SIN_TRABAJO = 0.16
    UMBRAL_APERTURA_CENTRAL_BLOQUEO = 0.24

    def medir_apertura_central(resultado):
        """
        Apertura mediana entre carreras en el 20--85 % del recorrido.

        Se excluye deliberadamente el extremo izquierdo para que un golpe de
        bomba profundo no infle artificialmente el trabajo hidráulico.
        """
        try:
            asc = resultado["Ascendente"]
            desc = resultado["Descendente"]
            xa = np.asarray(asc["posicion"], dtype=float)
            ya = np.asarray(asc["carga"], dtype=float)
            xd = np.asarray(desc["posicion"], dtype=float)
            yd = np.asarray(desc["carga"], dtype=float)
            x_total = np.concatenate([xa, xd])
            y_total = np.concatenate([ya, yd])
            rango_x = float(np.ptp(x_total))
            rango_y = float(np.ptp(y_total))
            if rango_x <= 1e-9 or rango_y <= 1e-9:
                return np.nan

            x_min = float(np.nanmin(x_total))
            grilla = x_min + rango_x * np.linspace(0.20, 0.85, 40)

            def interpolar_rama(x, y):
                orden = np.argsort(x)
                x_ordenado = x[orden]
                y_ordenado = y[orden]
                x_unico, indices = np.unique(x_ordenado, return_index=True)
                y_unico = y_ordenado[indices]
                if len(x_unico) < 2:
                    return np.full_like(grilla, np.nan)
                return np.interp(grilla, x_unico, y_unico)

            apertura = np.abs(
                interpolar_rama(xa, ya) - interpolar_rama(xd, yd)
            )
            return float(np.nanmedian(apertura) / rango_y)
        except Exception:
            return np.nan

    def medir_cierre_viajera_tardio(resultado):
        """
        Mide cuánto demora la ascendente en transferirse desde la carga
        inferior hacia la superior al comienzo de la carrera.
        """
        salida = {
            "Carga_Inicial_Asc_Relativa": np.nan,
            "Posicion_Cierre_50_pct": np.nan,
            "Posicion_Cierre_80_pct": np.nan,
            "Separacion_Ramas_Inicial_pct_gap": np.nan,
            "Extension_Ramas_Juntas_pct": np.nan,
        }
        try:
            asc = resultado["Ascendente"]
            desc = resultado["Descendente"]
            x = np.asarray(asc["posicion"], dtype=float)
            y = np.asarray(asc["carga"], dtype=float)
            x_desc = np.asarray(desc["posicion"], dtype=float)
            y_desc = np.asarray(desc["carga"], dtype=float)
            carga_inf = float(resultado["Carga_Desc_Geometrica"])
            carga_sup = float(resultado["Carga_Asc_Geometrica"])
            rango_x = float(np.ptp(x))
            gap = carga_sup - carga_inf
            if rango_x <= 1e-9 or gap <= 1e-9:
                return salida

            progreso = 100.0 * (x - np.nanmin(x)) / rango_x
            carga_rel = (y - carga_inf) / gap
            zona_inicial = progreso <= 5.0
            if np.any(zona_inicial):
                salida["Carga_Inicial_Asc_Relativa"] = float(
                    np.nanmedian(carga_rel[zona_inicial])
                )

            for nivel, clave in (
                (0.50, "Posicion_Cierre_50_pct"),
                (0.80, "Posicion_Cierre_80_pct"),
            ):
                candidatos = np.flatnonzero(carga_rel >= nivel)
                if len(candidatos):
                    salida[clave] = float(progreso[candidatos[0]])

            # Un cierre tardío verdadero exige que las dos ramas sigan
            # juntas al comienzo. Una pared izquierda inclinada, por sí
            # sola, no alcanza para establecer el diagnóstico.
            def preparar_rama(x_rama, y_rama):
                tabla = pd.DataFrame({
                    "x": np.asarray(x_rama, dtype=float),
                    "y": np.asarray(y_rama, dtype=float),
                })
                tabla = tabla.replace(
                    [np.inf, -np.inf],
                    np.nan,
                ).dropna()
                tabla = (
                    tabla.groupby("x", as_index=False)["y"]
                    .median()
                    .sort_values("x")
                )
                return (
                    tabla["x"].to_numpy(dtype=float),
                    tabla["y"].to_numpy(dtype=float),
                )

            xa, ya = preparar_rama(x, y)
            xd, yd = preparar_rama(x_desc, y_desc)

            if len(xa) >= 2 and len(xd) >= 2:
                x_min_total = float(
                    min(np.nanmin(xa), np.nanmin(xd))
                )
                x_max_total = float(
                    max(np.nanmax(xa), np.nanmax(xd))
                )
                recorrido_total = x_max_total - x_min_total
                x_inicio = float(
                    max(np.nanmin(xa), np.nanmin(xd))
                )
                x_fin_solape = float(
                    min(np.nanmax(xa), np.nanmax(xd))
                )
                x_fin_analisis = min(
                    x_fin_solape,
                    x_min_total + 0.25 * recorrido_total,
                )

                if (
                    recorrido_total > 1e-9
                    and x_fin_analisis > x_inicio
                ):
                    grilla = np.linspace(
                        x_inicio,
                        x_fin_analisis,
                        51,
                    )
                    asc_interp = np.interp(grilla, xa, ya)
                    desc_interp = np.interp(grilla, xd, yd)
                    separacion = (
                        np.abs(asc_interp - desc_interp) / gap
                    )
                    separacion_suave = (
                        pd.Series(separacion)
                        .rolling(
                            window=3,
                            center=True,
                            min_periods=1,
                        )
                        .median()
                        .to_numpy(dtype=float)
                    )
                    progreso_comun = (
                        100.0
                        * (grilla - x_min_total)
                        / recorrido_total
                    )

                    zona_inicial_comun = progreso_comun <= 8.0
                    if np.any(zona_inicial_comun):
                        salida[
                            "Separacion_Ramas_Inicial_pct_gap"
                        ] = float(
                            100.0
                            * np.nanmedian(
                                separacion_suave[
                                    zona_inicial_comun
                                ]
                            )
                        )

                    # El tramo termina cuando tres muestras seguidas
                    # superan 12 % de la separación entre horizontales.
                    fuera = separacion_suave > 0.12
                    corte = len(fuera)
                    for indice in range(
                        0,
                        max(len(fuera) - 2, 0),
                    ):
                        if np.all(fuera[indice:indice + 3]):
                            corte = indice
                            break

                    if corte == 0:
                        extension_juntas = 0.0
                    elif corte < len(progreso_comun):
                        extension_juntas = float(
                            progreso_comun[corte - 1]
                        )
                    else:
                        extension_juntas = float(
                            progreso_comun[-1]
                        )

                    salida[
                        "Extension_Ramas_Juntas_pct"
                    ] = extension_juntas
        except Exception:
            pass
        return salida


    for _, metrica in metricas_cartas.iterrows():
        carta_id = int(
            metrica["CartaId"]
        )

        resultado = base_diagnosticos.loc[
            base_diagnosticos[
                "CartaId"
            ].astype(int)
            == carta_id
        ].iloc[0]

        angulos_ideal = calcular_angulos_carta_ideal(
        resultado["Vertices_Ideal"]
        )

        angulo_ideal_izquierdo = (
            angulos_ideal["izquierdo"]
        )

        angulo_ideal_derecho = (
            angulos_ideal["derecho"]
        )

        alertas = []
        evidencias = []

        torque_reductor_pct = pd.to_numeric(
            metrica.get("Torque_Reductor_pct", np.nan),
            errors="coerce",
        )
        carga_estructural_pct = pd.to_numeric(
            metrica.get("Carga_Estructural_pct", np.nan),
            errors="coerce",
        )

        exceso_torque = bool(
            np.isfinite(torque_reductor_pct)
            and torque_reductor_pct > 105.0
        )
        if exceso_torque:
            alertas.append("Exceso de torque")
            evidencias.append(
                f"Torque de caja reductora: {torque_reductor_pct:.1f} %"
            )

        exceso_carga_estructural = bool(
            np.isfinite(carga_estructural_pct)
            and carga_estructural_pct > 100.0
        )
        if exceso_carga_estructural:
            alertas.append("Exceso de carga estructural")
            evidencias.append(
                f"Carga estructural en la viga: {carga_estructural_pct:.1f} %"
            )

        # --------------------------------------------------------
        # 1. INTEGRIDAD GEOMÉTRICA Y SIN TRABAJO DE BOMBA
        # --------------------------------------------------------

        carta_no_valida = bool(
            resultado.get(
                "Posible_Carta_No_Valida",
                False,
            )
        )

        if carta_no_valida:
            alertas.append(
                "Carta no válida - posible falla de medición o transmisión"
            )

            evidencias.extend(
                list(
                    resultado.get(
                        "Evidencias_Integridad",
                        [],
                    )
                )
            )

        compacidad_carta = pd.to_numeric(
            resultado.get(
                "Compacidad_Carta",
                np.nan,
            ),
            errors="coerce",
        )
        apertura_central_carta = medir_apertura_central(resultado)
        sumergencia_preliminar = pd.to_numeric(
            resultado.get(
                "Sumergencia_Relativa_SAM_Seleccionada_pct",
                np.nan,
            ),
            errors="coerce",
        )

        sin_trabajo_por_area = bool(
            not carta_no_valida
            and np.isfinite(compacidad_carta)
            and compacidad_carta
                <= UMBRAL_COMPACIDAD_SIN_TRABAJO
        )
        bloqueo_por_umbral_principal = bool(
            not carta_no_valida
            and np.isfinite(apertura_central_carta)
            and apertura_central_carta <= UMBRAL_APERTURA_CENTRAL_BLOQUEO
            and (
                (
                    np.isfinite(sumergencia_preliminar)
                    and sumergencia_preliminar < 0.0
                )
                or (
                    np.isfinite(compacidad_carta)
                    and compacidad_carta <= 0.25
                )
            )
        )

        # Banda fronteriza: evita que cartas prácticamente iguales
        # alternen entre bloqueo y compresión por unas décimas en el
        # umbral de apertura. Se exige simultáneamente poca apertura
        # central y una carta globalmente compacta; no se amplía el
        # umbral de apertura de forma indiscriminada.
        bloqueo_gas_fronterizo = bool(
            not carta_no_valida
            and np.isfinite(apertura_central_carta)
            and np.isfinite(compacidad_carta)
            and apertura_central_carta <= 0.27
            and compacidad_carta <= 0.28
        )

        bloqueo_gas_probable = bool(
            bloqueo_por_umbral_principal
            or bloqueo_gas_fronterizo
        )

        sin_trabajo = bool(
            not carta_no_valida
            and (
                resultado[
                    "Posible_Sin_Trabajo_Bomba"
                ]
                or sin_trabajo_por_area
                or bloqueo_gas_probable
            )
        )

        if sin_trabajo:
            alertas.append(
                "Posible sin trabajo de bomba"
            )

            if bloqueo_gas_probable:
                evidencias.append(
                    "Apertura central muy baja, excluyendo el impacto "
                    "izquierdo: probable bloqueo por gas "
                    f"({100 * apertura_central_carta:.1f} %)"
                )
            elif sin_trabajo_por_area:
                evidencias.append(
                    "Área encerrada muy pequeña respecto del "
                    "rectángulo envolvente: "
                    f"{100 * compacidad_carta:.1f} %"
                )
            else:
                evidencias.append(
                    "No se identificaron horizontales confiables"
                )

        # Las siguientes reglas necesitan carta ideal válida.
        horizontales_ok = (
            not carta_no_valida
            and
            resultado[
                "Estado_Horizontales"
            ]
            == "HORIZONTALES_OK"
        )

        friccion_detectada_geometricamente = bool(
            resultado.get(
                "Friccion_Elevada_Geometrica",
                False,
            )
        )
        correccion_friccion_aplicada = bool(
            resultado.get(
                "Correccion_Friccion_Aplicada",
                False,
            )
        )
        curvatura_inferior_friccion = pd.to_numeric(
            resultado.get(
                "Curvatura_Inferior_Friccion",
                np.nan,
            ),
            errors="coerce",
        )

        # Si el patrón preliminar no produce una corrección coherente
        # del gap, se exige una cubeta inferior inequívocamente marcada.
        # Esto evita etiquetar como fricción una ondulación moderada
        # asociada a una transferencia derecha de admisión incompleta.
        respaldo_friccion_sin_correccion = bool(
            np.isfinite(curvatura_inferior_friccion)
            and curvatura_inferior_friccion >= 0.80
        )

        friccion_geometrica_fuerte = bool(
            horizontales_ok
            and not sin_trabajo
            and friccion_detectada_geometricamente
            and (
                correccion_friccion_aplicada
                or respaldo_friccion_sin_correccion
            )
        )

        # La fricción por carta se asigna solamente cuando la geometría
        # cumple el detector fuerte. Los patrones suaves se evalúan luego
        # en contexto temporal, usando cartas vecinas del mismo pozo.
        friccion_elevada = friccion_geometrica_fuerte

        if friccion_elevada:
            alertas.append("Posible fricción elevada")
            if bool(
                resultado.get(
                    "Correccion_Friccion_Aplicada",
                    False,
                )
            ):
                evidencias.append(
                    "Geometría compatible con fricción; "
                    "horizontales corregidas hacia niveles "
                    "interiores robustos"
                )
            else:
                evidencias.append(
                    "Geometría compatible con fricción; la "
                    "corrección de horizontales no resultó material"
                )

        # Recuperar vacíos.
        vacio_si = metrica.get(
            "Area_Faltante_Superior_Izquierdo_pct",
            np.nan,
        )

        vacio_sd = metrica.get(
            "Area_Faltante_Superior_Derecho_pct",
            np.nan,
        )

        vacio_ii = metrica.get(
            "Area_Faltante_Inferior_Izquierdo_pct",
            np.nan,
        )

        vacio_id = metrica.get(
            "Area_Faltante_Inferior_Derecho_pct",
            np.nan,
        )

        angulo_izq = metrica[
            "Angulo_Lateral_Izquierdo_deg"
        ]

        angulo_der = metrica[
            "Angulo_Lateral_Derecho_deg"
        ]

        llenado_bruto = metrica[
            "Llenado_Calculado_pct"
        ]
        llenado = metrica[
            "Area_Dentro_Carta_Ideal_pct"
        ]

        # --------------------------------------------------------
        # 2. PÉRDIDA EN VÁLVULA VIAJERA
        # --------------------------------------------------------

        metricas_cierre_viajera = medir_cierre_viajera_tardio(resultado)
        carga_inicial_asc_relativa = metricas_cierre_viajera[
            "Carga_Inicial_Asc_Relativa"
        ]
        posicion_cierre_50 = metricas_cierre_viajera[
            "Posicion_Cierre_50_pct"
        ]
        posicion_cierre_80 = metricas_cierre_viajera[
            "Posicion_Cierre_80_pct"
        ]
        separacion_ramas_inicial = metricas_cierre_viajera[
            "Separacion_Ramas_Inicial_pct_gap"
        ]
        extension_ramas_juntas = metricas_cierre_viajera[
            "Extension_Ramas_Juntas_pct"
        ]
        cierre_tardio_viajera = bool(
            horizontales_ok
            and not sin_trabajo
            and np.isfinite(vacio_si)
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and np.isfinite(carga_inicial_asc_relativa)
            and np.isfinite(posicion_cierre_50)
            and np.isfinite(posicion_cierre_80)
            and np.isfinite(separacion_ramas_inicial)
            and np.isfinite(extension_ramas_juntas)
            and vacio_si >= 12.0
            and vacio_sd < 8.0
            and vacio_id < 20.0
            and carga_inicial_asc_relativa <= 0.15
            and posicion_cierre_50 >= 8.0
            and posicion_cierre_80 >= 12.0
            and separacion_ramas_inicial <= 12.0
            and extension_ramas_juntas >= 8.0
        )
        if cierre_tardio_viajera:
            alertas.append(
                "Posible cierre tardío de válvula viajera"
            )
            evidencias.append(
                "La ascendente conserva inicialmente la carga inferior "
                "y transfiere tardÃ­amente el peso de fluido"
            )

        vacios_superiores_valvula = bool(
            np.isfinite(vacio_si)
            and np.isfinite(vacio_sd)
            and (
                (
                    vacio_si
                        >= UMBRAL_VACIO_SUP_IZQ_VALVULA
                    and vacio_sd
                        >= UMBRAL_VACIO_SUP_DER_VALVULA
                )
                or (
                    vacio_si >= 8.0
                    and vacio_sd >= 18.0
                    and (vacio_si + vacio_sd) >= 30.0
                )
            )
        )

        perdida_valvula = (
            horizontales_ok
            and not sin_trabajo
            and not cierre_tardio_viajera
            and np.isfinite(vacio_si)
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and np.isfinite(
                angulo_ideal_izquierdo
            )
            and np.isfinite(
                angulo_ideal_derecho
            )
            and vacios_superiores_valvula
            and vacio_id
                < UMBRAL_VACIO_INF_DER_VALVULA
            and angulo_ideal_izquierdo < 90.0
            and angulo_ideal_derecho < 90.0
        )

        if perdida_valvula:
            alertas.append(
                "Posible pérdida en válvula viajera"
            )

            evidencias.append(
                "Vacíos superiores exteriores y laterales inclinados"
            )

        # --------------------------------------------------------
        # LLENADO AJUSTADO PARA VÁLVULA VIAJERA
        # --------------------------------------------------------

        # Cada cuadrante representa el 25 % del área ideal.
        # Si hay pérdida de viajera, los vacíos superiores
        # no se descuentan del llenado operativo.
        if (
            perdida_valvula
            and np.isfinite(llenado)
        ):
            llenado_operativo = (
                llenado
                + 0.25 * vacio_si
                + 0.25 * vacio_sd
            )

            llenado_operativo = min(
                llenado_operativo,
                100.0,
            )

        else:
            llenado_operativo = llenado

        # La carrera geometrica es el recorrido total de posicion. La carrera
        # efectiva descuenta la fraccion de llenado exactamente una vez.
        carrera_geometrica_calculada_pulg = pd.to_numeric(
            resultado.get(
                "Carrera_Geometrica_Fondo_Calculada_pulg",
                resultado.get(
                    "Carrera_Total_Fondo_Calculada_pulg",
                    np.nan,
                ),
            ),
            errors="coerce",
        )
        factor_llenado_operativo = (
            float(np.clip(llenado_operativo / 100.0, 0.0, 1.0))
            if np.isfinite(llenado_operativo)
            else np.nan
        )
        carrera_efectiva_calculada_pulg = (
            carrera_geometrica_calculada_pulg * factor_llenado_operativo
            if (
                np.isfinite(carrera_geometrica_calculada_pulg)
                and np.isfinite(factor_llenado_operativo)
            )
            else np.nan
        )
        desplazamiento_geometrico_calculado_m3_d = pd.to_numeric(
            resultado.get(
                "Desplazamiento_Bruto_Geometrico_Calculado_m3_d",
                resultado.get(
                    "Desplazamiento_Bruto_Total_Calculado_m3_d",
                    np.nan,
                ),
            ),
            errors="coerce",
        )
        desplazamiento_efectivo_calculado_m3_d = (
            desplazamiento_geometrico_calculado_m3_d
            * factor_llenado_operativo
            if (
                np.isfinite(desplazamiento_geometrico_calculado_m3_d)
                and np.isfinite(factor_llenado_operativo)
            )
            else np.nan
        )
        desplazamiento_total_calculado_m3_d = pd.to_numeric(
            resultado.get(
                "Desplazamiento_Bruto_Total_Calculado_m3_d",
                np.nan,
            ),
            errors="coerce",
        )
        escurrimiento_calculado_m3_d = (
            max(
                desplazamiento_total_calculado_m3_d
                - desplazamiento_efectivo_calculado_m3_d,
                0.0,
            )
            if (
                np.isfinite(desplazamiento_total_calculado_m3_d)
                and np.isfinite(desplazamiento_efectivo_calculado_m3_d)
            )
            else np.nan
        )
        llenado_implicito_carrera_efectiva_pct = (
            100.0
            * desplazamiento_efectivo_calculado_m3_d
            / desplazamiento_total_calculado_m3_d
            if (
                np.isfinite(desplazamiento_efectivo_calculado_m3_d)
                and np.isfinite(desplazamiento_total_calculado_m3_d)
                and desplazamiento_total_calculado_m3_d > 0
            )
            else np.nan
        )

        # --------------------------------------------------------
        # 3. GOLPE DE FLUIDO / COMPRESIÓN DE GAS
        # --------------------------------------------------------
        # --------------------------------------------------------
        # 3. GOLPE DE FLUIDO / COMPRESIÓN DE GAS
        # --------------------------------------------------------

        pendiente_transferencia = metrica[
            "Pendiente_Transferencia_Derecha"
        ]

        curvatura_transferencia = metrica[
            "Curvatura_Transferencia_Derecha"
        ]

        # Inicialización independiente de diagnósticos.
        golpe_fluido = False
        compresion_gas = False
        compresion_gas_suave = False
        severidad_admision = "NO_APLICA"

        # --------------------------------------------------------
        # VACÍO DERECHO MARCADO
        # --------------------------------------------------------
        vacio_derecho_marcado = (
            horizontales_ok
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and np.isfinite(llenado)
            and vacio_sd >= 20.0
            and vacio_id >= 30.0
            and llenado < 90.0
        )

        # --------------------------------------------------------
        # VACÍO DERECHO SUAVE
        # --------------------------------------------------------
        # Umbrales preliminares calibrados con las cartas
        # 26163920 y 26163934.
        vacio_derecho_suave = (
            horizontales_ok
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and np.isfinite(llenado)
            and vacio_sd >= UMBRAL_VACIO_SUP_DER_ADMISION_SUAVE
            and vacio_id >= 10.0
            and llenado < 92.0
        )

        despegue_inferior_derecho = metrica.get(
            "Despegue_Inferior_Derecho_pct",
            np.nan,
        )
        extension_despegue_inferior = metrica.get(
            "Extension_Despegue_Inferior_Derecho_pct",
            np.nan,
        )
        transferencia_inferior_sostenida = bool(
            metrica.get(
                "Transferencia_Inferior_Sostenida",
                False,
            )
        )

        hay_indicio_admision = bool(
            transferencia_inferior_sostenida
            and (
                vacio_derecho_marcado
                or vacio_derecho_suave
            )
        )

        # --------------------------------------------------------
        # TIPO DE TRANSFERENCIA
        # --------------------------------------------------------
        # Para considerar una transferencia abrupta exigimos
        # simultáneamente pendiente y curvatura elevadas.
        # Una pendiente alta aislada puede aparecer dentro de
        # una transición globalmente redondeada.
        ancho_transferencia_20_80 = metrica.get(
            "Ancho_Transferencia_20_80_pct", np.nan
        )
        inicio_transferencia_derecha = metrica.get(
            "Inicio_Transferencia_Derecha_pct", np.nan
        )

        if (
            np.isfinite(ancho_transferencia_20_80)
            and np.isfinite(inicio_transferencia_derecha)
        ):
            transferencia_desplazada = bool(
                inicio_transferencia_derecha < 97.0
            )
            transferencia_abrupta = bool(
                transferencia_desplazada
                and ancho_transferencia_20_80 <= 22.0
            )
            transferencia_progresiva = bool(
                transferencia_desplazada
                and ancho_transferencia_20_80 > 22.0
            )
        else:
            transferencia_abrupta = bool(
                np.isfinite(pendiente_transferencia)
                and np.isfinite(curvatura_transferencia)
                and pendiente_transferencia >= 4.0
                and curvatura_transferencia >= 18.0
            )
            transferencia_progresiva = bool(
                np.isfinite(pendiente_transferencia)
                and np.isfinite(curvatura_transferencia)
                and not transferencia_abrupta
            )

        # Respaldo conservador cuando la transferencia derecha no
        # pudo medirse. No alcanza con un vacío inferior aislado:
        # se exige también vacío superior, llenado incompleto y
        # horizontales confiables.
        transferencia_no_mensurable = bool(
            not np.isfinite(ancho_transferencia_20_80)
            and not np.isfinite(inicio_transferencia_derecha)
            and not np.isfinite(pendiente_transferencia)
            and not np.isfinite(curvatura_transferencia)
        )

        transferencia_progresiva_inferida = bool(
            transferencia_no_mensurable
            and horizontales_ok
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and np.isfinite(llenado)
            and vacio_sd >= UMBRAL_VACIO_SUP_DER_ADMISION_SUAVE
            and vacio_id >= 20.0
            and llenado < 92.0
        )

        # Algunas cartas devuelven el inicio apenas por encima
        # del 100 % por la normalización geométrica. Se admite como
        # compresión suave solamente si la transición tiene un ancho
        # significativo y su forma es suave. Esto excluye saltos
        # degenerados muy angostos en el extremo derecho.
        transferencia_progresiva_tardia = bool(
            horizontales_ok
            and np.isfinite(inicio_transferencia_derecha)
            and np.isfinite(ancho_transferencia_20_80)
            and np.isfinite(pendiente_transferencia)
            and np.isfinite(curvatura_transferencia)
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and np.isfinite(llenado)
            and inicio_transferencia_derecha >= 97.0
            and ancho_transferencia_20_80 >= 8.0
            and pendiente_transferencia < 4.0
            and vacio_sd >= UMBRAL_VACIO_SUP_DER_ADMISION_SUAVE
            and vacio_id >= 25.0
            and llenado < 92.0
        )

        # Respaldo para cartas fronterizas de alto llenado. Evita que
        # pequeñas variaciones alrededor del 90 % conviertan una misma
        # familia geométrica en "bien explotado". Se exige un vacío
        # inferior derecho apreciable y una transferencia desplazada y
        # relativamente rápida; no alcanza con el llenado aislado.
        golpe_fluido_fronterizo = bool(
            horizontales_ok
            and not sin_trabajo
            and np.isfinite(llenado)
            and np.isfinite(vacio_id)
            and np.isfinite(inicio_transferencia_derecha)
            and np.isfinite(ancho_transferencia_20_80)
            and 85.0 <= llenado <= 92.0
            and vacio_id >= 15.0
            and inicio_transferencia_derecha < 97.0
            and ancho_transferencia_20_80 <= 28.0
        )

        if golpe_fluido_fronterizo:
            hay_indicio_admision = True
            transferencia_abrupta = True

        if transferencia_progresiva_inferida:
            transferencia_progresiva = True
        if transferencia_progresiva_tardia:
            transferencia_progresiva = True

        if hay_indicio_admision:
            if transferencia_abrupta:
                golpe_fluido = True
                severidad_admision = (
                    "MODERADA"
                    if vacio_derecho_marcado
                    else "LEVE"
                )

                alertas.append(
                    "Posible golpe de fluido"
                )

                evidencias.append(
                    "Vacíos derechos y transferencia abrupta"
                )

            elif transferencia_progresiva:
                compresion_gas = True

                # Diferenciar la intensidad sin crear una clase
                # principal completamente distinta.
                # Diferenciar la intensidad de la compresión.
                compresion_gas_suave = bool(
                    vacio_derecho_suave
                    and not vacio_derecho_marcado
                )
                severidad_admision = (
                    "LEVE"
                    if compresion_gas_suave
                    else "MODERADA"
                )

                alertas.append(
                    "Posible compresión/interferencia de gas"
                )

                if compresion_gas_suave:
                    if transferencia_progresiva_tardia:
                        evidencias.append(
                            "Vacíos derechos moderados y "
                            "transferencia tardía de forma suave"
                        )
                    elif transferencia_progresiva_inferida:
                        evidencias.append(
                            "Vacíos derechos moderados; "
                            "transferencia no mensurable e "
                            "inferida como progresiva"
                        )
                    else:
                        evidencias.append(
                            "Vacíos derechos moderados y "
                            "transferencia progresiva"
                        )

                else:
                    evidencias.append(
                        "Vacíos derechos importantes y "
                        "transferencia progresiva"
                    )

        # --------------------------------------------------------
        # 4. GOLPE DE BOMBA
        # --------------------------------------------------------
        x_asc_golpe = np.asarray(
            resultado["Ascendente"]["posicion"], dtype=float
        )
        y_asc_golpe = np.asarray(
            resultado["Ascendente"]["carga"], dtype=float
        )
        x_desc_golpe = np.asarray(
            resultado["Descendente"]["posicion"], dtype=float
        )
        y_desc_golpe = np.asarray(
            resultado["Descendente"]["carga"], dtype=float
        )
        x_min_golpe = float(min(np.min(x_asc_golpe), np.min(x_desc_golpe)))
        x_max_golpe = float(max(np.max(x_asc_golpe), np.max(x_desc_golpe)))
        rango_x_golpe = max(x_max_golpe - x_min_golpe, 1e-9)
        mascara_central_inferior = (
            (x_desc_golpe >= x_min_golpe + 0.18 * rango_x_golpe)
            & (x_desc_golpe <= x_min_golpe + 0.72 * rango_x_golpe)
        )
        carga_inferior_geometrica = pd.to_numeric(
            resultado.get("Carga_Desc_Geometrica"), errors="coerce"
        )
        carga_inferior_central = (
            float(np.median(y_desc_golpe[mascara_central_inferior]))
            if np.count_nonzero(mascara_central_inferior) >= 4
            else np.nan
        )
        referencias_inferiores = [
            valor for valor in (
                carga_inferior_geometrica,
                carga_inferior_central,
            )
            if np.isfinite(valor)
        ]
        carga_inferior_golpe_robusta = (
            float(max(referencias_inferiores))
            if referencias_inferiores
            else float(np.nanmedian(y_desc_golpe))
        )

        metricas_golpe = medir_golpe_bomba_izquierdo(
            # Se usan ambas ramas porque el punto de corte entre carreras puede
            # dejar los puntos del impacto en cualquiera de los dos arreglos.
            # La propia función restringe la búsqueda al extremo izquierdo.
            posicion=np.concatenate([
                resultado["Ascendente"]["posicion"],
                resultado["Descendente"]["posicion"],
            ]),
            carga=np.concatenate([
                resultado["Ascendente"]["carga"],
                resultado["Descendente"]["carga"],
            ]),
            carga_inferior=float(carga_inferior_geometrica),
        )
        metricas_golpe_fronterizo = medir_golpe_bomba_izquierdo(
            posicion=np.concatenate([x_asc_golpe, x_desc_golpe]),
            carga=np.concatenate([y_asc_golpe, y_desc_golpe]),
            carga_inferior=carga_inferior_golpe_robusta,
            profundidad_min_pct=0.075,
        )
        golpe_bomba_fronterizo_estrecho = bool(
            8.0 <= metricas_golpe_fronterizo[
                "Profundidad_Golpe_Inferior_pct"
            ] < 10.5
            and metricas_golpe_fronterizo[
                "Ancho_Golpe_Inferior_pct"
            ] <= 4.0
            and metricas_golpe_fronterizo[
                "Posicion_Minimo_Golpe_pct"
            ] <= 6.0
            and metricas_golpe_fronterizo[
                "Puntos_Golpe_Inferior"
            ] <= 4
        )
        if golpe_bomba_fronterizo_estrecho:
            metricas_golpe = metricas_golpe_fronterizo

        rulo_golpe_bomba = bool(
            detectar_rulo_golpe_bomba_izquierdo(
                resultado["Descendente"]["posicion"],
                resultado["Descendente"]["carga"],
            )
        )

        golpe_bomba = bool(
            horizontales_ok
            and (
                metricas_golpe["Golpe_Localizado_Izquierda"]
                or golpe_bomba_fronterizo_estrecho
                or (rulo_golpe_bomba and compresion_gas)
                or (
                    bloqueo_gas_probable
                    and metricas_golpe[
                        "Profundidad_Golpe_Inferior_pct"
                    ] >= 25.0
                    and metricas_golpe[
                        "Ancho_Golpe_Inferior_pct"
                    ] <= 20.0
                    and metricas_golpe[
                        "Posicion_Minimo_Golpe_pct"
                    ] <= 5.0
                )
            )
        )

        if golpe_bomba:
            alertas.append("Posible golpe de bomba")
            evidencias.append(
                "Excursión breve y profunda bajo la horizontal inferior, "
                "localizada en el extremo izquierdo de la descendente"
            )

            # El impacto de bomba contamina el codo azul izquierdo. Por
            # definicion del SAM Modificado, en esta familia la horizontal
            # inferior usa solamente el codo azul derecho. El punto izquierdo
            # se conserva para auditoria, pero no interviene en Fo ni PIP.
            resultado = resultado.copy()
            resultado["Regla_Inferior_SAM_Modificado"] = (
                "SOLO_CODO_AZUL_DERECHO"
            )
            resultado["Azul_Izquierdo_Incluido_SAM_Modificado"] = False

            # En una carta rectangular el impacto izquierdo no invalida los
            # otros laterales. Se recorren las ramas en su orden físico y se
            # toma la primera salida sostenida de cada vertical: arriba a la
            # izquierda para el rojo y abajo a la derecha para el único azul.
            x_asc_bomba = np.asarray(
                resultado["Ascendente"]["posicion"], dtype=float
            )
            y_asc_bomba = np.asarray(
                resultado["Ascendente"]["carga"], dtype=float
            )
            x_desc_bomba = np.asarray(
                resultado["Descendente"]["posicion"], dtype=float
            )
            y_desc_bomba = np.asarray(
                resultado["Descendente"]["carga"], dtype=float
            )
            x_min_bomba = float(min(np.min(x_asc_bomba), np.min(x_desc_bomba)))
            x_max_bomba = float(max(np.max(x_asc_bomba), np.max(x_desc_bomba)))
            rango_x_bomba = max(x_max_bomba - x_min_bomba, 1e-9)
            y_min_bomba = float(min(np.min(y_asc_bomba), np.min(y_desc_bomba)))
            y_max_bomba = float(max(np.max(y_asc_bomba), np.max(y_desc_bomba)))
            rango_y_bomba = max(y_max_bomba - y_min_bomba, 1e-9)

            # El rojo izquierdo se obtiene en la banda superior de la
            # transferencia izquierda. El objetivo de carga evita el rulo
            # inferior del impacto y la penalización por interioridad lo
            # mantiene sobre la vertical o pseudo vertical.
            indices_rojo_izq_bomba = np.flatnonzero(
                x_asc_bomba <= x_min_bomba + 0.16 * rango_x_bomba
            )
            rojo_izq_bomba = None
            if (
                len(indices_rojo_izq_bomba) >= 4
                and np.ptp(y_asc_bomba[indices_rojo_izq_bomba])
                >= 0.35 * rango_y_bomba
            ):
                objetivo_rojo_bomba = y_min_bomba + 0.75 * rango_y_bomba
                score_rojo_bomba = (
                    abs(
                        y_asc_bomba[indices_rojo_izq_bomba]
                        - objetivo_rojo_bomba
                    ) / rango_y_bomba
                    + 0.30 * (
                        x_asc_bomba[indices_rojo_izq_bomba] - x_min_bomba
                    ) / rango_x_bomba
                )
                indice_rojo_bomba = int(indices_rojo_izq_bomba[
                    np.argmin(score_rojo_bomba)
                ])
                rojo_izq_bomba = (
                    float(x_asc_bomba[indice_rojo_bomba]),
                    float(y_asc_bomba[indice_rojo_bomba]),
                )
            indices_rojo_der_bomba = np.flatnonzero(
                x_desc_bomba >= x_max_bomba - 0.16 * rango_x_bomba
            )
            rojo_der_bomba = None
            if (
                len(indices_rojo_der_bomba) >= 4
                and np.ptp(y_desc_bomba[indices_rojo_der_bomba])
                >= 0.35 * rango_y_bomba
            ):
                objetivo_rojo_bomba = y_min_bomba + 0.75 * rango_y_bomba
                score_rojo_der_bomba = (
                    abs(
                        y_desc_bomba[indices_rojo_der_bomba]
                        - objetivo_rojo_bomba
                    ) / rango_y_bomba
                    + 0.30 * (
                        x_max_bomba - x_desc_bomba[indices_rojo_der_bomba]
                    ) / rango_x_bomba
                )
                indice_rojo_der_bomba = int(indices_rojo_der_bomba[
                    np.argmin(score_rojo_der_bomba)
                ])
                rojo_der_bomba = (
                    float(x_desc_bomba[indice_rojo_der_bomba]),
                    float(y_desc_bomba[indice_rojo_der_bomba]),
                )

            def candidato_azul_derecho_bomba(x, y):
                """Selecciona un codo inferior sólo en el lateral derecho.

                La geometría manda: se exigen posición derecha, transferencia
                pseudo vertical sostenida y separación de ambas horizontales.
                Entre candidatos geométricamente válidos, la consistencia de
                sumergencia actúa únicamente como penalización suave. Así se
                evita rescatar el rulo izquierdo o mover un punto a una zona
                físicamente atractiva pero geométricamente incorrecta.
                """
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                if len(x) < 5:
                    return None

                superior_prueba = pd.to_numeric(
                    resultado.get("Carga_Superior_SAM_Seleccionada_lbf"),
                    errors="coerce",
                )
                profundidad_prueba = pd.to_numeric(
                    resultado.get("Profundidad_Bomba_m"), errors="coerce"
                )
                area_prueba = pd.to_numeric(
                    resultado.get("Area_Piston_SAM_pulg2"), errors="coerce"
                )
                gradiente_prueba = pd.to_numeric(
                    resultado.get("Gradiente_SAM_psi_m"), errors="coerce"
                )
                descarga_prueba = pd.to_numeric(
                    resultado.get("Presion_Descarga_Bomba_SAM_psi"),
                    errors="coerce",
                )
                casing_prueba = pd.to_numeric(
                    resultado.get("Presion_Casing_SAM_kg_cm2"),
                    errors="coerce",
                )
                llenado_prueba = pd.to_numeric(
                    llenado_operativo,
                    errors="coerce",
                )

                candidatos = []
                for indice in range(1, len(x) - 1):
                    posicion_rel = (
                        x[indice] - x_min_bomba
                    ) / rango_x_bomba
                    carga_rel = (
                        y[indice] - y_min_bomba
                    ) / rango_y_bomba

                    # Nunca se permite que el "azul derecho" provenga del
                    # rulo izquierdo ni de las pseudo horizontales.
                    if not (0.52 <= posicion_rel and 0.10 <= carga_rel <= 0.58):
                        continue

                    dx_rel = abs(x[indice + 1] - x[indice - 1]) / rango_x_bomba
                    dy_rel = abs(y[indice + 1] - y[indice - 1]) / rango_y_bomba
                    verticalidad = dx_rel / max(dy_rel, 1e-9)
                    if verticalidad > 0.55:
                        continue

                    # La rama debe seguir avanzando desde la derecha hacia la
                    # horizontal inferior; una inversión amplia suele ser una
                    # ondulación o un falso codo.
                    if x[indice + 1] > x[indice] + 0.012 * rango_x_bomba:
                        continue

                    penalizacion_fisica = 0.0
                    sumergencia_rel_candidata = np.nan
                    if np.isfinite([
                        superior_prueba,
                        profundidad_prueba,
                        area_prueba,
                        gradiente_prueba,
                        descarga_prueba,
                        casing_prueba,
                    ]).all() and area_prueba > 0 and gradiente_prueba > 0:
                        peso_prueba = superior_prueba - y[indice]
                        pip_prueba = descarga_prueba - peso_prueba / area_prueba
                        sumergencia_prueba = (
                            pip_prueba - casing_prueba * KG_CM2_A_PSI
                        ) / gradiente_prueba
                        sumergencia_rel_candidata = (
                            100.0 * sumergencia_prueba / profundidad_prueba
                        )
                        if np.isfinite(sumergencia_rel_candidata):
                            if golpe_fluido or compresion_gas:
                                if sumergencia_rel_candidata < 0.0:
                                    penalizacion_fisica = min(
                                        2.0,
                                        0.5 + abs(sumergencia_rel_candidata) / 15.0,
                                    )
                                elif sumergencia_rel_candidata > 15.0:
                                    penalizacion_fisica = min(
                                        1.5,
                                        (sumergencia_rel_candidata - 15.0) / 30.0,
                                    )
                            elif (
                                np.isfinite(llenado_prueba)
                                and llenado_prueba >= 95.0
                            ):
                                # Con llenado prácticamente completo, una
                                # sumergencia negativa es poco consistente y
                                # el rango 10–15% funciona como referencia
                                # suave. Nunca habilita un candidato que haya
                                # fallado los filtros geométricos anteriores.
                                if sumergencia_rel_candidata < 0.0:
                                    penalizacion_fisica = min(
                                        2.0,
                                        0.65
                                        + abs(sumergencia_rel_candidata) / 15.0,
                                    )
                                elif sumergencia_rel_candidata < 10.0:
                                    penalizacion_fisica = (
                                        10.0 - sumergencia_rel_candidata
                                    ) / 10.0
                                elif sumergencia_rel_candidata < 15.0:
                                    penalizacion_fisica = 0.20 * (
                                        15.0 - sumergencia_rel_candidata
                                    ) / 5.0
                            elif sumergencia_rel_candidata < 0.0:
                                penalizacion_fisica = min(
                                    2.0,
                                    abs(sumergencia_rel_candidata) / 20.0,
                                )

                    score = verticalidad + 0.65 * penalizacion_fisica
                    candidatos.append((
                        score,
                        verticalidad,
                        indice,
                        sumergencia_rel_candidata,
                    ))

                if not candidatos:
                    return None
                _, verticalidad, indice, sumergencia_rel_candidata = min(
                    candidatos, key=lambda item: (item[0], item[1])
                )
                return {
                    "posicion": float(x[indice]),
                    "carga": float(y[indice]),
                    "verticalidad": float(verticalidad),
                    "sumergencia_relativa_pct": float(
                        sumergencia_rel_candidata
                    ) if np.isfinite(sumergencia_rel_candidata) else np.nan,
                }

            def candidato_rojo_izquierdo_legado(x, y):
                inicio = int(np.argmin(
                    x + 0.03 * (y - y_min_bomba)
                    / rango_y_bomba * rango_x_bomba
                ))
                recorrido = np.arange(inicio, len(x))
                if len(recorrido) < 5:
                    recorrido = np.arange(inicio, -1, -1)
                desplazamiento = (
                    x[recorrido] - x_min_bomba
                ) / rango_x_bomba
                for k in range(1, len(recorrido) - 2):
                    futuras = desplazamiento[k:k + 3]
                    if (
                        futuras[0] >= 0.012
                        and np.count_nonzero(
                            np.diff(futuras) >= -0.004
                        ) >= 1
                        and futuras[-1] >= 0.025
                    ):
                        indice = int(recorrido[max(0, k - 1)])
                        return float(x[indice]), float(y[indice])
                return None

            rojo_izq_bomba_legado = candidato_rojo_izquierdo_legado(
                x_asc_bomba, y_asc_bomba
            )
            rojo_izq_actual = pd.to_numeric(
                resultado.get("Carga_Roja_Izquierda_SAM_Modificado_lbf"),
                errors="coerce",
            )
            azul_der_actual = pd.to_numeric(
                resultado.get("Carga_Azul_Derecha_SAM_Modificado_lbf"),
                errors="coerce",
            )
            superior_actual = pd.to_numeric(
                resultado.get("Carga_Superior_SAM_Seleccionada_lbf"),
                errors="coerce",
            )
            rojo_der_actual = pd.to_numeric(
                resultado.get("Carga_Roja_Derecha_SAM_Modificado_lbf"),
                errors="coerce",
            )
            aplicar_rojo_izq_banda = bool(
                rojo_izq_bomba is not None
                and np.isfinite(rojo_izq_actual)
                and rojo_izq_bomba[1]
                > rojo_izq_actual + 0.06 * rango_y_bomba
            )
            if not aplicar_rojo_izq_banda:
                aplicar_rojo_izq_legado = bool(
                    rojo_izq_bomba_legado is not None
                    and np.isfinite(rojo_izq_actual)
                    and rojo_izq_bomba_legado[1]
                    < rojo_izq_actual - 0.012 * rango_y_bomba
                    and rojo_izq_bomba_legado[1]
                    >= y_min_bomba + 0.55 * rango_y_bomba
                )
                if aplicar_rojo_izq_legado:
                    rojo_izq_bomba = rojo_izq_bomba_legado
            else:
                aplicar_rojo_izq_legado = False
            aplicar_rojo_izq_bomba = bool(
                aplicar_rojo_izq_banda or aplicar_rojo_izq_legado
            )
            if aplicar_rojo_izq_bomba:
                resultado["Posicion_Roja_Izquierda_SAM_Modificado_pulg"] = (
                    rojo_izq_bomba[0]
                )
                resultado["Carga_Roja_Izquierda_SAM_Modificado_lbf"] = (
                    rojo_izq_bomba[1]
                )
                if np.isfinite(rojo_der_actual):
                    resultado["Carga_Superior_SAM_Seleccionada_lbf"] = 0.5 * (
                        rojo_izq_bomba[1] + rojo_der_actual
                    )
            aplicar_rojo_der_bomba = bool(
                rojo_der_bomba is not None
                and np.isfinite(rojo_der_actual)
                and rojo_der_bomba[1]
                > rojo_der_actual + 0.06 * rango_y_bomba
            )
            if aplicar_rojo_der_bomba:
                resultado["Posicion_Roja_Derecha_SAM_Modificado_pulg"] = (
                    rojo_der_bomba[0]
                )
                resultado["Carga_Roja_Derecha_SAM_Modificado_lbf"] = (
                    rojo_der_bomba[1]
                )
                rojo_izq_para_media = pd.to_numeric(
                    resultado.get(
                        "Carga_Roja_Izquierda_SAM_Modificado_lbf"
                    ), errors="coerce"
                )
                if np.isfinite(rojo_izq_para_media):
                    resultado["Carga_Superior_SAM_Seleccionada_lbf"] = 0.5 * (
                        rojo_izq_para_media + rojo_der_bomba[1]
                    )
            # La validación física del azul usa la horizontal superior final,
            # una vez aplicadas las eventuales correcciones de ambos rojos.
            azul_der_bomba = candidato_azul_derecho_bomba(
                x_desc_bomba, y_desc_bomba
            )
            aplicar_azul_der_bomba = bool(
                azul_der_bomba is not None
                and np.isfinite(azul_der_actual)
            )
            if aplicar_azul_der_bomba:
                resultado["Posicion_Azul_Derecha_SAM_Modificado_pulg"] = (
                    azul_der_bomba["posicion"]
                )
                resultado["Carga_Azul_Derecha_SAM_Modificado_lbf"] = (
                    azul_der_bomba["carga"]
                )
                resultado["Verticalidad_Azul_Derecho_SAM_Modificado"] = (
                    azul_der_bomba["verticalidad"]
                )
                resultado[
                    "Sumergencia_Relativa_Candidata_Azul_Derecho_pct"
                ] = azul_der_bomba["sumergencia_relativa_pct"]

            for tabla_sam in (resultados_cartas, base_diagnosticos):
                mascara_carta_sam = tabla_sam["CartaId"].astype(int) == carta_id
                tabla_sam.loc[
                    mascara_carta_sam, "Regla_Inferior_SAM_Modificado"
                ] = "SOLO_CODO_AZUL_DERECHO"
                tabla_sam.loc[
                    mascara_carta_sam,
                    "Azul_Izquierdo_Incluido_SAM_Modificado",
                ] = False
            azul_derecho = pd.to_numeric(
                resultado.get("Carga_Azul_Derecha_SAM_Modificado_lbf"),
                errors="coerce",
            )
            superior_sam = pd.to_numeric(
                resultado.get("Carga_Superior_SAM_Seleccionada_lbf"),
                errors="coerce",
            )
            area_sam = pd.to_numeric(
                resultado.get("Area_Piston_SAM_pulg2"), errors="coerce"
            )
            gradiente_sam = pd.to_numeric(
                resultado.get("Gradiente_SAM_psi_m"), errors="coerce"
            )
            descarga_sam = pd.to_numeric(
                resultado.get("Presion_Descarga_Bomba_SAM_psi"),
                errors="coerce",
            )
            casing_sam_kg = pd.to_numeric(
                resultado.get("Presion_Casing_SAM_kg_cm2"), errors="coerce"
            )
            profundidad_sam = pd.to_numeric(
                resultado.get("Profundidad_Bomba_m"), errors="coerce"
            )
            peso_sam = superior_sam - azul_derecho
            if (
                np.isfinite([
                    azul_derecho, superior_sam, area_sam, gradiente_sam,
                    descarga_sam, casing_sam_kg, profundidad_sam,
                ]).all()
                and peso_sam > 0
                and area_sam > 0
                and gradiente_sam > 0
                and profundidad_sam > 0
            ):
                pip_sam = descarga_sam - peso_sam / area_sam
                sumergencia_sam = (
                    pip_sam - casing_sam_kg * KG_CM2_A_PSI
                ) / gradiente_sam
                correccion_sam = {
                    "Metodo_SAM_Seleccionado": (
                        "SAM_MODIFICADO_GOLPE_BOMBA_AZUL_DERECHO"
                    ),
                    "Regla_Inferior_SAM_Modificado": (
                        "SOLO_CODO_AZUL_DERECHO"
                    ),
                    "Azul_Izquierdo_Incluido_SAM_Modificado": False,
                    "Carga_Inferior_SAM_Seleccionada_lbf": azul_derecho,
                    "Peso_Fluido_SAM_Seleccionado_lbf": peso_sam,
                    "Diferencial_Carga_SAM_psi": peso_sam / area_sam,
                    "PIP_SAM_Seleccionado_psi": pip_sam,
                    "Sumergencia_SAM_Seleccionada_m": sumergencia_sam,
                    "Sumergencia_Relativa_SAM_Seleccionada_pct": (
                        100.0 * sumergencia_sam / profundidad_sam
                    ),
                    "Nivel_Dinamico_SAM_Modificado_m": (
                        profundidad_sam - sumergencia_sam
                    ),
                    "Carga_Roja_Izquierda_SAM_Modificado_lbf": resultado.get(
                        "Carga_Roja_Izquierda_SAM_Modificado_lbf", np.nan
                    ),
                    "Posicion_Roja_Izquierda_SAM_Modificado_pulg": resultado.get(
                        "Posicion_Roja_Izquierda_SAM_Modificado_pulg", np.nan
                    ),
                    "Carga_Azul_Derecha_SAM_Modificado_lbf": azul_derecho,
                    "Posicion_Azul_Derecha_SAM_Modificado_pulg": resultado.get(
                        "Posicion_Azul_Derecha_SAM_Modificado_pulg", np.nan
                    ),
                    "Carga_Superior_SAM_Seleccionada_lbf": superior_sam,
                }
                for campo, valor in correccion_sam.items():
                    resultado[campo] = valor
                    resultados_cartas.loc[
                        resultados_cartas["CartaId"].astype(int) == carta_id,
                        campo,
                    ] = valor
                    base_diagnosticos.loc[
                        base_diagnosticos["CartaId"].astype(int) == carta_id,
                        campo,
                    ] = valor

                mascara_carta_base = (
                    base_diagnosticos["CartaId"].astype(int) == carta_id
                )
                base_diagnosticos.loc[
                    mascara_carta_base, "Peso_Fluido_Usado_lbf"
                ] = peso_sam
                base_diagnosticos.loc[
                    mascara_carta_base, "Sumergencia_Usada_m"
                ] = sumergencia_sam

        # Consistencia lateral para cartas regulares de llenado alto. En esta
        # familia los cuatro codos deben quedar sobre las transferencias, no en
        # las pseudo horizontales. La corrección sólo actúa sobre valores
        # claramente fuera de banda y no se mezcla con golpe, compresión ni
        # pérdida en viajera, que tienen geometrías propias.
        sumergencia_regular_actual = pd.to_numeric(
            resultado.get(
                "Sumergencia_Relativa_SAM_Seleccionada_pct", np.nan
            ),
            errors="coerce",
        )
        metodo_sam_actual = str(
            resultado.get("Metodo_SAM_Seleccionado", "")
        ).upper()
        cargas_previas_laterales = pd.to_numeric(pd.Series([
            resultado.get("Carga_Roja_Izquierda_SAM_Modificado_lbf"),
            resultado.get("Carga_Roja_Derecha_SAM_Modificado_lbf"),
            resultado.get("Carga_Azul_Izquierda_SAM_Modificado_lbf"),
            resultado.get("Carga_Azul_Derecha_SAM_Modificado_lbf"),
        ]), errors="coerce").to_numpy(dtype=float)
        cargas_carta_previas = np.concatenate([
            np.asarray(resultado["Ascendente"]["carga"], dtype=float),
            np.asarray(resultado["Descendente"]["carga"], dtype=float),
        ])
        y_min_previo_regular = float(np.min(cargas_carta_previas))
        rango_y_previo_regular = max(float(np.ptp(cargas_carta_previas)), 1e-9)
        posiciones_carta_previas = np.concatenate([
            np.asarray(resultado["Ascendente"]["posicion"], dtype=float),
            np.asarray(resultado["Descendente"]["posicion"], dtype=float),
        ])
        rango_x_previo_regular = max(
            float(np.ptp(posiciones_carta_previas)), 1e-9
        )
        carta_angosta_y_alta = bool(
            45.0 <= rango_x_previo_regular <= 65.0
            and rango_y_previo_regular / rango_x_previo_regular >= 180.0
        )
        rojos_previos = cargas_previas_laterales[:2]
        azules_previos = cargas_previas_laterales[2:]
        invariantes_laterales_rotas = bool(
            np.isfinite(cargas_previas_laterales).all()
            and (
                abs(rojos_previos[0] - azules_previos[0])
                < 0.12 * rango_y_previo_regular
                or abs(rojos_previos[1] - azules_previos[1])
                < 0.12 * rango_y_previo_regular
                or np.max(azules_previos) >= np.min(rojos_previos)
                or np.max(
                    (azules_previos - y_min_previo_regular)
                    / rango_y_previo_regular
                ) > 0.55
                or (
                    np.isfinite(sumergencia_regular_actual)
                    and sumergencia_regular_actual < 0.0
                )
            )
        )
        morfologia_especial_no_confirmada = bool(
            (
                "MORFOLOGIA_VALVULA_VIAJERA" in metodo_sam_actual
                or "MORFOLOGIA_COMPRESION_GAS" in metodo_sam_actual
            )
            and invariantes_laterales_rotas
            and carta_angosta_y_alta
        )
        carta_regular_llena = bool(
            horizontales_ok
            and np.isfinite(llenado_operativo)
            and llenado_operativo >= 85.0
            and (
                morfologia_especial_no_confirmada
                or (
                    not golpe_bomba
                    and not golpe_fluido
                    and not compresion_gas
                    and not perdida_valvula
                    and
                    np.isfinite(sumergencia_regular_actual)
                    and sumergencia_regular_actual >= 10.0
                    and not transferencia_inferior_sostenida
                )
            )
        )
        if carta_regular_llena:
            resultado = resultado.copy()
            x_asc_regular = np.asarray(
                resultado["Ascendente"]["posicion"], dtype=float
            )
            y_asc_regular = np.asarray(
                resultado["Ascendente"]["carga"], dtype=float
            )
            x_desc_regular = np.asarray(
                resultado["Descendente"]["posicion"], dtype=float
            )
            y_desc_regular = np.asarray(
                resultado["Descendente"]["carga"], dtype=float
            )
            x_min_regular = float(min(
                np.min(x_asc_regular), np.min(x_desc_regular)
            ))
            x_max_regular = float(max(
                np.max(x_asc_regular), np.max(x_desc_regular)
            ))
            y_min_regular = float(min(
                np.min(y_asc_regular), np.min(y_desc_regular)
            ))
            y_max_regular = float(max(
                np.max(y_asc_regular), np.max(y_desc_regular)
            ))
            rango_x_regular = max(x_max_regular - x_min_regular, 1e-9)
            rango_y_regular = max(y_max_regular - y_min_regular, 1e-9)

            def candidato_lateral_regular(x, y, lado, objetivo_carga_rel):
                candidatos = []
                for indice in range(1, len(x) - 1):
                    x_rel = (x[indice] - x_min_regular) / rango_x_regular
                    y_rel = (y[indice] - y_min_regular) / rango_y_regular
                    if lado == "izquierdo" and x_rel > 0.18:
                        continue
                    if lado == "derecho" and x_rel < 0.82:
                        continue
                    dx_rel = abs(x[indice + 1] - x[indice - 1]) / rango_x_regular
                    dy_rel = abs(y[indice + 1] - y[indice - 1]) / rango_y_regular
                    verticalidad = dx_rel / max(dy_rel, 1e-9)
                    if verticalidad > 0.55:
                        continue
                    score = (
                        abs(y_rel - objetivo_carga_rel)
                        + 0.20 * verticalidad
                    )
                    candidatos.append((score, indice))
                if not candidatos:
                    return None
                indice = min(candidatos, key=lambda item: item[0])[1]
                return float(x[indice]), float(y[indice])

            rojo_izq_actual = pd.to_numeric(
                resultado.get("Carga_Roja_Izquierda_SAM_Modificado_lbf"),
                errors="coerce",
            )
            rojo_der_actual = pd.to_numeric(
                resultado.get("Carga_Roja_Derecha_SAM_Modificado_lbf"),
                errors="coerce",
            )
            azul_der_actual = pd.to_numeric(
                resultado.get("Carga_Azul_Derecha_SAM_Modificado_lbf"),
                errors="coerce",
            )
            azul_izq_actual = pd.to_numeric(
                resultado.get("Carga_Azul_Izquierda_SAM_Modificado_lbf"),
                errors="coerce",
            )
            rojo_izq_rel = (
                (rojo_izq_actual - y_min_regular) / rango_y_regular
                if np.isfinite(rojo_izq_actual) else np.nan
            )
            rojo_der_rel = (
                (rojo_der_actual - y_min_regular) / rango_y_regular
                if np.isfinite(rojo_der_actual) else np.nan
            )
            azul_der_rel = (
                (azul_der_actual - y_min_regular) / rango_y_regular
                if np.isfinite(azul_der_actual) else np.nan
            )
            azul_izq_rel = (
                (azul_izq_actual - y_min_regular) / rango_y_regular
                if np.isfinite(azul_izq_actual) else np.nan
            )
            rojo_izq_nuevo = candidato_lateral_regular(
                x_asc_regular, y_asc_regular, "izquierdo", 0.78
            )
            rojo_der_nuevo = candidato_lateral_regular(
                (
                    x_desc_regular
                    if morfologia_especial_no_confirmada
                    else x_asc_regular
                ),
                (
                    y_desc_regular
                    if morfologia_especial_no_confirmada
                    else y_asc_regular
                ),
                "derecho",
                0.78,
            )
            azul_izq_nuevo = candidato_lateral_regular(
                x_desc_regular, y_desc_regular, "izquierdo", 0.32
            )
            azul_der_nuevo = candidato_lateral_regular(
                x_desc_regular, y_desc_regular, "derecho", 0.35
            )

            correccion_regular = {}
            if (
                rojo_izq_nuevo is not None
                and np.isfinite(rojo_izq_rel)
                and (rojo_izq_rel < 0.65 or rojo_izq_rel > 0.86)
            ):
                correccion_regular.update({
                    "Posicion_Roja_Izquierda_SAM_Modificado_pulg": rojo_izq_nuevo[0],
                    "Carga_Roja_Izquierda_SAM_Modificado_lbf": rojo_izq_nuevo[1],
                })
            if (
                rojo_der_nuevo is not None
                and np.isfinite(rojo_der_rel)
                and (
                    morfologia_especial_no_confirmada
                    or rojo_der_rel < 0.65
                    or rojo_der_rel > 0.86
                )
            ):
                correccion_regular.update({
                    "Posicion_Roja_Derecha_SAM_Modificado_pulg": rojo_der_nuevo[0],
                    "Carga_Roja_Derecha_SAM_Modificado_lbf": rojo_der_nuevo[1],
                })
            if (
                azul_izq_nuevo is not None
                and np.isfinite(azul_izq_rel)
                and morfologia_especial_no_confirmada
                and (azul_izq_rel < 0.18 or azul_izq_rel > 0.55)
            ):
                correccion_regular.update({
                    "Posicion_Azul_Izquierda_SAM_Modificado_pulg": azul_izq_nuevo[0],
                    "Carga_Azul_Izquierda_SAM_Modificado_lbf": azul_izq_nuevo[1],
                })
            if (
                azul_der_nuevo is not None
                and np.isfinite(azul_der_rel)
                and (
                    azul_der_rel > 0.55
                    or (
                        morfologia_especial_no_confirmada
                        and azul_der_rel < 0.18
                    )
                )
            ):
                correccion_regular.update({
                    "Posicion_Azul_Derecha_SAM_Modificado_pulg": azul_der_nuevo[0],
                    "Carga_Azul_Derecha_SAM_Modificado_lbf": azul_der_nuevo[1],
                })

            if correccion_regular:
                valores_regulares_originales = {
                    campo: resultado.get(campo)
                    for campo in correccion_regular
                }
                for campo, valor in correccion_regular.items():
                    resultado[campo] = valor
                rojo_izq_final = pd.to_numeric(resultado.get(
                    "Carga_Roja_Izquierda_SAM_Modificado_lbf"
                ), errors="coerce")
                rojo_der_final = pd.to_numeric(resultado.get(
                    "Carga_Roja_Derecha_SAM_Modificado_lbf"
                ), errors="coerce")
                azul_izq_final = pd.to_numeric(resultado.get(
                    "Carga_Azul_Izquierda_SAM_Modificado_lbf"
                ), errors="coerce")
                azul_der_final = pd.to_numeric(resultado.get(
                    "Carga_Azul_Derecha_SAM_Modificado_lbf"
                ), errors="coerce")
                superior_final = 0.5 * (rojo_izq_final + rojo_der_final)
                incluir_azul_izq = bool(resultado.get(
                    "Azul_Izquierdo_Incluido_SAM_Modificado", True
                ))
                inferior_final = (
                    0.5 * (azul_izq_final + azul_der_final)
                    if incluir_azul_izq and np.isfinite(azul_izq_final)
                    else azul_der_final
                )
                area_final = pd.to_numeric(
                    resultado.get("Area_Piston_SAM_pulg2"), errors="coerce"
                )
                gradiente_final = pd.to_numeric(
                    resultado.get("Gradiente_SAM_psi_m"), errors="coerce"
                )
                descarga_final = pd.to_numeric(
                    resultado.get("Presion_Descarga_Bomba_SAM_psi"), errors="coerce"
                )
                casing_final = pd.to_numeric(
                    resultado.get("Presion_Casing_SAM_kg_cm2"), errors="coerce"
                )
                profundidad_final = pd.to_numeric(
                    resultado.get("Profundidad_Bomba_m"), errors="coerce"
                )
                peso_final = superior_final - inferior_final
                if (
                    np.isfinite([
                        superior_final, inferior_final, area_final,
                        gradiente_final, descarga_final, casing_final,
                        profundidad_final,
                    ]).all()
                    and peso_final > 0 and area_final > 0
                    and gradiente_final > 0 and profundidad_final > 0
                ):
                    pip_final = descarga_final - peso_final / area_final
                    sumergencia_final = (
                        pip_final - casing_final * KG_CM2_A_PSI
                    ) / gradiente_final
                    sumergencia_relativa_final = (
                        100.0 * sumergencia_final / profundidad_final
                    )
                    correccion_regular.update({
                        "Carga_Superior_SAM_Seleccionada_lbf": superior_final,
                        "Carga_Inferior_SAM_Seleccionada_lbf": inferior_final,
                        "Peso_Fluido_SAM_Seleccionado_lbf": peso_final,
                        "Diferencial_Carga_SAM_psi": peso_final / area_final,
                        "PIP_SAM_Seleccionado_psi": pip_final,
                        "Sumergencia_SAM_Seleccionada_m": sumergencia_final,
                        "Sumergencia_Relativa_SAM_Seleccionada_pct": (
                            sumergencia_relativa_final
                        ),
                        "Nivel_Dinamico_SAM_Modificado_m": (
                            profundidad_final - sumergencia_final
                        ),
                    })
                    correccion_fisicamente_admisible = bool(
                        morfologia_especial_no_confirmada
                        or sumergencia_relativa_final >= 10.0
                        or (
                            np.isfinite(sumergencia_regular_actual)
                            and sumergencia_relativa_final
                            > sumergencia_regular_actual
                        )
                    )
                    if correccion_fisicamente_admisible:
                        for campo, valor in correccion_regular.items():
                            resultado[campo] = valor
                            for tabla_sam in (
                                resultados_cartas, base_diagnosticos
                            ):
                                if campo not in tabla_sam.columns:
                                    tabla_sam[campo] = np.nan
                                tabla_sam.loc[
                                    tabla_sam["CartaId"].astype(int) == carta_id,
                                    campo,
                                ] = valor
                    else:
                        for campo, valor in valores_regulares_originales.items():
                            resultado[campo] = valor

        # Fine tuning exclusivo para pérdida en viajera ya confirmada. Se
        # ejecuta después de evaluar golpe de bomba para que una corrección
        # visual no cambie la clasificación morfológica de la carta.
        if perdida_valvula and not golpe_bomba:
            resultado = resultado.copy()
            x_asc_viajera = np.asarray(
                resultado["Ascendente"]["posicion"], dtype=float
            )
            y_asc_viajera = np.asarray(
                resultado["Ascendente"]["carga"], dtype=float
            )
            x_min_viajera = float(np.min(x_asc_viajera))
            x_max_viajera = float(np.max(x_asc_viajera))
            rango_x_viajera = max(x_max_viajera - x_min_viajera, 1e-9)
            y_todas_viajera = np.concatenate([
                y_asc_viajera,
                np.asarray(resultado["Descendente"]["carga"], dtype=float),
            ])
            rango_y_viajera = max(float(np.ptp(y_todas_viajera)), 1e-9)
            rojo_izq_actual = pd.to_numeric(
                resultado.get("Carga_Roja_Izquierda_SAM_Modificado_lbf"),
                errors="coerce",
            )
            rojo_der_actual = pd.to_numeric(
                resultado.get("Carga_Roja_Derecha_SAM_Modificado_lbf"),
                errors="coerce",
            )
            azul_izq_actual = pd.to_numeric(
                resultado.get("Carga_Azul_Izquierda_SAM_Modificado_lbf"),
                errors="coerce",
            )
            azul_der_actual = pd.to_numeric(
                resultado.get("Carga_Azul_Derecha_SAM_Modificado_lbf"),
                errors="coerce",
            )

            def rojo_adentro_oblicua(lado, rojo_actual, azul_actual):
                mascara = (
                    x_asc_viajera <= x_min_viajera + 0.34 * rango_x_viajera
                    if lado == "izquierda"
                    else x_asc_viajera >= x_max_viajera - 0.34 * rango_x_viajera
                )
                indices = np.flatnonzero(
                    mascara
                    & (y_asc_viajera < rojo_actual - 0.012 * rango_y_viajera)
                    & (y_asc_viajera > azul_actual + 0.30 * rango_y_viajera)
                )
                if len(indices) == 0:
                    return None
                objetivo = rojo_actual - 0.035 * rango_y_viajera
                indice = int(indices[np.argmin(
                    abs(y_asc_viajera[indices] - objetivo)
                )])
                return (
                    float(x_asc_viajera[indice]),
                    float(y_asc_viajera[indice]),
                )

            rojo_izq_nuevo = rojo_adentro_oblicua(
                "izquierda", rojo_izq_actual, azul_izq_actual
            )
            rojo_der_nuevo = rojo_adentro_oblicua(
                "derecha", rojo_der_actual, azul_der_actual
            )
            inferior_viajera = pd.to_numeric(
                resultado.get("Carga_Inferior_SAM_Seleccionada_lbf"),
                errors="coerce",
            )
            superior_actual = pd.to_numeric(
                resultado.get("Carga_Superior_SAM_Seleccionada_lbf"),
                errors="coerce",
            )
            area_viajera = pd.to_numeric(
                resultado.get("Area_Piston_SAM_pulg2"), errors="coerce"
            )
            gradiente_viajera = pd.to_numeric(
                resultado.get("Gradiente_SAM_psi_m"), errors="coerce"
            )
            descarga_viajera = pd.to_numeric(
                resultado.get("Presion_Descarga_Bomba_SAM_psi"), errors="coerce"
            )
            casing_viajera = pd.to_numeric(
                resultado.get("Presion_Casing_SAM_kg_cm2"), errors="coerce"
            )
            profundidad_viajera = pd.to_numeric(
                resultado.get("Profundidad_Bomba_m"), errors="coerce"
            )
            if (
                rojo_izq_nuevo is not None
                and rojo_der_nuevo is not None
                and np.isfinite([
                    inferior_viajera, superior_actual, area_viajera,
                    gradiente_viajera, descarga_viajera, casing_viajera,
                    profundidad_viajera,
                ]).all()
            ):
                superior_nuevo = 0.5 * (
                    rojo_izq_nuevo[1] + rojo_der_nuevo[1]
                )
                peso_actual = superior_actual - inferior_viajera
                peso_nuevo = superior_nuevo - inferior_viajera
                if 0 < peso_nuevo <= peso_actual + 1e-9:
                    pip_nuevo = descarga_viajera - peso_nuevo / area_viajera
                    sumergencia_nueva = (
                        pip_nuevo - casing_viajera * KG_CM2_A_PSI
                    ) / gradiente_viajera
                    correccion_viajera = {
                        "Carga_Roja_Izquierda_SAM_Modificado_lbf": rojo_izq_nuevo[1],
                        "Posicion_Roja_Izquierda_SAM_Modificado_pulg": rojo_izq_nuevo[0],
                        "Carga_Roja_Derecha_SAM_Modificado_lbf": rojo_der_nuevo[1],
                        "Posicion_Roja_Derecha_SAM_Modificado_pulg": rojo_der_nuevo[0],
                        "Carga_Superior_SAM_Seleccionada_lbf": superior_nuevo,
                        "Peso_Fluido_SAM_Seleccionado_lbf": peso_nuevo,
                        "Diferencial_Carga_SAM_psi": peso_nuevo / area_viajera,
                        "PIP_SAM_Seleccionado_psi": pip_nuevo,
                        "Sumergencia_SAM_Seleccionada_m": sumergencia_nueva,
                        "Sumergencia_Relativa_SAM_Seleccionada_pct": (
                            100.0 * sumergencia_nueva / profundidad_viajera
                        ),
                        "Nivel_Dinamico_SAM_Modificado_m": (
                            profundidad_viajera - sumergencia_nueva
                        ),
                    }
                    for campo, valor in correccion_viajera.items():
                        resultado[campo] = valor
                        for tabla_sam in (resultados_cartas, base_diagnosticos):
                            tabla_sam.loc[
                                tabla_sam["CartaId"].astype(int) == carta_id,
                                campo,
                            ] = valor
                    mascara_carta_base = (
                        base_diagnosticos["CartaId"].astype(int) == carta_id
                    )
                    base_diagnosticos.loc[
                        mascara_carta_base, "Peso_Fluido_Usado_lbf"
                    ] = peso_nuevo
                    base_diagnosticos.loc[
                        mascara_carta_base, "Sumergencia_Usada_m"
                    ] = sumergencia_nueva

        # Si la transferencia derecha no pudo medirse y la carta conserva
        # un llenado alto, un vacío superior pequeño y solamente un
        # vacío inferior moderado, no se atribuye automáticamente la
        # geometría a gas cuando existe un golpe de bomba izquierdo fuerte.
        # En esta familia el impacto domina la deformación y el respaldo de
        # transferencia no aporta evidencia independiente de admisión.
        admision_no_confirmada_por_golpe_izquierdo = bool(
            compresion_gas
            and transferencia_progresiva_inferida
            and golpe_bomba
            and np.isfinite(llenado)
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and llenado >= 85.0
            and vacio_sd < 12.0
            and vacio_id < 45.0
            and metricas_golpe[
                "Profundidad_Golpe_Inferior_pct"
            ] >= 20.0
        )

        if admision_no_confirmada_por_golpe_izquierdo:
            compresion_gas = False
            compresion_gas_suave = False
            severidad_admision = None
            alertas = [
                alerta
                for alerta in alertas
                if alerta
                != "Posible compresión/interferencia de gas"
            ]
            evidencias.append(
                "Transferencia derecha no confirmada; geometría "
                "dominada por golpe de bomba izquierdo"
            )

        # Respaldo muy acotado para cartas fronterizas en las que coexisten
        # un golpe de bomba a la izquierda y un llenado incompleto asociado
        # a la transferencia derecha. Algunas de estas cartas no entregan
        # métricas 20-80 estables y, por eso, quedaban únicamente como golpe
        # de bomba aunque conservaran la firma leve de golpe de fluido.
        golpe_fluido_fronterizo_con_golpe_bomba = bool(
            golpe_bomba
            and horizontales_ok
            and not sin_trabajo
            # Golpe de fluido y compresión de gas representan dos
            # formas alternativas de la transferencia de admisión.
            # No se permite que este respaldo agregue golpe de fluido
            # a una carta ya clasificada como compresión.
            and not compresion_gas
            and not admision_no_confirmada_por_golpe_izquierdo
            and transferencia_inferior_sostenida
            and np.isfinite(extension_despegue_inferior)
            and extension_despegue_inferior >= 5.0
            and np.isfinite(llenado)
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and 85.0 <= llenado <= 92.0
            and vacio_sd >= UMBRAL_VACIO_SUP_DER_ADMISION_SUAVE
            and vacio_id >= 15.0
        )

        if golpe_fluido_fronterizo_con_golpe_bomba:
            golpe_fluido = True
            severidad_admision = "LEVE"

            if "Posible golpe de fluido" not in alertas:
                alertas.append(
                    "Posible golpe de fluido"
                )

            evidencias.append(
                "Golpe de bomba coexistente con vacío derecho "
                "y llenado fronterizo"
            )

        # --------------------------------------------------------
        # 5. POZO SUBEXPLOTADO
        # --------------------------------------------------------

        # Puede existir una corrección SAM posterior (por ejemplo, golpe de
        # bomba usando sólo el azul derecho). El diagnóstico consume siempre
        # el valor SAM final del resultado, no la sumergencia API ni una copia
        # previa a esa corrección.
        sumergencia_relativa = pd.to_numeric(
            resultado.get(
                "Sumergencia_Relativa_SAM_Seleccionada_pct",
                np.nan,
            ),
            errors="coerce",
        )
        peso_fluido_diagnostico = pd.to_numeric(
            resultado.get("Peso_Fluido_SAM_Seleccionado_lbf", np.nan),
            errors="coerce",
        )
        sumergencia_diagnostico_m = pd.to_numeric(
            resultado.get("Sumergencia_SAM_Seleccionada_m", np.nan),
            errors="coerce",
        )
        profundidad_diagnostico_m = pd.to_numeric(
            resultado.get("Profundidad_Bomba_m", np.nan),
            errors="coerce",
        )
        datos_operativos_validos = bool(
            np.isfinite(peso_fluido_diagnostico)
            and peso_fluido_diagnostico > 0.0
            and np.isfinite(sumergencia_diagnostico_m)
            and sumergencia_diagnostico_m >= 0.0
            and np.isfinite(profundidad_diagnostico_m)
            and sumergencia_diagnostico_m <= profundidad_diagnostico_m
            and np.isfinite(llenado_operativo)
            and 0.0 <= llenado_operativo <= 140.0
        )

        # Para evaluar una oportunidad de subexplotación solamente se
        # necesitan el llenado calculado de la carta y una sumergencia SAM
        # utilizable, derivada de las horizontales nuevas.
        datos_subexplotacion_validos = bool(
            datos_operativos_validos
            and np.isfinite(llenado_operativo)
            and 0.0 <= llenado_operativo <= 140.0
            and np.isfinite(sumergencia_relativa)
            and sumergencia_relativa >= 0.0
        )

        # Cuando la geometría redondeada fue corregida por fricción, el
        # vacío derecho calculado con las horizontales originales puede
        # imitar una compresión suave. Si, después de la corrección, el
        # llenado vuelve a ser alto y existe sumergencia suficiente, se
        # interpreta como oportunidad de extracción con fricción, no como
        # admisión incompleta. No se anula un golpe de fluido ni una
        # compresión marcada: solamente la variante suave.
        friccion_reclasifica_compresion_suave = bool(
            bool(
                resultado.get(
                    "Correccion_Friccion_Aplicada",
                    False,
                )
            )
            and compresion_gas
            and compresion_gas_suave
            and datos_subexplotacion_validos
            and np.isfinite(llenado_operativo)
            and np.isfinite(sumergencia_relativa)
            and llenado_operativo >= 85.0
            and sumergencia_relativa >= 10.0
        )

        if friccion_reclasifica_compresion_suave:
            compresion_gas = False
            compresion_gas_suave = False
            severidad_admision = "NO_APLICA"
            alertas = [
                alerta
                for alerta in alertas
                if alerta
                != "Posible compresión/interferencia de gas"
            ]
            evidencias.append(
                "La corrección por fricción recupera llenado alto; "
                "se descarta compresión suave como causa principal"
            )

        subexplotado = bool(
            horizontales_ok
            and not sin_trabajo
            and not golpe_fluido
            and not compresion_gas
            and (
                not transferencia_inferior_sostenida
                or friccion_reclasifica_compresion_suave
            )
            and datos_subexplotacion_validos
            and np.isfinite(llenado_operativo)
            and np.isfinite(sumergencia_relativa)
            and llenado_operativo >= 85.0
            and sumergencia_relativa >= 10.0
        )

        if subexplotado:
            alertas.append(
                "Posible pozo subexplotado"
            )

            evidencias.append(
                "Llenado alto y sumergencia mayor al 10 % de profundidad"
            )



        # --------------------------------------------------------
        # 6. POSIBLE TUBING LIBRE
        # --------------------------------------------------------
        # --------------------------------------------------------
        # 6. POSIBLE TUBING LIBRE
        # --------------------------------------------------------
        # Se utilizan exclusivamente los laterales de la carta ideal.

        angulo_tubing_izquierdo = np.nan
        angulo_tubing_derecho = np.nan
        metodo_angulo_tubing = "NO_CALCULADO"

        vertices_tubing = resultado[
            "Vertices_Ideal"
        ]

        if vertices_tubing is not None:
            vertices_tubing = np.asarray(
                vertices_tubing,
                dtype=float,
            )

            if (
                vertices_tubing.shape == (4, 2)
                and np.all(
                    np.isfinite(vertices_tubing)
                )
            ):
                # Orden:
                # 0 = superior izquierdo
                # 1 = superior derecho
                # 2 = inferior derecho
                # 3 = inferior izquierdo

                superior_izquierdo = (
                    vertices_tubing[0]
                )

                superior_derecho = (
                    vertices_tubing[1]
                )

                inferior_derecho = (
                    vertices_tubing[2]
                )

                inferior_izquierdo = (
                    vertices_tubing[3]
                )

                ancho_referencia = abs(
                    superior_derecho[0]
                    - superior_izquierdo[0]
                )

                altura_referencia = abs(
                    superior_izquierdo[1]
                    - inferior_izquierdo[1]
                )

                if (
                    ancho_referencia > 1e-9
                    and altura_referencia > 1e-9
                ):
                    # --------------------------------------------
                    # LATERAL IZQUIERDO
                    # --------------------------------------------
                    dx_izquierdo = (
                        superior_izquierdo[0]
                        - inferior_izquierdo[0]
                    ) / ancho_referencia

                    dy_izquierdo = (
                        superior_izquierdo[1]
                        - inferior_izquierdo[1]
                    ) / altura_referencia

                    angulo_tubing_izquierdo = float(
                        np.degrees(
                            np.arctan2(
                                dy_izquierdo,
                                dx_izquierdo,
                            )
                        )
                    )

                    # --------------------------------------------
                    # LATERAL DERECHO
                    # --------------------------------------------
                    dx_derecho = (
                        superior_derecho[0]
                        - inferior_derecho[0]
                    ) / ancho_referencia

                    dy_derecho = (
                        superior_derecho[1]
                        - inferior_derecho[1]
                    ) / altura_referencia

                    orientacion_derecha = float(
                        np.degrees(
                            np.arctan2(
                                dy_derecho,
                                dx_derecho,
                            )
                        )
                    )

                    # Ángulo interior respecto de la horizontal
                    # inferior que apunta hacia la izquierda.
                    angulo_tubing_derecho = (
                        180.0
                        - orientacion_derecha
                    )

                    metodo_angulo_tubing = (
                        "LATERALES_CARTA_IDEAL"
                    )

        # Condición simple solicitada:
        # izquierda menor a 90° y derecha igual o mayor a 90°.
        tubing_libre = bool(
            horizontales_ok
            and not sin_trabajo
            and not perdida_valvula
            and not cierre_tardio_viajera
            and np.isfinite(
                angulo_ideal_izquierdo
            )
            and np.isfinite(
                angulo_ideal_derecho
            )
            and angulo_ideal_izquierdo < 83.0
            and angulo_ideal_derecho >= 90.0
        )

        if tubing_libre:
            alertas.append(
                "Posible tubing libre"
            )

            evidencias.append(
                "Lateral teórico izquierdo menor a 90°, "
                "lateral teórico derecho igual o mayor a 90° "
                "y sin evidencia de pérdida en válvula viajera"
            )
        # --------------------------------------------------------
        # DIAGNÓSTICO PRINCIPAL
        # --------------------------------------------------------

        # Una carta geométricamente inválida no admite diagnósticos
        # adicionales. También se eliminan alertas operativas que,
        # aunque provengan de otros campos de la API, podrían inducir
        # a interpretar como confiable una adquisición corrupta.
        if carta_no_valida:
            alertas = [
                "Carta no válida - posible falla de medición o transmisión"
            ]
            evidencias = list(
                resultado.get(
                    "Evidencias_Integridad",
                    [],
                )
            )
            exceso_torque = False
            exceso_carga_estructural = False
            sin_trabajo = False
            perdida_valvula = False
            cierre_tardio_viajera = False
            golpe_fluido = False
            compresion_gas = False
            compresion_gas_suave = False
            severidad_admision = "NO_APLICA"
            golpe_bomba = False
            tubing_libre = False
            subexplotado = False
            friccion_elevada = False

        # La falta de trabajo efectivo es incompatible con los
        # diagnósticos que requieren una bomba operando. Se conservan
        # alertas mecánicas independientes como torque y carga.
        if sin_trabajo:
            diagnosticos_incompatibles = {
                "Posible pozo subexplotado",
                "Posible golpe de fluido",
                "Posible compresión/interferencia de gas",
                "Posible pérdida en válvula viajera",
                "Posible tubing libre",
                "Posible fricción elevada",
            }
            diagnosticos_incompatibles.add(
                "Posible cierre tardío de válvula viajera"
            )
            alertas = [
                alerta
                for alerta in alertas
                if alerta not in diagnosticos_incompatibles
            ]
            if "Posible sin trabajo de bomba" not in alertas:
                alertas.append("Posible sin trabajo de bomba")

            perdida_valvula = False
            cierre_tardio_viajera = False
            golpe_fluido = False
            compresion_gas = False
            compresion_gas_suave = False
            severidad_admision = "NO_APLICA"
            tubing_libre = False
            subexplotado = False
            friccion_elevada = False

        # El exceso de torque y el exceso de carga estructural
        # permanecen como alertas operativas. No reemplazan el
        # diagnóstico dinamométrico principal de la carta.
        if carta_no_valida:
            diagnostico_principal = (
                "Carta no válida - posible falla de medición o transmisión"
            )
            accion = (
                "Revisar celda de carga, sensor de posición, "
                "sincronización y transmisión de datos"
            )
            confianza = 0.92
        elif sin_trabajo:
            diagnostico_principal = "Posible sin trabajo de bomba"
            accion = "Revisar bomba, sarta y carta de superficie"
            confianza = 0.90
        elif subexplotado:
            diagnostico_principal = "Posible pozo subexplotado"
            accion = "Evaluar aumento de régimen y revisar alertas secundarias"
            confianza = 0.72
        elif golpe_fluido:
            diagnostico_principal = "Posible golpe de fluido"
            accion = "Evaluar disminución de régimen"
            confianza = 0.78
        elif compresion_gas:
            diagnostico_principal = (
                "Posible compresión/interferencia de gas"
            )
            confianza = 0.68 if compresion_gas_suave else 0.74
            accion = (
                "Evaluar condición de admisión y revisar régimen"
            )
        elif perdida_valvula:
            diagnostico_principal = "Posible pérdida en válvula viajera"
            accion = "Revisar válvula viajera"
            confianza = 0.76
        elif cierre_tardio_viajera:
            diagnostico_principal = (
                "Posible cierre tardío de válvula viajera"
            )
            accion = (
                "Revisar válvula viajera, suciedad y dispositivo "
                "mecánico antibloqueo de gas"
            )
            confianza = 0.74
        elif golpe_bomba:
            diagnostico_principal = "Posible golpe de bomba"
            accion = "Revisar espaciamiento"
            confianza = 0.82
        elif tubing_libre:
            diagnostico_principal = "Posible tubing libre"
            accion = "Revisar condición y anclaje del tubing"
            confianza = 0.68
        elif friccion_elevada:
            diagnostico_principal = "Posible fricción elevada"
            accion = (
                "Revisar rozamiento de sarta, tubing, alineación "
                "y condiciones mecánicas"
            )
            confianza = 0.70
        elif exceso_torque:
            diagnostico_principal = "Exceso de torque"
            accion = (
                "Revisar balanceo, régimen y capacidad de la caja reductora"
            )
            confianza = 0.98
        elif exceso_carga_estructural:
            diagnostico_principal = "Exceso de carga estructural"
            accion = (
                "Revisar carga admisible de la unidad y condición estructural"
            )
            confianza = 0.98
        else:
            diagnostico_principal = "Pozo bien explotado"
            accion = "Mantener seguimiento operativo"
            confianza = 0.60

        # Las correcciones morfológicas posteriores al diagnóstico se hacen
        # sobre la Serie local ``resultado``. Se sincronizan también con las
        # tablas canónicas: la app reconstruye su tabla desde
        # ``resultados_cartas`` y, sin esta propagación, recuperaba los SAM
        # previos aunque ``diagnosticos_cartas`` ya mostrara los nuevos.
        campos_sam_sincronizar = [
            campo for campo in resultado.index
            if "SAM_" in str(campo) or "SAM_Modificado" in str(campo)
        ]
        for tabla_sam in (resultados_cartas, base_diagnosticos):
            mascara_carta_sam = tabla_sam["CartaId"].astype(int) == carta_id
            for campo_sam in campos_sam_sincronizar:
                if campo_sam in tabla_sam.columns:
                    tabla_sam.loc[
                        mascara_carta_sam, campo_sam
                    ] = resultado.get(campo_sam, np.nan)

        filas_diagnosticos.append({
            "CartaId": carta_id,
            "Pozo": metrica["Pozo"],
            "Fecha": metrica["Fecha"],
            "Diagnostico_Principal":
                diagnostico_principal,
            "Confianza":
                confianza,
            "Accion_Sugerida":
                accion,
            "Alertas":
                alertas,
            "Evidencias":
                evidencias,
            "Exceso_Torque":
                exceso_torque,
            "Torque_Reductor_pct":
                torque_reductor_pct,
            "Exceso_Carga_Estructural":
                exceso_carga_estructural,
            "Carga_Estructural_pct":
                carga_estructural_pct,
            "Carta_No_Valida":
                carta_no_valida,
            "Evidencias_Integridad":
                resultado.get(
                    "Evidencias_Integridad",
                    [],
                ),
            "Saltos_Grandes_Carta":
                resultado.get(
                    "Saltos_Grandes_Carta",
                    np.nan,
                ),
            "Reversiones_Posicion_Carta":
                resultado.get(
                    "Reversiones_Posicion_Carta",
                    np.nan,
                ),
            "Cruces_Propios_Carta":
                resultado.get(
                    "Cruces_Propios_Carta",
                    np.nan,
                ),
            "Cruces_Extremo_Izquierdo_Carta":
                resultado.get(
                    "Cruces_Extremo_Izquierdo_Carta",
                    np.nan,
                ),
            "Cruces_Fuera_Extremo_Izquierdo_Carta":
                resultado.get(
                    "Cruces_Fuera_Extremo_Izquierdo_Carta",
                    np.nan,
                ),
            "Rango_Carga_Sobre_Peso_API":
                resultado.get(
                    "Rango_Carga_Sobre_Peso_API",
                    np.nan,
                ),
            "Sin_Trabajo_Bomba":
                sin_trabajo,
            "Sin_Trabajo_Por_Area":
                sin_trabajo_por_area,
            "Bloqueo_Gas_Probable":
                bloqueo_gas_probable,
            "Compacidad_Carta":
                compacidad_carta,
            "Apertura_Central_Carta":
                apertura_central_carta,
            "Perdida_Valvula_Viajera":
                perdida_valvula,
            "Cierre_Tardio_Valvula_Viajera":
                cierre_tardio_viajera,
            "Carga_Inicial_Asc_Relativa":
                carga_inicial_asc_relativa,
            "Posicion_Cierre_50_pct":
                posicion_cierre_50,
            "Posicion_Cierre_80_pct":
                posicion_cierre_80,
            "Separacion_Ramas_Inicial_pct_gap":
                separacion_ramas_inicial,
            "Extension_Ramas_Juntas_pct":
                extension_ramas_juntas,
            "Golpe_Fluido":
                golpe_fluido,
            "Compresion_Gas":
                compresion_gas,
            "Compresion_Gas_Suave":
                compresion_gas_suave,
            "Severidad_Admision":
                severidad_admision,
            "Golpe_Bomba":
                golpe_bomba,
            "Profundidad_Golpe_Inferior_pct":
                metricas_golpe["Profundidad_Golpe_Inferior_pct"],
            "Ancho_Golpe_Inferior_pct":
                metricas_golpe["Ancho_Golpe_Inferior_pct"],
            "Posicion_Minimo_Golpe_pct":
                metricas_golpe["Posicion_Minimo_Golpe_pct"],
            "Puntos_Golpe_Inferior":
                metricas_golpe["Puntos_Golpe_Inferior"],
            "Golpe_Localizado_Izquierda":
                metricas_golpe["Golpe_Localizado_Izquierda"],
            "Rulo_Golpe_Bomba_Izquierdo": rulo_golpe_bomba,
            "Tubing_Libre":
                tubing_libre,
            "Friccion_Elevada":
                friccion_elevada,
            "Correccion_Friccion_Aplicada":
                bool(
                    resultado.get(
                        "Correccion_Friccion_Aplicada",
                        False,
                    )
                ),
            "Arqueo_Superior_Friccion_pct_gap":
                resultado.get(
                    "Arqueo_Superior_Friccion_pct_gap",
                    np.nan,
                ),
            "Arqueo_Inferior_Friccion_pct_gap":
                resultado.get(
                    "Arqueo_Inferior_Friccion_pct_gap",
                    np.nan,
                ),
            "Reduccion_Gap_Friccion_pct":
                resultado.get(
                    "Reduccion_Gap_Friccion_pct",
                    np.nan,
                ),
            "Angulo_Tubing_Izquierdo_deg":
                angulo_tubing_izquierdo,

            "Metodo_Angulo_Tubing":
                metodo_angulo_tubing,
            "Pozo_Subexplotado":
                subexplotado,
            "Llenado_Bruto_pct":
                np.nan if sin_trabajo else llenado_bruto,
            "Llenado_Original_pct":
                np.nan if sin_trabajo else llenado,
            "Llenado_Operativo_pct":
                np.nan if sin_trabajo else llenado_operativo,
            "Fuente_Variables_Diagnostico":
                "SAM_MODIFICADO_HORIZONTALES_NUEVAS",
            "Llenado_Diagnostico_pct":
                np.nan if sin_trabajo else llenado_operativo,
            "Peso_Fluido_Diagnostico_lbf":
                peso_fluido_diagnostico,
            "Sumergencia_Diagnostico_m":
                sumergencia_diagnostico_m,
            "Sumergencia_Relativa_pct":
                sumergencia_relativa,
            "Sumergencia_API_m":
                resultado.get("Sumergencia_API_m", np.nan),
            "Calculo_Sumergencia_Propia_Valido":
                bool(
                    resultado.get(
                        "Calculo_Sumergencia_Propia_Valido",
                        False,
                    )
                ),
            "Motivo_Sumergencia_Propia_No_Valida":
                resultado.get(
                    "Motivo_Sumergencia_Propia_No_Valida",
                    "",
                ),
            "Peso_Fluido_Horizontales_lbf":
                resultado.get("Peso_Fluido_Horizontales_lbf", np.nan),
            "Calculo_SAM_Modificado_Valido": bool(
                resultado.get("Calculo_SAM_Modificado_Valido", False)
            ),
            "Motivo_SAM_Modificado_No_Valido": resultado.get(
                "Motivo_SAM_Modificado_No_Valido", ""
            ),
            "Metodo_SAM_Seleccionado": resultado.get(
                "Metodo_SAM_Seleccionado",
                "SAM_MODIFICADO_EXTREMOS_TRANSFERENCIA",
            ),
            "Regla_Inferior_SAM_Modificado": resultado.get(
                "Regla_Inferior_SAM_Modificado",
                "PROMEDIO_DOS_CODOS_AZULES",
            ),
            "Azul_Izquierdo_Incluido_SAM_Modificado": bool(
                resultado.get("Azul_Izquierdo_Incluido_SAM_Modificado", True)
            ),
            "Carga_Roja_Izquierda_SAM_Modificado_lbf": resultado.get(
                "Carga_Roja_Izquierda_SAM_Modificado_lbf", np.nan
            ),
            "Carga_Roja_Derecha_SAM_Modificado_lbf": resultado.get(
                "Carga_Roja_Derecha_SAM_Modificado_lbf", np.nan
            ),
            "Carga_Azul_Izquierda_SAM_Modificado_lbf": resultado.get(
                "Carga_Azul_Izquierda_SAM_Modificado_lbf", np.nan
            ),
            "Carga_Azul_Derecha_SAM_Modificado_lbf": resultado.get(
                "Carga_Azul_Derecha_SAM_Modificado_lbf", np.nan
            ),
            "Posicion_Roja_Izquierda_SAM_Modificado_pulg": resultado.get(
                "Posicion_Roja_Izquierda_SAM_Modificado_pulg", np.nan
            ),
            "Posicion_Roja_Derecha_SAM_Modificado_pulg": resultado.get(
                "Posicion_Roja_Derecha_SAM_Modificado_pulg", np.nan
            ),
            "Posicion_Azul_Izquierda_SAM_Modificado_pulg": resultado.get(
                "Posicion_Azul_Izquierda_SAM_Modificado_pulg", np.nan
            ),
            "Posicion_Azul_Derecha_SAM_Modificado_pulg": resultado.get(
                "Posicion_Azul_Derecha_SAM_Modificado_pulg", np.nan
            ),
            "Carga_Superior_SAM_Seleccionada_lbf": resultado.get(
                "Carga_Superior_SAM_Seleccionada_lbf", np.nan
            ),
            "Carga_Inferior_SAM_Seleccionada_lbf": resultado.get(
                "Carga_Inferior_SAM_Seleccionada_lbf", np.nan
            ),
            "Peso_Fluido_SAM_Seleccionado_lbf": resultado.get(
                "Peso_Fluido_SAM_Seleccionado_lbf", np.nan
            ),
            "Area_Piston_SAM_pulg2": resultado.get(
                "Area_Piston_SAM_pulg2", np.nan
            ),
            "Diferencial_Carga_SAM_psi": resultado.get(
                "Diferencial_Carga_SAM_psi", np.nan
            ),
            "Presion_Tubing_SAM_kg_cm2": resultado.get(
                "Presion_Tubing_SAM_kg_cm2", np.nan
            ),
            "Presion_Casing_SAM_kg_cm2": resultado.get(
                "Presion_Casing_SAM_kg_cm2", np.nan
            ),
            "Gravedad_Especifica_SAM": resultado.get(
                "Gravedad_Especifica_SAM", np.nan
            ),
            "Gradiente_SAM_psi_m": resultado.get(
                "Gradiente_SAM_psi_m", np.nan
            ),
            "Presion_Descarga_Bomba_SAM_psi": resultado.get(
                "Presion_Descarga_Bomba_SAM_psi", np.nan
            ),
            "PIP_SAM_Seleccionado_psi": resultado.get(
                "PIP_SAM_Seleccionado_psi", np.nan
            ),
            "Sumergencia_SAM_Seleccionada_m": resultado.get(
                "Sumergencia_SAM_Seleccionada_m", np.nan
            ),
            "Sumergencia_Relativa_SAM_Seleccionada_pct": resultado.get(
                "Sumergencia_Relativa_SAM_Seleccionada_pct", np.nan
            ),
            "Nivel_Dinamico_SAM_Modificado_m": resultado.get(
                "Nivel_Dinamico_SAM_Modificado_m", np.nan
            ),
            "Delta_Sumergencia_SAM_Seleccionada_vs_API_m": (
                resultado.get("Sumergencia_SAM_Seleccionada_m", np.nan)
                - resultado.get("Sumergencia_API_m", np.nan)
            ),
            "Peso_Fluido_API_lbf":
                resultado.get("Peso_Fluido_API_lbf", np.nan),
            "Area_Piston_pulg2":
                resultado.get("Area_Piston_pulg2", np.nan),
            "Presion_Diferencial_Horizontales_psi":
                resultado.get(
                    "Presion_Diferencial_Horizontales_psi",
                    np.nan,
                ),
            "SG_Fluido_Asumido":
                resultado.get("SG_Fluido_Asumido", np.nan),
            "Gradiente_Fluido_Asumido_psi_m":
                resultado.get(
                    "Gradiente_Fluido_Asumido_psi_m",
                    np.nan,
                ),
            "Sumergencia_Propia_m":
                resultado.get("Sumergencia_Propia_m", np.nan),
            "Sumergencia_Relativa_Propia_pct":
                resultado.get(
                    "Sumergencia_Relativa_Propia_pct",
                    np.nan,
                ),
            "Delta_Sumergencia_Propia_vs_API_m": (
                resultado.get("Sumergencia_Propia_m", np.nan)
                - resultado.get("Sumergencia_API_m", np.nan)
            ),
            "Gravedad_Especifica_API":
                resultado.get("Gravedad_Especifica_API", np.nan),
            "Carrera_Efectiva_Fondo_pulg":
                resultado.get("Carrera_Efectiva_Fondo_pulg", np.nan),
            "Desplazamiento_Bruto_Efectivo_m3_d":
                resultado.get(
                    "Desplazamiento_Bruto_Efectivo_m3_d",
                    np.nan,
                ),
            "Carrera_Geometrica_Fondo_Calculada_pulg":
                carrera_geometrica_calculada_pulg,
            "Carrera_Efectiva_Fondo_Calculada_pulg":
                carrera_efectiva_calculada_pulg,
            "Carrera_Entre_Cruces_Horizontal_Peso_pulg":
                resultado.get(
                    "Carrera_Entre_Cruces_Horizontal_Peso_pulg",
                    np.nan,
                ),
            "Posicion_Cruce_Superior_Izquierda_pulg":
                resultado.get(
                    "Posicion_Cruce_Superior_Izquierda_pulg", np.nan
                ),
            "Posicion_Cruce_Superior_Derecha_pulg":
                resultado.get(
                    "Posicion_Cruce_Superior_Derecha_pulg", np.nan
                ),
            "Cantidad_Cruces_Horizontal_Superior":
                resultado.get(
                    "Cantidad_Cruces_Horizontal_Superior", 0
                ),
            "Carrera_Efectiva_Fondo_API_pulg":
                resultado.get("Carrera_Efectiva_Fondo_API_pulg", np.nan),
            "Desplazamiento_Desde_Carrera_Efectiva_API_m3_d":
                resultado.get(
                    "Desplazamiento_Desde_Carrera_Efectiva_API_m3_d",
                    np.nan,
                ),
            "Desplazamiento_Bruto_Geometrico_Calculado_m3_d":
                desplazamiento_geometrico_calculado_m3_d,
            "Desplazamiento_Bruto_Efectivo_Calculado_m3_d":
                desplazamiento_efectivo_calculado_m3_d,
            "Carrera_Total_Fondo_Calculada_pulg":
                resultado.get(
                    "Carrera_Total_Fondo_Calculada_pulg", np.nan
                ),
            "Desplazamiento_Bruto_Total_Calculado_m3_d":
                resultado.get(
                    "Desplazamiento_Bruto_Total_Calculado_m3_d",
                    np.nan,
                ),
            "Escurrimiento_Calculado_m3_d":
                escurrimiento_calculado_m3_d,
            "Llenado_Implicito_Carrera_Efectiva_pct":
                llenado_implicito_carrera_efectiva_pct,
            "Desplazamiento_Bruto_Efectivo_API_m3_d":
                resultado.get(
                    "Desplazamiento_Bruto_Efectivo_API_m3_d", np.nan
                ),
            "Desplazamiento_Bruto_Total_API_m3_d":
                resultado.get(
                    "Desplazamiento_Bruto_Total_API_m3_d", np.nan
                ),
            "Escurrimiento_API_m3_d":
                resultado.get("Escurrimiento_API_m3_d", np.nan),
            "Delta_Desplazamiento_Calculado_vs_API_m3_d": (
                desplazamiento_efectivo_calculado_m3_d
                - pd.to_numeric(
                    resultado.get(
                        "Desplazamiento_Bruto_Efectivo_API_m3_d",
                        np.nan,
                    ),
                    errors="coerce",
                )
            ),
            "Delta_Desplazamiento_Total_Calculado_vs_API_m3_d":
                resultado.get(
                    "Delta_Desplazamiento_Total_Calculado_vs_API_m3_d",
                    np.nan,
                ),
            "Delta_Escurrimiento_Calculado_vs_API_m3_d": (
                escurrimiento_calculado_m3_d
                - pd.to_numeric(
                    resultado.get("Escurrimiento_API_m3_d", np.nan),
                    errors="coerce",
                )
            ),
            "Vacio_Superior_Izquierdo_pct":
                vacio_si,
            "Vacio_Superior_Derecho_pct":
                vacio_sd,
            "Vacio_Inferior_Derecho_pct":
                vacio_id,
            "Despegue_Inferior_Derecho_pct":
                despegue_inferior_derecho,
            "Extension_Despegue_Inferior_Derecho_pct":
                extension_despegue_inferior,
            "Transferencia_Inferior_Sostenida":
                transferencia_inferior_sostenida,
            "Angulo_Lateral_Izquierdo_deg":
                angulo_izq,
            "Angulo_Lateral_Derecho_deg":
                angulo_der,
            "Pendiente_Transferencia_Derecha":
                pendiente_transferencia,
            "Curvatura_Transferencia_Derecha":
                curvatura_transferencia,
            "Ancho_Transferencia_20_80_pct":
                ancho_transferencia_20_80,
            "Inicio_Transferencia_Derecha_pct":
                inicio_transferencia_derecha,
            "Angulo_Ideal_Izquierdo_deg":
                angulo_ideal_izquierdo,

            "Angulo_Ideal_Derecho_deg":
                angulo_ideal_derecho,
        })


    diagnosticos_cartas = pd.DataFrame(
        filas_diagnosticos
    )

    # ------------------------------------------------------------
    # PROPAGACIÓN TEMPORAL DE FRICCIÓN DESACTIVADA
    # ------------------------------------------------------------
    # Una cubeta suave aislada es muy frecuente y no alcanza para
    # diagnosticar fricción. Solo se propaga la alerta cuando el mismo
    # pozo presenta al menos dos cartas con firma geométrica fuerte y la
    # carta candidata está temporalmente próxima a esas observaciones.
    if False and (
        not diagnosticos_cartas.empty
        and not base_diagnosticos.empty
        and "Friccion_Elevada_Geometrica"
            in base_diagnosticos.columns
    ):
        campos_contexto_friccion = [
            "CartaId",
            "Friccion_Elevada_Geometrica",
            "Arqueo_Superior_Friccion_pct_gap",
            "Arqueo_Inferior_Friccion_pct_gap",
            "Curvatura_Superior_Friccion",
            "Curvatura_Inferior_Friccion",
        ]

        contexto_friccion = (
            base_diagnosticos[
                [
                    campo
                    for campo in campos_contexto_friccion
                    if campo in base_diagnosticos.columns
                ]
            ]
            .drop_duplicates("CartaId", keep="last")
            .set_index("CartaId")
        )

        fechas_contexto = pd.to_datetime(
            diagnosticos_cartas["Fecha"],
            errors="coerce",
        )

        for pozo, indices_pozo in diagnosticos_cartas.groupby(
            "Pozo",
            sort=False,
        ).groups.items():
            indices_pozo = list(indices_pozo)
            semillas = []

            for indice in indices_pozo:
                carta_id = diagnosticos_cartas.at[indice, "CartaId"]
                if carta_id not in contexto_friccion.index:
                    continue

                fila_contexto = contexto_friccion.loc[carta_id]
                es_semilla = bool(
                    fila_contexto.get(
                        "Friccion_Elevada_Geometrica",
                        False,
                    )
                )
                es_semilla = bool(
                    es_semilla
                    and not bool(
                        diagnosticos_cartas.at[
                            indice,
                            "Carta_No_Valida",
                        ]
                    )
                    and not bool(
                        diagnosticos_cartas.at[
                            indice,
                            "Sin_Trabajo_Bomba",
                        ]
                    )
                    and not bool(
                        diagnosticos_cartas.at[
                            indice,
                            "Golpe_Fluido",
                        ]
                    )
                    and not bool(
                        diagnosticos_cartas.at[
                            indice,
                            "Compresion_Gas",
                        ]
                    )
                )
                if es_semilla:
                    semillas.append(indice)

            # Una única detección fuerte no autoriza a etiquetar el
            # resto de las cartas del pozo.
            if len(semillas) < 2:
                continue

            fechas_semillas = fechas_contexto.loc[semillas].dropna()
            if fechas_semillas.empty:
                continue

            for indice in indices_pozo:
                if indice in semillas:
                    continue

                if any(
                    bool(diagnosticos_cartas.at[indice, campo])
                    for campo in [
                        "Carta_No_Valida",
                        "Sin_Trabajo_Bomba",
                        "Golpe_Fluido",
                        "Compresion_Gas",
                    ]
                ):
                    continue

                fecha_candidata = fechas_contexto.loc[indice]
                if pd.isna(fecha_candidata):
                    continue

                distancia_horas = (
                    (fechas_semillas - fecha_candidata)
                    .abs()
                    .dt.total_seconds()
                    .div(3600.0)
                    .min()
                )
                if not np.isfinite(distancia_horas) or distancia_horas > 6.0:
                    continue

                carta_id = diagnosticos_cartas.at[indice, "CartaId"]
                if carta_id not in contexto_friccion.index:
                    continue
                fila_contexto = contexto_friccion.loc[carta_id]

                arqueo_sup = pd.to_numeric(
                    fila_contexto.get(
                        "Arqueo_Superior_Friccion_pct_gap",
                        np.nan,
                    ),
                    errors="coerce",
                )
                arqueo_inf = pd.to_numeric(
                    fila_contexto.get(
                        "Arqueo_Inferior_Friccion_pct_gap",
                        np.nan,
                    ),
                    errors="coerce",
                )
                curv_sup = pd.to_numeric(
                    fila_contexto.get(
                        "Curvatura_Superior_Friccion",
                        np.nan,
                    ),
                    errors="coerce",
                )
                curv_inf = pd.to_numeric(
                    fila_contexto.get(
                        "Curvatura_Inferior_Friccion",
                        np.nan,
                    ),
                    errors="coerce",
                )
                llenado = pd.to_numeric(
                    diagnosticos_cartas.at[
                        indice,
                        "Llenado_Operativo_pct",
                    ],
                    errors="coerce",
                )
                sumergencia = pd.to_numeric(
                    diagnosticos_cartas.at[
                        indice,
                        "Sumergencia_Relativa_pct",
                    ],
                    errors="coerce",
                )

                forma_compatible = bool(
                    np.isfinite(llenado)
                    and llenado >= 85.0
                    and np.isfinite(sumergencia)
                    and sumergencia >= 20.0
                    and np.isfinite(arqueo_sup)
                    and -2.0 <= arqueo_sup <= 8.0
                    and np.isfinite(arqueo_inf)
                    and 3.5 <= arqueo_inf <= 13.0
                    and np.isfinite(curv_sup)
                    and -0.40 <= curv_sup <= 0.40
                    and np.isfinite(curv_inf)
                    and -0.10 <= curv_inf <= 1.50
                )
                if not forma_compatible:
                    continue

                diagnosticos_cartas.at[
                    indice,
                    "Friccion_Elevada",
                ] = True

                alertas_contexto = diagnosticos_cartas.at[
                    indice,
                    "Alertas",
                ]
                alertas_contexto = (
                    list(alertas_contexto)
                    if isinstance(alertas_contexto, list)
                    else []
                )
                if "Posible fricción elevada" not in alertas_contexto:
                    alertas_contexto.append("Posible fricción elevada")
                diagnosticos_cartas.at[indice, "Alertas"] = alertas_contexto

                evidencias_contexto = diagnosticos_cartas.at[
                    indice,
                    "Evidencias",
                ]
                evidencias_contexto = (
                    list(evidencias_contexto)
                    if isinstance(evidencias_contexto, list)
                    else []
                )
                evidencias_contexto.append(
                    "Patrón suave compatible con fricción y repetido "
                    "en cartas cercanas del mismo pozo"
                )
                diagnosticos_cartas.at[
                    indice,
                    "Evidencias",
                ] = evidencias_contexto

                if (
                    diagnosticos_cartas.at[
                        indice,
                        "Diagnostico_Principal",
                    ]
                    == "Pozo bien explotado"
                ):
                    diagnosticos_cartas.at[
                        indice,
                        "Diagnostico_Principal",
                    ] = "Posible fricción elevada"
                    diagnosticos_cartas.at[
                        indice,
                        "Accion_Sugerida",
                    ] = (
                        "Revisar rozamiento de sarta, tubing, "
                        "alineación y condiciones mecánicas"
                    )
                    diagnosticos_cartas.at[indice, "Confianza"] = 0.62


    # ============================================================
    # INVALIDACIÓN FINAL: SIN TRABAJO O CARTA NULA
    # ============================================================
    # Esta etapa es deliberadamente posterior a todas las correcciones y
    # diagnósticos. Así ninguna rutina anterior puede volver a poblar carta
    # patrón, llenado o sumergencia para una carta sin trabajo hidráulico o
    # geométricamente nula. Los valores originales permanecen en ``datos`` y
    # ``muestra`` como entrada auditable, pero no se publican como resultados.
    mascara_sin_calculo = (
        diagnosticos_cartas["Sin_Trabajo_Bomba"].fillna(False)
        | diagnosticos_cartas["Carta_No_Valida"].fillna(False)
    )
    ids_sin_calculo = set(
        diagnosticos_cartas.loc[
            mascara_sin_calculo, "CartaId"
        ].astype(int)
    )
    motivo_sin_calculo = "SIN_TRABAJO_DE_BOMBA_O_CARTA_NO_VALIDA"

    columnas_numericas_invalidar = {
        # Llenado y derivados directos del llenado.
        "Llenado_Bruto_pct",
        "Llenado_Original_pct",
        "Llenado_Operativo_pct",
        "Llenado_Calculado_pct",
        "Llenado_Usado_pct",
        "Llenado_API_pct",
        "Llenado_Implicito_Carrera_Efectiva_pct",
        "Carrera_Efectiva_Fondo_Calculada_pulg",
        "Desplazamiento_Bruto_Efectivo_Calculado_m3_d",
        "Escurrimiento_Calculado_m3_d",
        # Carta patrón.
        "Area_Ideal",
        "Angulo_Ideal_Izquierdo_deg",
        "Angulo_Ideal_Derecho_deg",
        # Sumergencia propia, SAM, peso y presiones derivadas.
        "Peso_Fluido_Horizontales_lbf",
        "Carga_Hidraulica_Efectiva_lbf",
        "Presion_Diferencial_Horizontales_psi",
        "Sumergencia_Propia_m",
        "Sumergencia_Relativa_Propia_pct",
        "Nivel_Dinamico_Propio_m",
        "Delta_Sumergencia_Propia_vs_API_m",
        "Carga_Roja_Izquierda_SAM_Modificado_lbf",
        "Carga_Roja_Derecha_SAM_Modificado_lbf",
        "Carga_Azul_Izquierda_SAM_Modificado_lbf",
        "Carga_Azul_Derecha_SAM_Modificado_lbf",
        "Posicion_Roja_Izquierda_SAM_Modificado_pulg",
        "Posicion_Roja_Derecha_SAM_Modificado_pulg",
        "Posicion_Azul_Izquierda_SAM_Modificado_pulg",
        "Posicion_Azul_Derecha_SAM_Modificado_pulg",
        "Carga_Superior_SAM_Seleccionada_lbf",
        "Carga_Inferior_SAM_Seleccionada_lbf",
        "Peso_Fluido_SAM_Seleccionado_lbf",
        "Peso_Fluido_Diagnostico_lbf",
        "Diferencial_Carga_SAM_psi",
        "Presion_Descarga_Bomba_SAM_psi",
        "PIP_SAM_Seleccionado_psi",
        "Sumergencia_SAM_Seleccionada_m",
        "Sumergencia_Diagnostico_m",
        "Sumergencia_Relativa_SAM_Seleccionada_pct",
        "Nivel_Dinamico_SAM_Modificado_m",
        "Delta_Sumergencia_SAM_Seleccionada_vs_API_m",
        # La API se conserva en la entrada, pero no se expone como resultado
        # de estas cartas en la tabla diagnóstica.
        "Sumergencia_API_m",
        "Sumergencia_Relativa_pct",
    }

    for tabla in (resultados_cartas, base_diagnosticos, diagnosticos_cartas):
        if tabla.empty or "CartaId" not in tabla.columns:
            continue
        mascara_tabla = tabla["CartaId"].astype(int).isin(ids_sin_calculo)
        columnas_presentes = [
            columna for columna in columnas_numericas_invalidar
            if columna in tabla.columns
        ]
        if columnas_presentes:
            tabla.loc[mascara_tabla, columnas_presentes] = np.nan
        if "Vertices_Ideal" in tabla.columns:
            tabla.loc[mascara_tabla, "Vertices_Ideal"] = None
        if "Calculo_SAM_Modificado_Valido" in tabla.columns:
            tabla.loc[mascara_tabla, "Calculo_SAM_Modificado_Valido"] = False
        if "Motivo_SAM_Modificado_No_Valido" in tabla.columns:
            tabla.loc[
                mascara_tabla, "Motivo_SAM_Modificado_No_Valido"
            ] = motivo_sin_calculo
        if "Calculo_Sumergencia_Propia_Valido" in tabla.columns:
            tabla.loc[
                mascara_tabla, "Calculo_Sumergencia_Propia_Valido"
            ] = False
        if "Motivo_Sumergencia_Propia_No_Valida" in tabla.columns:
            tabla.loc[
                mascara_tabla, "Motivo_Sumergencia_Propia_No_Valida"
            ] = motivo_sin_calculo


    print(
        "Cantidad por diagnóstico principal:"
    )

    display(
        diagnosticos_cartas[
            "Diagnostico_Principal"
        ]
        .value_counts()
        .rename_axis(
            "Diagnostico"
        )
        .reset_index(
            name="Cantidad"
        )
    )


    display(
        diagnosticos_cartas.round(2)
    )

    return {
        "datos": datos,
        "muestra": muestra,
        "invalidas": invalidas,
        "total_declarado": total_declarado,
        "resultados_cartas": resultados_cartas,
        "base_diagnosticos": base_diagnosticos,
        "metricas_cartas": metricas_cartas,
        "diagnosticos_cartas": diagnosticos_cartas,
        "errores_cartas": errores_cartas,
    }
