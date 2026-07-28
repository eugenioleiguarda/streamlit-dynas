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


def procesar_json(origen, silencioso=True):
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
        confiables = not (gap <= 0 or len(evidencias) >= 3)
        return {
            "confiables": bool(confiables),
            "estado": "HORIZONTALES_OK" if confiables else "HORIZONTALES_NO_ENCONTRADAS",
            "evidencias": evidencias,
            "compacidad_carta": compacidad,
            "ratio_gap_api": ratio_gap_api,
            "pendiente_relativa_superior": pendiente_sup,
            "pendiente_relativa_inferior": pendiente_inf,
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
            resultados.append({
                "CartaId": carta_id, "Pozo": carta["Pozo"], "Fecha": carta["Fecha"],
                "GPM": pd.to_numeric(carta.get("GPM"), errors="coerce"),
                "Profundidad_Bomba_m": pd.to_numeric(carta.get("ProfundidadBomba"), errors="coerce"),
                "Diametro_Piston_pulg": pd.to_numeric(carta.get("DiametroPistonBomba"), errors="coerce"),
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
                "Carga_Asc_Geometrica": carga_asc, "Carga_Desc_Geometrica": carga_desc,
                "Separacion_Horizontales": carga_asc - carga_desc,
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

    # Nombres inequívocos de los valores que vamos a utilizar.
    base_diagnosticos["Llenado_Usado_pct"] = base_diagnosticos["Llenado_Calculado_pct"]
    base_diagnosticos["Peso_Fluido_Usado_lbf"] = base_diagnosticos["Peso_Fluido_API_lbf"]
    base_diagnosticos["Sumergencia_Usada_m"] = base_diagnosticos["Sumergencia_API_m"]

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
    base_diagnosticos["Peso_API_Valido"] = (
        np.isfinite(base_diagnosticos["Peso_Fluido_Usado_lbf"])
        & (base_diagnosticos["Peso_Fluido_Usado_lbf"] > 0)
    )
    base_diagnosticos["Sumergencia_API_Valida"] = (
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
        base_diagnosticos["Peso_API_Valido"]
        & base_diagnosticos["Sumergencia_API_Valida"]
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
            profundidad_max_pct >= 12
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
        }
        try:
            asc = resultado["Ascendente"]
            x = np.asarray(asc["posicion"], dtype=float)
            y = np.asarray(asc["carga"], dtype=float)
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
            metrica.get("Sumergencia_Relativa_pct", np.nan),
            errors="coerce",
        )

        sin_trabajo_por_area = bool(
            not carta_no_valida
            and np.isfinite(compacidad_carta)
            and compacidad_carta
                <= UMBRAL_COMPACIDAD_SIN_TRABAJO
        )
        bloqueo_gas_probable = bool(
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
        cierre_tardio_viajera = bool(
            horizontales_ok
            and not sin_trabajo
            and np.isfinite(vacio_si)
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and np.isfinite(carga_inicial_asc_relativa)
            and np.isfinite(posicion_cierre_50)
            and np.isfinite(posicion_cierre_80)
            and vacio_si >= 12.0
            and vacio_sd < 8.0
            and vacio_id < 20.0
            and carga_inicial_asc_relativa <= 0.15
            and posicion_cierre_50 >= 8.0
            and posicion_cierre_80 >= 12.0
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
            and vacio_sd >= 4.0
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
            and vacio_sd >= 4.0
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
            and vacio_sd >= 4.0
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
            carga_inferior=float(resultado["Carga_Desc_Geometrica"]),
        )

        golpe_bomba = bool(
            horizontales_ok
            and (
                metricas_golpe["Golpe_Localizado_Izquierda"]
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

        # Respaldo muy acotado para cartas fronterizas en las que coexisten
        # un golpe de bomba a la izquierda y un llenado incompleto asociado
        # a la transferencia derecha. Algunas de estas cartas no entregan
        # métricas 20-80 estables y, por eso, quedaban únicamente como golpe
        # de bomba aunque conservaran la firma leve de golpe de fluido.
        golpe_fluido_fronterizo_con_golpe_bomba = bool(
            golpe_bomba
            and horizontales_ok
            and not sin_trabajo
            and np.isfinite(llenado)
            and np.isfinite(vacio_sd)
            and np.isfinite(vacio_id)
            and 85.0 <= llenado <= 92.0
            and vacio_sd >= 4.0
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

        sumergencia_relativa = metrica[
            "Sumergencia_Relativa_pct"
        ]

        datos_operativos_validos = bool(
            metrica["Datos_Operativos_Validos"]
        )

        subexplotado = bool(
            horizontales_ok
            and not sin_trabajo
            and not golpe_fluido
            and not compresion_gas
            and not transferencia_inferior_sostenida
            and datos_operativos_validos
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
            "Tubing_Libre":
                tubing_libre,
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
            "Sumergencia_Relativa_pct":
                sumergencia_relativa,
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
