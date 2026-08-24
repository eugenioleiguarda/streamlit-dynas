from __future__ import annotations

import ast
import json
from math import ceil
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pipeline_diagnostico as pipeline_diagnostico_modulo
from vfm_produccion import predecir_vfm
from controles_reales import (
    cruzar_controles,
    leer_controles,
    normalizar_pozo,
)

# Community Cloud carga el módulo desde el commit en cada redeploy. Evitamos
# recargarlo explícitamente porque un rerun puede conservar una referencia
# anterior que ya no coincide con sys.modules y abortar el arranque.
a_array = pipeline_diagnostico_modulo.a_array
procesar_json = pipeline_diagnostico_modulo.procesar_json
calcular_sumergencia_relativa = (
    pipeline_diagnostico_modulo.calcular_sumergencia_relativa
)
calcular_indicadores_moviles_15d = (
    pipeline_diagnostico_modulo.calcular_indicadores_moviles_15d
)
analizar_subexplotacion_temporal = (
    pipeline_diagnostico_modulo.analizar_subexplotacion_temporal
)
analizar_falta_aporte_temporal = (
    pipeline_diagnostico_modulo.analizar_falta_aporte_temporal
)
analizar_bloqueo_temporal = (
    pipeline_diagnostico_modulo.analizar_bloqueo_temporal
)


st.set_page_config(
    page_title="Diagnóstico de pozos",
    page_icon="📈",
    layout="wide",
)

PIPELINE_CACHE_VERSION = (
    "2026-08-24-v87-codos-consolidados-en-grafico-73d1"
)

VENTANA_DIAGNOSTICO_ROBUSTO = 6
MIN_REPETICIONES_DIAGNOSTICO_ROBUSTO = 4

COLORES = {
    "Posible pozo subexplotado": "#16833b",
    "Posible sin trabajo de bomba": "#f59e0b",
    "Carta no válida - posible falla de medición o transmisión": "#b91c1c",
    "Posible golpe de fluido": "#e87918",
    "Posible compresión/interferencia de gas": "#2563eb",
    "Posible pérdida en válvula viajera": "#8b5cf6",
    "Posible golpe de bomba": "#e11d48",
    "Posible tubing libre": "#795548",
    "Posible cierre tardío de válvula viajera": "#a855f7",
    "Posible fricción elevada": "#b45309",
    "Pozo bien explotado": "#14b8a6",
    "Exceso de torque": "#dc2626",
    "Exceso de carga estructural": "#991b1b",
}


def lista_alertas(valor):
    if isinstance(valor, list):
        return [str(x) for x in valor]
    if isinstance(valor, (tuple, set, np.ndarray)):
        return [str(x) for x in valor]
    if valor is None or (np.isscalar(valor) and pd.isna(valor)):
        return []
    if isinstance(valor, str):
        texto = valor.strip()
        if texto.startswith("["):
            try:
                convertido = ast.literal_eval(texto)
                if isinstance(convertido, list):
                    return [str(x) for x in convertido]
            except Exception:
                pass
        return [texto] if texto else []
    return [str(valor)]


@st.cache_data(show_spinner=False)
def ejecutar_pipeline(
    contenido: bytes,
    version: str,
    presion_tubing_kg_cm2: float,
    presion_casing_kg_cm2: float,
    gravedad_especifica_sam: float,
    gradiente_sam_psi_m: float,
):
    _ = version
    return procesar_json(
        contenido,
        presion_tubing_kg_cm2=presion_tubing_kg_cm2,
        presion_casing_kg_cm2=presion_casing_kg_cm2,
        gravedad_especifica_sam=gravedad_especifica_sam,
        gradiente_sam_psi_m=gradiente_sam_psi_m,
    )


@st.cache_data(show_spinner=False)
def ejecutar_vfm(contenido: bytes):
    return predecir_vfm(contenido)


@st.cache_data(show_spinner=False)
def ejecutar_controles(contenido: bytes | None):
    return leer_controles(contenido)


@st.cache_data(show_spinner=False)
def leer_tendencias(contenido: bytes | None):
    """Lee el CSV histórico generado por descargar_tendencias_30_dias.ps1."""
    if contenido is None:
        return pd.DataFrame()

    from io import BytesIO

    tabla = pd.read_csv(
        BytesIO(contenido),
        sep=";",
        encoding="utf-8-sig",
    )

    obligatorias = {"Pozo", "Fecha"}
    faltantes = obligatorias.difference(tabla.columns)
    if faltantes:
        raise ValueError(
            "El CSV de tendencias no contiene: "
            + ", ".join(sorted(faltantes))
        )

    tabla["Fecha"] = pd.to_datetime(
        tabla["Fecha"],
        errors="coerce",
    )
    tabla = tabla.dropna(
        subset=["Pozo", "Fecha"]
    ).copy()
    tabla["Pozo_Clave"] = tabla["Pozo"].map(
        normalizar_pozo
    )

    for columna in tabla.columns:
        if columna in {
            "CartaId",
            "Pozo",
            "Fecha",
            "Pozo_Clave",
        }:
            continue

        tabla[columna] = pd.to_numeric(
            tabla[columna]
                .astype(str)
                .str.replace(",", ".", regex=False),
            errors="coerce",
        )

    if {
        "Sumergencia_API_m",
        "Profundidad_Bomba_m",
    }.issubset(tabla.columns):
        tabla["Sumergencia_Relativa_API_pct"] = (
            calcular_sumergencia_relativa(
                tabla["Sumergencia_API_m"],
                tabla["Profundidad_Bomba_m"],
            )
        )

    return (
        tabla
        .sort_values(["Pozo", "Fecha"])
        .drop_duplicates("CartaId", keep="last")
        .reset_index(drop=True)
    )


def figura_tendencia_diaria(
    tabla: pd.DataFrame,
    series: list[tuple[str, str, str]],
    titulo: str,
    unidad: str,
):
    """Grafica todas las mediciones disponibles en orden cronológico."""
    figura = go.Figure()

    if tabla.empty:
        return figura

    disponibles = [
        (columna, etiqueta, color)
        for columna, etiqueta, color in series
        if columna in tabla.columns
        and tabla[columna].notna().any()
    ]

    for columna, etiqueta, color in disponibles:
        mediciones = (
            tabla[["Fecha", columna]]
            .dropna()
            .sort_values("Fecha")
        )

        figura.add_trace(
            go.Scattergl(
                x=mediciones["Fecha"],
                y=mediciones[columna],
                mode="lines+markers",
                name=etiqueta,
                line=dict(color=color, width=1.5),
                marker=dict(size=5, opacity=0.75),
                hovertemplate=(
                    "%{x|%d/%m/%Y %H:%M:%S}<br>"
                    + etiqueta
                    + ": %{y:.2f} "
                    + unidad
                    + "<extra></extra>"
                ),
            )
        )

    figura.update_layout(
        title=titulo,
        height=350,
        hovermode="x unified",
        xaxis_title="Fecha",
        yaxis_title=(
            f"{titulo} [{unidad}]"
            if unidad
            else titulo
        ),
        legend=dict(
            orientation="h",
            y=1.14,
            x=1,
            xanchor="right",
        ),
        margin=dict(l=45, r=25, t=70, b=40),
        template="plotly_white",
    )

    return figura


def fecha_referencia_json(contenido: bytes):
    try:
        respuesta = json.loads(contenido.decode("utf-8-sig"))
        items = respuesta.get("items", []) if isinstance(respuesta, dict) else respuesta
        fechas = pd.to_datetime(
            [x.get("Fecha") for x in items if isinstance(x, dict)],
            errors="coerce",
        )
        return fechas.max() if len(fechas) else pd.NaT
    except Exception:
        return pd.NaT


def excluir_vfm_cartas_invalidas(salida, produccion):
    if produccion is None or produccion.empty:
        return produccion
    diagnosticos = salida["diagnosticos_cartas"].copy()
    if "Carta_No_Valida" not in diagnosticos:
        return produccion
    diagnosticos["Fecha_Dia"] = pd.to_datetime(
        diagnosticos["Fecha"], errors="coerce"
    ).dt.normalize()
    validas = (
        diagnosticos.assign(
            Carta_Valida=~diagnosticos["Carta_No_Valida"].fillna(False)
        )
        .groupby(["Pozo", "Fecha_Dia"], dropna=False)["Carta_Valida"]
        .any()
        .rename("Hay_Carta_Valida")
        .reset_index()
    )
    salida_vfm = produccion.merge(
        validas,
        on=["Pozo", "Fecha_Dia"],
        how="left",
    )
    return salida_vfm.loc[
        salida_vfm["Hay_Carta_Valida"].fillna(False)
    ].drop(columns="Hay_Carta_Valida")


def construir_tabla_cartas(
    salida,
    produccion,
    controles,
    presion_tubing_kg_cm2=10.0,
    presion_casing_kg_cm2=10.0,
    gradiente_psi_m=1.411,
):
    muestra = salida["muestra"].copy()
    diagnosticos = salida["diagnosticos_cartas"].copy()

    # Fuente canónica del módulo SAM Modificado. La salida detallada por
    # carta siempre contiene estos campos; se vuelven a incorporar aquí por
    # CartaId para no depender de una consolidación intermedia ni de una
    # salida cacheada que tenga diagnosticos_cartas con un esquema anterior.
    resultados = salida.get("resultados_cartas", pd.DataFrame()).copy()
    columnas_sam = [
        columna for columna in resultados.columns
        if "SAM_" in columna or "SAM_Modificado" in columna
    ]
    if "CartaId" in resultados.columns and columnas_sam:
        sam_por_carta = resultados[
            ["CartaId"] + columnas_sam
        ].drop_duplicates("CartaId")
        columnas_reemplazadas = [
            columna for columna in columnas_sam
            if columna in diagnosticos.columns
        ]
        diagnosticos = diagnosticos.drop(
            columns=columnas_reemplazadas,
            errors="ignore",
        ).merge(
            sam_por_carta,
            on="CartaId",
            how="left",
            validate="one_to_one",
        )
    diagnosticos["Alertas_lista"] = diagnosticos["Alertas"].map(lista_alertas)
    diagnosticos["Diagnosticos_Todos"] = diagnosticos.apply(
        lambda f: list(dict.fromkeys(
            [str(f["Diagnostico_Principal"])] + f["Alertas_lista"]
        )),
        axis=1,
    )

    # Se conserva toda la información original de la API. La interfaz usa
    # solamente algunas columnas, pero la descarga debe poder auditar todos
    # los datos de entrada y todos los resultados calculados.
    metadata = muestra.drop_duplicates("CartaId").copy()
    tabla = diagnosticos.merge(
        metadata,
        on="CartaId",
        how="left",
        suffixes=("", "_API"),
    )

    # Respaldo contra módulos retenidos por Streamlit entre redeploys. Si
    # llegaron el peso y los puntos V2 pero no su sumergencia, la app la
    # reconstruye con las mismas hipótesis globales del pipeline.
    peso_v2 = pd.to_numeric(
        tabla.get(
            "Peso_Fluido_SAM_V2_lbf",
            pd.Series(np.nan, index=tabla.index),
        ), errors="coerce",
    )
    profundidad_v2 = pd.to_numeric(
        tabla.get(
            "Profundidad_Bomba_m",
            tabla.get(
                "ProfundidadBomba",
                pd.Series(np.nan, index=tabla.index),
            ),
        ), errors="coerce",
    )
    diametro_v2 = pd.to_numeric(
        tabla.get(
            "Diametro_Piston_pulg",
            tabla.get(
                "DiametroPistonBomba",
                pd.Series(np.nan, index=tabla.index),
            ),
        ), errors="coerce",
    )
    area_v2 = np.pi * diametro_v2.pow(2) / 4.0
    gradiente_v2 = float(gradiente_psi_m)
    presion_descarga_v2 = (
        float(presion_tubing_kg_cm2) * 14.223343307
        + gradiente_v2 * profundidad_v2
    )
    pip_v2 = presion_descarga_v2 - peso_v2 / area_v2
    sumergencia_v2_respaldo = (
        pip_v2 - float(presion_casing_kg_cm2) * 14.223343307
    ) / gradiente_v2
    valida_v2 = (
        peso_v2.gt(0)
        & profundidad_v2.gt(0)
        & diametro_v2.gt(0)
        & np.isfinite(sumergencia_v2_respaldo)
    )
    for campo, respaldo in (
        ("Sumergencia_SAM_V2_m", sumergencia_v2_respaldo),
        (
            "Sumergencia_Relativa_SAM_V2_pct",
            100.0 * sumergencia_v2_respaldo / profundidad_v2,
        ),
    ):
        existente = pd.to_numeric(
            tabla.get(campo, pd.Series(np.nan, index=tabla.index)),
            errors="coerce",
        )
        tabla[campo] = existente.where(
            existente.notna(), respaldo.where(valida_v2)
        )

    # La carrera geométrica propia es el recorrido total medido por la
    # propia carta de fondo: máximo(PosicionesFondo) - mínimo(PosicionesFondo).
    # Normalmente ya llega calculada desde el pipeline. Este respaldo evita
    # que el campo quede vacío si se pierde durante la consolidación con los
    # metadatos del JSON o si se procesa una salida antigua almacenada.
    carrera_geometrica_desde_json = tabla.apply(
        carrera_fondo_carta,
        axis=1,
    )
    columna_carrera_geometrica = (
        "Carrera_Geometrica_Fondo_Calculada_pulg"
    )
    if columna_carrera_geometrica not in tabla.columns:
        tabla[columna_carrera_geometrica] = (
            carrera_geometrica_desde_json
        )
    else:
        tabla[columna_carrera_geometrica] = pd.to_numeric(
            tabla[columna_carrera_geometrica],
            errors="coerce",
        ).fillna(carrera_geometrica_desde_json)

    tabla["Fecha"] = pd.to_datetime(tabla["Fecha"], errors="coerce")
    tabla["Fecha_Dia"] = tabla["Fecha"].dt.normalize()

    comparacion = cruzar_controles(produccion, controles)
    tabla = tabla.merge(
        comparacion.drop(columns=["Pozo_Clave"], errors="ignore"),
        on=["Pozo", "Fecha_Dia"],
        how="left",
        validate="many_to_one",
    )

    # El VFM puede existir para el mismo pozo-día porque otra carta del
    # grupo fue válida. Sin embargo, no debe asociarse visualmente a una
    # carta individual cuya geometría fue declarada inválida.
    if "Carta_No_Valida" in tabla.columns:
        mascara_invalida = tabla["Carta_No_Valida"].fillna(False)
        columnas_vfm_carta = [
            "VFM_Num_Cartas_Dia",
            "VFM_Bruta_m3_d",
            "VFM_Petroleo_m3_d",
            "VFM_Agua_pct",
            "VFM_Bruta_Via_Residuo_m3_d",
            "VFM_Petroleo_Via_Agua_m3_d",
            "Delta_Bruta_m3_d",
            "Error_Bruta_pct",
            "Delta_Petroleo_m3_d",
            "Error_Petroleo_pct",
            "Delta_Agua_pp",
        ]
        columnas_vfm_carta = [
            c for c in columnas_vfm_carta
            if c in tabla.columns
        ]
        tabla.loc[
            mascara_invalida,
            columnas_vfm_carta,
        ] = np.nan
        if "Comentario_VFM_Control" in tabla.columns:
            tabla.loc[
                mascara_invalida,
                "Comentario_VFM_Control",
            ] = (
                "VFM no informado: carta no válida."
            )

    return tabla


def figura_carta(
    carta,
    resultado,
    diagnostico,
    mostrar_horizontales_peso=False,
    mostrar_carta_patrones=True,
    mostrar_sam_v2=False,
):
    # El diagnóstico consolidado es la fuente canónica de las horizontales y
    # sus codos. Algunas correcciones físicas se aplican durante la
    # consolidación; por eso no se debe volver a dibujar un punto anterior
    # conservado en resultados_cartas. La salida técnica queda únicamente
    # como respaldo para campos ausentes.
    if resultado is None:
        resultado_grafico = diagnostico
    else:
        resultado_grafico = resultado.copy()
        columnas_sam_diagnostico = [
            columna for columna in diagnostico.index
            if "SAM_" in columna or "SAM_Modificado" in columna
        ]
        for columna in columnas_sam_diagnostico:
            valor = diagnostico.get(columna)
            if np.isscalar(valor) and not pd.isna(valor):
                resultado_grafico[columna] = valor
    resultado = resultado_grafico

    x = a_array(carta["Fondo_Posiciones"])
    y = a_array(carta["Fondo_Cargas"])
    principal = diagnostico.get("Diagnostico_Principal", "Pozo bien explotado")
    color = COLORES.get(principal, "#374151")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.append(x, x[0]),
        y=np.append(y, y[0]),
        mode="lines",
        name="Carta real",
        line=dict(color="#374151", width=2),
        fill="toself",
        fillcolor="rgba(92,150,190,.13)",
    ))
    sin_trabajo = bool(
        diagnostico.get("Sin_Trabajo_Bomba", False)
    )
    carta_no_valida = bool(
        diagnostico.get("Carta_No_Valida", False)
    )
    mostrar_carta_ideal = bool(
        mostrar_carta_patrones
        and not sin_trabajo
        and not carta_no_valida
    )
    vertices = (
        resultado.get("Vertices_Ideal")
        if resultado is not None and mostrar_carta_ideal
        else None
    )
    try:
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim == 2 and len(vertices) >= 4:
            vertices = np.vstack([vertices[:4], vertices[0]])
            fig.add_trace(go.Scatter(
                x=vertices[:, 0],
                y=vertices[:, 1],
                mode="lines",
                name="Carta patrones de cálculo",
                line=dict(color="#9063cd", width=3, dash="dash"),
            ))
    except Exception:
        pass

    # Estas rectas son exclusivamente las usadas por el calculo propio de
    # peso de fluido y sumergencia. No participan del diagnostico de carta.
    if mostrar_horizontales_peso and resultado is not None:
        # Auditoria visual del orden de adquisicion utilizado por el pipeline.
        # Se superponen las dos carreras sobre la carta cerrada sin alterar
        # ningun calculo ni la carta patrones.
        for clave, nombre, color_rama in (
            ("Ascendente", "Carrera ascendente", "#22c55e"),
            ("Descendente", "Carrera descendente", "#fb7185"),
        ):
            rama = resultado.get(clave, {})
            x_rama = np.asarray(rama.get("posicion", []), dtype=float)
            y_rama = np.asarray(rama.get("carga", []), dtype=float)
            cantidad_rama = min(len(x_rama), len(y_rama))
            x_rama = x_rama[:cantidad_rama]
            y_rama = y_rama[:cantidad_rama]
            validos_rama = np.isfinite(x_rama) & np.isfinite(y_rama)
            if np.count_nonzero(validos_rama) >= 2:
                fig.add_trace(go.Scatter(
                    x=x_rama[validos_rama],
                    y=y_rama[validos_rama],
                    mode="lines",
                    name=nombre,
                    line=dict(color=color_rama, width=3),
                    hovertemplate=(
                        f"{nombre}<br>Posición: %{{x:.1f}}"
                        "<br>Carga: %{y:.0f}<extra></extra>"
                    ),
                ))
        calculo_valido = bool(
            resultado.get("Calculo_SAM_Modificado_Valido", False)
        )
        carga_superior_peso = pd.to_numeric(
            resultado.get(
                "Carga_Superior_SAM_Seleccionada_lbf",
                np.nan,
            ),
            errors="coerce",
        )
        carga_inferior_peso = pd.to_numeric(
            resultado.get(
                "Carga_Inferior_SAM_Seleccionada_lbf",
                np.nan,
            ),
            errors="coerce",
        )
        if (
            calculo_valido
            and np.isfinite(carga_superior_peso)
            and np.isfinite(carga_inferior_peso)
            and len(x) > 0
        ):
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            fig.add_trace(go.Scatter(
                x=[x_min, x_max],
                y=[carga_superior_peso, carga_superior_peso],
                mode="lines",
                name="SAM Modificado - superior",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=[x_min, x_max],
                y=[carga_inferior_peso, carga_inferior_peso],
                mode="lines",
                name="SAM Modificado - inferior",
                line=dict(color="#06b6d4", width=2, dash="dot"),
            ))
            puntos_rojos_x = pd.to_numeric(pd.Series([
                resultado.get("Posicion_Roja_Izquierda_SAM_Modificado_pulg"),
                resultado.get("Posicion_Roja_Derecha_SAM_Modificado_pulg"),
            ]), errors="coerce").to_numpy()
            puntos_rojos_y = pd.to_numeric(pd.Series([
                resultado.get("Carga_Roja_Izquierda_SAM_Modificado_lbf"),
                resultado.get("Carga_Roja_Derecha_SAM_Modificado_lbf"),
            ]), errors="coerce").to_numpy()
            puntos_azules_x = pd.to_numeric(pd.Series([
                resultado.get("Posicion_Azul_Izquierda_SAM_Modificado_pulg"),
                resultado.get("Posicion_Azul_Derecha_SAM_Modificado_pulg"),
            ]), errors="coerce").to_numpy()
            puntos_azules_y = pd.to_numeric(pd.Series([
                resultado.get("Carga_Azul_Izquierda_SAM_Modificado_lbf"),
                resultado.get("Carga_Azul_Derecha_SAM_Modificado_lbf"),
            ]), errors="coerce").to_numpy()
            if not bool(resultado.get(
                "Azul_Izquierdo_Incluido_SAM_Modificado", True
            )):
                puntos_azules_x = puntos_azules_x[1:]
                puntos_azules_y = puntos_azules_y[1:]
            if np.isfinite(puntos_rojos_x).all() and np.isfinite(puntos_rojos_y).all():
                fig.add_trace(go.Scatter(
                    x=puntos_rojos_x, y=puntos_rojos_y,
                    mode="markers", name="Codos superiores",
                    marker=dict(color="#ef4444", size=8),
                ))
            if np.isfinite(puntos_azules_x).all() and np.isfinite(puntos_azules_y).all():
                fig.add_trace(go.Scatter(
                    x=puntos_azules_x, y=puntos_azules_y,
                    mode="markers", name="Codos inferiores",
                    marker=dict(color="#38bdf8", size=8),
                ))
    # Auditoría experimental en modo sombra. Estos puntos no intervienen
    # en el peso, la sumergencia, el llenado ni los diagnósticos oficiales.
    if mostrar_sam_v2 and resultado is not None:
        if bool(resultado.get("Calculo_SAM_V2_Valido", False)):
            x_v2_rojos = pd.to_numeric(pd.Series([
                resultado.get("Posicion_Roja_Izquierda_SAM_V2_pulg"),
                resultado.get("Posicion_Roja_Derecha_SAM_V2_pulg"),
            ]), errors="coerce").to_numpy()
            y_v2_rojos = pd.to_numeric(pd.Series([
                resultado.get("Carga_Roja_Izquierda_SAM_V2_lbf"),
                resultado.get("Carga_Roja_Derecha_SAM_V2_lbf"),
            ]), errors="coerce").to_numpy()
            x_v2_azules = pd.to_numeric(pd.Series([
                resultado.get("Posicion_Azul_Izquierda_SAM_V2_pulg"),
                resultado.get("Posicion_Azul_Derecha_SAM_V2_pulg"),
            ]), errors="coerce").to_numpy()
            y_v2_azules = pd.to_numeric(pd.Series([
                resultado.get("Carga_Azul_Izquierda_SAM_V2_lbf"),
                resultado.get("Carga_Azul_Derecha_SAM_V2_lbf"),
            ]), errors="coerce").to_numpy()
            sup_v2 = pd.to_numeric(
                resultado.get("Carga_Superior_SAM_V2_lbf"), errors="coerce"
            )
            inf_v2 = pd.to_numeric(
                resultado.get("Carga_Inferior_SAM_V2_lbf"), errors="coerce"
            )
            if len(x) and np.isfinite([sup_v2, inf_v2]).all():
                limites_x = [float(np.nanmin(x)), float(np.nanmax(x))]
                fig.add_trace(go.Scatter(
                    x=limites_x, y=[sup_v2, sup_v2], mode="lines",
                    name="V2 sombra - superior",
                    line=dict(color="#fde047", width=2, dash="dash"),
                ))
                fig.add_trace(go.Scatter(
                    x=limites_x, y=[inf_v2, inf_v2], mode="lines",
                    name="V2 sombra - inferior",
                    line=dict(color="#67e8f9", width=2, dash="dash"),
                ))
            if np.isfinite(x_v2_rojos).all() and np.isfinite(y_v2_rojos).all():
                fig.add_trace(go.Scatter(
                    x=x_v2_rojos, y=y_v2_rojos, mode="markers",
                    name="V2 extremos superiores",
                    marker=dict(color="#fde047", size=11, symbol="diamond-open"),
                ))
            if np.isfinite(x_v2_azules).all() and np.isfinite(y_v2_azules).all():
                fig.add_trace(go.Scatter(
                    x=x_v2_azules, y=y_v2_azules, mode="markers",
                    name="V2 extremos inferiores",
                    marker=dict(color="#67e8f9", size=11, symbol="diamond-open"),
                ))
    fecha = pd.to_datetime(diagnostico.get("Fecha"), errors="coerce")
    fecha_texto = fecha.strftime("%d/%m/%Y %H:%M") if pd.notna(fecha) else ""
    fig.update_layout(
        title=dict(
            text=(
                f"{carta['Pozo']} · Carta {int(carta['CartaId'])}"
                f"<br><sup>{fecha_texto} · {principal}</sup>"
            ),
            font=dict(color=color, size=15),
        ),
        height=340,
        margin=dict(l=35, r=20, t=72, b=105),
        xaxis_title="Posición",
        yaxis_title="Carga",
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0,
            xanchor="left",
            yanchor="top",
        ),
        template="plotly_white",
    )
    return fig


def valor_texto(valor, formato=".1f", sufijo=""):
    return (
        f"{valor:{formato}}{sufijo}"
        if pd.notna(valor) and np.isfinite(float(valor))
        else "—"
    )


def diagnostico_consolidado(cartas_ultimas):
    """
    Regla de consolidación:
    - robusto: aparece en al menos 4 de las últimas 6 cartas;
    - variabilidad: tres o más diagnósticos principales distintos,
      o CV del VFM bruto superior al 25 %.
    """
    if cartas_ultimas.empty:
        return "Sin cartas", "No hay información suficiente.", False

    conteos = {}
    for lista in cartas_ultimas["Diagnosticos_Todos"]:
        for diagnostico in set(lista):
            conteos[diagnostico] = conteos.get(diagnostico, 0) + 1

    robustos = sorted(
        [
            (d, n)
            for d, n in conteos.items()
            if n >= MIN_REPETICIONES_DIAGNOSTICO_ROBUSTO
        ],
        key=lambda x: (-x[1], x[0]),
    )
    principales_distintos = cartas_ultimas["Diagnostico_Principal"].nunique()
    vfm = pd.to_numeric(
        cartas_ultimas.get("VFM_Bruta_m3_d"), errors="coerce"
    ).dropna()
    cv = (
        100 * vfm.std(ddof=0) / vfm.mean()
        if len(vfm) >= 2 and vfm.mean() > 0
        else np.nan
    )
    variable = bool(
        principales_distintos >= 3
        or (np.isfinite(cv) and cv > 25)
    )

    if robustos:
        texto = " · ".join(f"{d} ({n}/{len(cartas_ultimas)})" for d, n in robustos)
        return "Diagnóstico robusto", texto, variable
    if variable:
        return (
            "Comportamiento variable",
            "No hay un diagnóstico repetido en 4 cartas y existe variación entre mediciones.",
            True,
        )
    return (
        "Sin diagnóstico robusto",
        "Ningún diagnóstico aparece en al menos 4 de las cartas analizadas.",
        False,
    )


ACCIONES_POR_DIAGNOSTICO_ROBUSTO = {
    "Carta no válida - posible falla de medición o transmisión": (
        "Revisar celda de carga, sensor de posición, sincronización y transmisión de datos"
    ),
    "Posible sin trabajo de bomba": "Revisar bomba, sarta y carta de superficie",
    "Posible pozo subexplotado": "Evaluar aumento de régimen y revisar alertas secundarias",
    "Posible golpe de fluido": "Evaluar disminución de régimen",
    "Posible compresión/interferencia de gas": (
        "Evaluar condición de admisión y revisar régimen"
    ),
    "Posible pérdida en válvula viajera": "Revisar válvula viajera",
    "Posible cierre tardío de válvula viajera": (
        "Revisar válvula viajera, suciedad y dispositivo mecánico antibloqueo de gas"
    ),
    "Posible golpe de bomba": "Revisar espaciamiento",
    "Posible tubing libre": "Revisar condición y anclaje del tubing",
    "Posible fricción elevada": (
        "Revisar rozamiento de sarta, tubing, alineación y condiciones mecánicas"
    ),
    "Exceso de torque": "Revisar balanceo, régimen y capacidad de la caja reductora",
    "Exceso de carga estructural": (
        "Revisar carga admisible de la unidad y condición estructural"
    ),
    "Pozo bien explotado": "Mantener seguimiento operativo",
}


def diagnosticos_robustos_ventana(cartas):
    conteos = {}
    for lista in cartas.get("Diagnosticos_Todos", pd.Series(dtype=object)):
        for diagnostico in set(lista_alertas(lista)):
            conteos[diagnostico] = conteos.get(diagnostico, 0) + 1
    return [
        diagnostico
        for diagnostico, cantidad in sorted(
            conteos.items(), key=lambda item: (-item[1], item[0])
        )
        if cantidad >= MIN_REPETICIONES_DIAGNOSTICO_ROBUSTO
    ]


def accion_para_diagnosticos_robustos(diagnosticos):
    acciones = []
    for diagnostico in diagnosticos:
        accion = ACCIONES_POR_DIAGNOSTICO_ROBUSTO.get(diagnostico)
        if accion and accion not in acciones:
            acciones.append(accion)
    return (
        " · ".join(acciones)
        if acciones
        else "Mantener seguimiento hasta confirmar un diagnóstico robusto"
    )


def _pendiente_theil_sen_local(dias, valores):
    pendientes = []
    for indice in range(len(valores) - 1):
        delta_dias = dias[indice + 1:] - dias[indice]
        validos = delta_dias > 0
        if np.any(validos):
            pendientes.extend(
                ((valores[indice + 1:][validos] - valores[indice])
                 / delta_dias[validos]).tolist()
            )
    return float(np.median(pendientes)) if pendientes else np.nan


def resumen_sumergencia_pozo(cartas_pozo):
    trabajo = cartas_pozo.copy()
    trabajo["Fecha"] = pd.to_datetime(trabajo["Fecha"], errors="coerce")
    trabajo["Sumergencia"] = pd.to_numeric(
        trabajo.get("Sumergencia_SAM_Seleccionada_m"), errors="coerce"
    )
    trabajo["Sumergencia_pct"] = pd.to_numeric(
        trabajo.get("Sumergencia_Relativa_SAM_Seleccionada_pct"),
        errors="coerce",
    )
    trabajo = trabajo.dropna(subset=["Fecha"]).sort_values("Fecha")
    ultimas_seis = trabajo.tail(VENTANA_DIAGNOSTICO_ROBUSTO)
    valores_seis = ultimas_seis["Sumergencia"].dropna()
    representativa = (
        float(valores_seis.mean()) if not valores_seis.empty else np.nan
    )
    porcentajes_seis = ultimas_seis.loc[
        ultimas_seis["Sumergencia"].notna(), "Sumergencia_pct"
    ].dropna()
    representativa_pct = (
        float(porcentajes_seis.mean()) if not porcentajes_seis.empty else np.nan
    )
    negativas_seis = int((valores_seis < 0).sum())

    validas = trabajo.dropna(subset=["Sumergencia"])
    if validas.empty:
        return {
            "representativa": np.nan,
            "representativa_pct": np.nan,
            "muestras_representativa": 0,
            "estado": "Sin datos suficientes",
            "detalle": "No hay sumergencias SAM válidas para analizar.",
            "advertencia": "Validar datos cargados y horizontales SAM.",
            "color": "#e87918",
        }

    fecha_final = validas["Fecha"].max()
    fecha_corte = fecha_final - pd.Timedelta(days=15)
    cubre_15_dias = bool(validas["Fecha"].min() <= fecha_corte)
    ventana_15d = validas.loc[validas["Fecha"] >= fecha_corte].copy()
    usar_15d = bool(cubre_15_dias and len(ventana_15d) >= 2)
    ventana = ventana_15d if usar_15d else validas.copy()
    etiqueta_ventana = (
        "últimos 15 días" if usar_15d else "cartas disponibles"
    )
    ventana["Dia"] = ventana["Fecha"].dt.floor("D")
    diaria = (
        ventana.groupby("Dia", as_index=False)["Sumergencia"]
        .median().sort_values("Dia")
    )
    valores = diaria["Sumergencia"].to_numpy(dtype=float)
    dias = (
        (diaria["Dia"] - diaria["Dia"].min())
        .dt.total_seconds().to_numpy(dtype=float) / 86400.0
    )
    pendiente = _pendiente_theil_sen_local(dias, valores)
    lapso = float(np.ptp(dias)) if len(dias) >= 2 else 0.0
    mediana = float(np.median(valores))
    mad = float(np.median(np.abs(valores - mediana)))
    ruido_robusto = 1.4826 * mad
    cambio_estimado = pendiente * lapso if np.isfinite(pendiente) else np.nan
    umbral_cambio = max(15.0, 0.10 * max(abs(mediana), 100.0))
    volatil = bool(
        len(valores) >= 3
        and ruido_robusto > max(20.0, 0.20 * max(abs(mediana), 100.0))
        and (
            not np.isfinite(cambio_estimado)
            or abs(cambio_estimado) < 2.0 * ruido_robusto
        )
    )
    if len(valores) < 2 or lapso <= 0:
        estado = "Sin datos suficientes"
        color = "#e87918"
    elif volatil:
        estado = "Volátil"
        color = "#e87918"
    elif abs(cambio_estimado) <= umbral_cambio:
        estado = "Estable"
        color = "#16833b"
    elif pendiente > 0:
        estado = "Subiendo"
        color = "#2563eb"
    else:
        estado = "Bajando"
        color = "#dc2626"

    unidad_dias = "día" if len(diaria) == 1 else "días"
    detalle = (
        f"{estado} en {etiqueta_ventana}: "
        f"{len(ventana)} cartas, {len(diaria)} {unidad_dias} con datos"
    )
    if np.isfinite(pendiente):
        detalle += f", tendencia {pendiente:+.1f} m/día"
    detalle += "."

    if np.isfinite(representativa) and representativa < 0:
        advertencia = (
            "Sumergencia representativa negativa: validar datos cargados, "
            "diámetro/área de pistón y ubicación de las horizontales SAM."
        )
    elif negativas_seis:
        advertencia = (
            f"{negativas_seis} de {len(valores_seis)} sumergencias recientes "
            "son negativas; revisar datos y horizontales de esas cartas."
        )
    else:
        advertencia = "Sin sumergencias negativas en las últimas seis cartas válidas."
    return {
        "representativa": representativa,
        "representativa_pct": representativa_pct,
        "muestras_representativa": int(len(valores_seis)),
        "estado": estado,
        "detalle": detalle,
        "advertencia": advertencia,
        "color": color,
    }


def tabla_diagnosticos_robustos(historico):
    """
    Una fila por pozo. Se consideran las últimas seis cartas y un
    diagnóstico es robusto cuando aparece al menos cuatro veces.
    """
    filas = []
    for pozo, grupo in historico.groupby("Pozo", sort=True):
        ultimas = grupo.nlargest(VENTANA_DIAGNOSTICO_ROBUSTO, "Fecha")
        conteos = {}
        for lista in ultimas["Diagnosticos_Todos"]:
            for diagnostico in set(lista_alertas(lista)):
                conteos[diagnostico] = conteos.get(diagnostico, 0) + 1
        robustos = sorted(
            diagnostico
            for diagnostico, cantidad in conteos.items()
            if cantidad >= MIN_REPETICIONES_DIAGNOSTICO_ROBUSTO
        )
        filas.append({
            "Pozo": pozo,
            "Cantidad_Cartas": int(len(grupo)),
            "Cartas_Analizadas_Robustez": int(len(ultimas)),
            "Diagnosticos_Robustos_Lista": robustos,
            "Tiene_Diagnostico_Robusto": bool(robustos),
        })
    return pd.DataFrame(filas)


def carrera_bomba(fila):
    pares = [
        ("CarreraEfectivaBombaInicio", "CarreraEfectivaBombaFin"),
        ("CarreraMinimaBomba", "CarreraMaximaBomba"),
    ]
    for inicio, fin in pares:
        if inicio in fila and fin in fila:
            a = pd.to_numeric(pd.Series([fila.get(inicio)]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([fila.get(fin)]), errors="coerce").iloc[0]
            if pd.notna(a) and pd.notna(b):
                return float(b - a)
    return np.nan


def carrera_superficie(fila):
    """Carrera de superficie informada por la API, en pulgadas."""
    minima = pd.to_numeric(
        pd.Series([fila.get("CarreraMinimaSuperficie")]),
        errors="coerce",
    ).iloc[0]
    maxima = pd.to_numeric(
        pd.Series([fila.get("CarreraMaximaSuperficie")]),
        errors="coerce",
    ).iloc[0]
    if pd.notna(minima) and pd.notna(maxima):
        return float(maxima - minima)
    return np.nan


def carrera_fondo_carta(carta):
    """Recorrido de fondo de una carta: máximo menos mínimo, en pulgadas."""
    posiciones = a_array(carta.get("Fondo_Posiciones"))
    posiciones = posiciones[np.isfinite(posiciones)]
    if len(posiciones) < 2:
        return np.nan
    return float(np.ptp(posiciones))


def preparar_csv_descarga(tabla):
    """Convierte listas y fechas a valores estables para descargar en CSV."""
    salida = tabla.copy()
    for columna in salida.columns:
        if pd.api.types.is_datetime64_any_dtype(salida[columna]):
            salida[columna] = salida[columna].dt.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        elif salida[columna].dtype == "object":
            salida[columna] = salida[columna].map(
                lambda valor: json.dumps(
                    valor,
                    ensure_ascii=False,
                    default=str,
                )
                if isinstance(valor, (list, tuple, dict, np.ndarray))
                else valor
            )
    return salida


ACCIONES_POR_DIAGNOSTICO = {
    "Posible pozo subexplotado":
        "Evaluar aumento de régimen y revisar alertas secundarias",
    "Posible sin trabajo de bomba":
        "Revisar bomba, sarta y carta de superficie",
    "Carta no válida - posible falla de medición o transmisión":
        "Revisar celda de carga, adquisición y transmisión de datos",
    "Posible golpe de fluido":
        "Evaluar disminución de régimen",
    "Posible compresión/interferencia de gas":
        "Evaluar condición de admisión y revisar régimen",
    "Posible pérdida en válvula viajera":
        "Revisar válvula viajera",
    "Posible golpe de bomba":
        "Revisar espaciamiento",
    "Posible cierre tardío de válvula viajera":
        "Revisar válvula viajera, suciedad y dispositivo antibloqueo de gas",
    "Posible tubing libre":
        "Revisar condición y anclaje del tubing",
    "Exceso de torque":
        "Revisar balanceo, régimen y capacidad de la caja reductora",
    "Exceso de carga estructural":
        "Revisar carga admisible de la unidad y condición estructural",
    "Pozo bien explotado":
        "Mantener seguimiento operativo",
}


def texto_lista(valor):
    elementos = lista_alertas(valor)
    return " | ".join(str(x) for x in elementos)


def construir_exportacion_cartas(cartas):
    salida = cartas.copy()
    salida["Diagnosticos_Todos_Texto"] = salida[
        "Diagnosticos_Todos"
    ].map(texto_lista)
    salida["Alertas_Secundarias_Texto"] = salida.apply(
        lambda fila: " | ".join(
            diagnostico
            for diagnostico in lista_alertas(fila.get("Alertas_lista"))
            if diagnostico != str(fila.get("Diagnostico_Principal"))
        ),
        axis=1,
    )
    if "Evidencias" in salida.columns:
        salida["Evidencias_Texto"] = salida["Evidencias"].map(texto_lista)

    salida["Carrera_Fondo_Carta_pulg"] = salida["CartaId"].map(
        lambda carta_id: carrera_fondo_carta(
            cartas_por_id.get(int(carta_id), {})
        )
    )
    salida["Carrera_Superficie_API_pulg"] = salida.apply(
        carrera_superficie,
        axis=1,
    )

    preferidas = [
        "CartaId",
        "Pozo",
        "Fecha",
        "Diagnostico_Principal",
        "Diagnosticos_Todos_Texto",
        "Alertas_Secundarias_Texto",
        "Accion_Sugerida",
        "Confianza",
        "Evidencias_Texto",
        "Carta_No_Valida",
        "Sin_Trabajo_Bomba",
        "Bloqueo_Gas_Probable",
        "Apertura_Central_Carta",
        "Perdida_Valvula_Viajera",
        "Cierre_Tardio_Valvula_Viajera",
        "Golpe_Fluido",
        "Compresion_Gas",
        "Golpe_Bomba",
        "Tubing_Libre",
        "Pozo_Subexplotado",
        "Exceso_Torque",
        "Exceso_Carga_Estructural",
        "Llenado_Bruto_pct",
        "Llenado_Original_pct",
        "Llenado_Operativo_pct",
        "Sumergencia_Relativa_pct",
        "Sumergencia_API_m",
        "Sumergencia_Propia_m",
        "Sumergencia_Relativa_SAM_Seleccionada_pct",
        "Delta_Sumergencia_SAM_Seleccionada_vs_API_m",
        "Calculo_Sumergencia_Propia_Valido",
        "Motivo_Sumergencia_Propia_No_Valida",
        "Peso_Fluido_Horizontales_lbf",
        "Peso_Fluido_API_lbf",
        "Area_Piston_pulg2",
        "Presion_Diferencial_Horizontales_psi",
        "SG_Fluido_Asumido",
        "Gradiente_Fluido_Asumido_psi_m",
        "Torque_Reductor_pct",
        "Carga_Estructural_pct",
        "GPM",
        "Carrera_Fondo_Carta_pulg",
        "Carrera_Geometrica_Fondo_Calculada_pulg",
        "Carrera_Efectiva_Fondo_Calculada_pulg",
        "Carrera_Superficie_API_pulg",
        "Desplazamiento_Bruto_Geometrico_Calculado_m3_d",
        "Desplazamiento_Bruto_Efectivo_Calculado_m3_d",
        "Desplazamiento_Bruto_Efectivo_API_m3_d",
        "ProfundidadBomba",
        "DiametroPistonBomba",
        "VFM_Bruta_m3_d",
        "VFM_Petroleo_m3_d",
        "VFM_Agua_pct",
        "Control_Bruta_m3_d",
        "Control_Petroleo_m3_d",
        "Control_Agua_pct",
        "Comentario_VFM_Control",
    ]
    preferidas = [c for c in preferidas if c in salida.columns]
    restantes = [c for c in salida.columns if c not in preferidas]
    return salida[preferidas + restantes]


def construir_exportacion_pozos(cartas):
    filas = []
    for pozo, grupo in cartas.groupby("Pozo", sort=True):
        grupo = grupo.sort_values("Fecha", ascending=False)
        ultimas_seis = grupo.head(VENTANA_DIAGNOSTICO_ROBUSTO)
        ultima = grupo.iloc[0]

        conteos = {}
        for diagnosticos in ultimas_seis["Diagnosticos_Todos"]:
            for diagnostico in set(lista_alertas(diagnosticos)):
                conteos[diagnostico] = conteos.get(diagnostico, 0) + 1

        robustos = sorted(
            (
                (diagnostico, cantidad)
                for diagnostico, cantidad in conteos.items()
                if cantidad >= MIN_REPETICIONES_DIAGNOSTICO_ROBUSTO
            ),
            key=lambda item: (-item[1], item[0]),
        )
        diagnosticos_robustos = " | ".join(
            f"{diagnostico} ({cantidad}/{len(ultimas_seis)})"
            for diagnostico, cantidad in robustos
        )
        acciones_robustas = " | ".join(dict.fromkeys(
            ACCIONES_POR_DIAGNOSTICO.get(
                diagnostico,
                "Revisión operativa",
            )
            for diagnostico, _ in robustos
        ))

        fila = {
            "Pozo": pozo,
            "Cantidad_Cartas": int(len(grupo)),
            "Cartas_Analizadas_Robustez": int(len(ultimas_seis)),
            "Fecha_Ultima_Carta": ultima.get("Fecha"),
            "Tiene_Diagnostico_Robusto": bool(robustos),
            "Diagnosticos_Robustos": (
                diagnosticos_robustos
                if robustos
                else "Sin diagnóstico robusto"
            ),
            "Acciones_Diagnosticos_Robustos": (
                acciones_robustas
                if robustos
                else "Mantener seguimiento y revisar variabilidad"
            ),
            "Diagnostico_Ultima_Carta": ultima.get(
                "Diagnostico_Principal"
            ),
            "Diagnosticos_Todos_Ultima_Carta": texto_lista(
                ultima.get("Diagnosticos_Todos")
            ),
            "Accion_Ultima_Carta": ultima.get("Accion_Sugerida"),
        }

        campos_ultima = [
            "GPM",
            "ProfundidadBomba",
            "DiametroPistonBomba",
            "Llenado_Operativo_pct",
            "Sumergencia_Relativa_pct",
            "Sumergencia_API_m",
            "Sumergencia_Propia_m",
            "Sumergencia_Relativa_SAM_Seleccionada_pct",
            "Delta_Sumergencia_SAM_Seleccionada_vs_API_m",
            "Torque_Reductor_pct",
            "Carga_Estructural_pct",
            "VFM_Bruta_m3_d",
            "VFM_Petroleo_m3_d",
            "VFM_Agua_pct",
            "Control_Bruta_m3_d",
            "Control_Petroleo_m3_d",
            "Control_Agua_pct",
            "Comentario_VFM_Control",
        ]
        for campo in campos_ultima:
            if campo in grupo.columns:
                fila[campo] = ultima.get(campo)

        fila["Carrera_Superficie_API_pulg"] = carrera_superficie(ultima)
        filas.append(fila)

    return pd.DataFrame(filas)


st.title("Diagnóstico de pozos y evolución de cartas")
st.caption(
    "El resumen cuenta pozos únicos. El explorador conserva todas las cartas "
    "y el detalle consolida la evolución de cada pozo."
)

st.sidebar.caption(
    f"Versión del algoritmo: {PIPELINE_CACHE_VERSION}"
)
reprocesamiento_solicitado = st.sidebar.button(
    "Reprocesar todos los archivos",
    use_container_width=True,
)
if reprocesamiento_solicitado:
    st.cache_data.clear()
    st.sidebar.info(
        "Reprocesando todos los JSON con la versión actual…"
    )

archivos = st.sidebar.file_uploader(
    "Cargar uno o más JSON de la API",
    type=["json"],
    accept_multiple_files=True,
)
mostrar_horizontales_peso = st.sidebar.checkbox(
    "Mostrar horizontales SAM Modificado",
    value=False,
    help=(
        "Muestra las líneas obtenidas de los cuatro codos laterales. No "
        "modifica la carta patrones de cálculo ni los "
        "diagnósticos."
    ),
)
mostrar_sam_v2 = False
mostrar_carta_patrones = st.sidebar.checkbox(
    "Mostrar carta patrones de cálculo",
    value=True,
    help=(
        "Muestra u oculta la referencia geométrica usada por los cálculos. "
        "Ocultarla no modifica el llenado ni los diagnósticos."
    ),
)
with st.sidebar.expander("Parámetros SAM Modificado", expanded=False):
    presion_tubing_sam_kg_cm2 = st.number_input(
        "Presión de tubing [kg/cm²]",
        min_value=0.0,
        value=10.0,
        step=0.5,
    )
    presion_casing_sam_kg_cm2 = st.number_input(
        "Presión de casing [kg/cm²]",
        min_value=0.0,
        value=10.0,
        step=0.5,
    )
    gravedad_especifica_sam = st.number_input(
        "Gravedad específica del fluido",
        min_value=0.01,
        value=0.994,
        step=0.001,
        format="%.3f",
    )
    gradiente_sam_psi_m = st.number_input(
        "Gradiente de fluido [psi/m]",
        min_value=0.001,
        value=1.411,
        step=0.001,
        format="%.3f",
        help=(
            "Valor global editable. Para SG=0,994, la aproximación "
            "hidrostática es 1,411 psi/m."
        ),
    )
archivo_controles = st.sidebar.file_uploader(
    "Actualizar controles reales (opcional)",
    type=["xlsx"],
    help="Si no se carga un Excel se utiliza controles_reales.xlsx del repositorio.",
)
archivo_tendencias = st.sidebar.file_uploader(
    "Cargar tendencias históricas (opcional)",
    type=["csv"],
    help=(
        "CSV generado por descargar_tendencias_30_dias.ps1. "
        "Se utiliza únicamente para los gráficos históricos."
    ),
)
if not archivos:
    st.info("Cargá uno o más JSON desde la barra lateral.")
    st.stop()

try:
    controles = ejecutar_controles(
        archivo_controles.getvalue() if archivo_controles is not None else None
    )
except Exception as exc:
    st.error("No fue posible leer los controles reales.")
    st.exception(exc)
    st.stop()

try:
    tendencias = leer_tendencias(
        archivo_tendencias.getvalue()
        if archivo_tendencias is not None
        else None
    )
except Exception as exc:
    st.error("No fue posible leer el CSV de tendencias.")
    st.exception(exc)
    st.stop()

tablas = []
vfm_partes = []
cartas_por_id = {}
resultados_por_id = {}
errores_archivos = []

with st.spinner("Procesando cartas, diagnósticos y VFM…"):
    for archivo in archivos:
        try:
            contenido = archivo.getvalue()
            referencia = fecha_referencia_json(contenido)
            salida = ejecutar_pipeline(
                contenido,
                PIPELINE_CACHE_VERSION,
                presion_tubing_sam_kg_cm2,
                presion_casing_sam_kg_cm2,
                gravedad_especifica_sam,
                gradiente_sam_psi_m,
            )
            produccion = excluir_vfm_cartas_invalidas(
                salida,
                ejecutar_vfm(contenido),
            )
            tabla = construir_tabla_cartas(
                salida,
                produccion,
                controles,
                presion_tubing_kg_cm2=presion_tubing_sam_kg_cm2,
                presion_casing_kg_cm2=presion_casing_sam_kg_cm2,
                gradiente_psi_m=gradiente_sam_psi_m,
            )
            tabla["Archivo_Origen"] = archivo.name
            tabla["Fecha_Referencia_JSON"] = referencia
            tablas.append(tabla)

            vfm_parte = cruzar_controles(produccion, controles)
            vfm_parte["Archivo_Origen"] = archivo.name
            vfm_parte["Fecha_Referencia_JSON"] = referencia
            vfm_partes.append(vfm_parte)

            for _, fila in salida["muestra"].iterrows():
                cartas_por_id[int(fila["CartaId"])] = fila
            for _, fila in salida["resultados_cartas"].iterrows():
                resultados_por_id[int(fila["CartaId"])] = fila
        except Exception as exc:
            errores_archivos.append(f"{archivo.name}: {exc}")

if reprocesamiento_solicitado:
    st.sidebar.success("Reprocesamiento terminado.")
    st.toast(
        "Todos los archivos fueron reprocesados con el algoritmo actual.",
        icon="✅",
    )

if not tablas:
    st.error("No fue posible procesar ninguno de los JSON.")
    for error in errores_archivos:
        st.caption(error)
    st.stop()

historico = (
    pd.concat(tablas, ignore_index=True)
    .sort_values(["Fecha", "Fecha_Referencia_JSON"])
    .drop_duplicates("CartaId", keep="last")
    .reset_index(drop=True)
)
vfm_historico = (
    pd.concat(vfm_partes, ignore_index=True)
    .sort_values(["Fecha_Referencia_JSON", "Fecha_Dia"])
    .drop_duplicates(["Pozo", "Fecha_Referencia_JSON"], keep="last")
    .reset_index(drop=True)
)

diagnosticos_disponibles = sorted({
    d
    for lista in historico["Diagnosticos_Todos"]
    for d in lista
})
pozos_disponibles = sorted(historico["Pozo"].dropna().unique())
resumen_robusto_total = tabla_diagnosticos_robustos(historico)
diagnosticos_robustos_disponibles = sorted({
    diagnostico
    for lista in resumen_robusto_total["Diagnosticos_Robustos_Lista"]
    for diagnostico in lista
})

st.sidebar.header("Filtros de pozos")
filtro_robusto = st.sidebar.multiselect(
    "Diagnóstico robusto del pozo",
    diagnosticos_robustos_disponibles,
    help=(
        "Sólo considera diagnósticos presentes en al menos cuatro "
        "de las últimas seis cartas."
    ),
)
filtro_pozo = st.sidebar.multiselect("Pozo", pozos_disponibles)
busqueda = st.sidebar.text_input("Buscar pozo")
st.sidebar.header("Filtro individual por carta")
filtro_individual = st.sidebar.multiselect(
    "Diagnóstico principal o secundario de la carta",
    diagnosticos_disponibles,
    help="Puede incluir diagnósticos débiles observados una o dos veces.",
)
modo_sumergencia = st.sidebar.selectbox(
    "Sumergencia SAM Modificado",
    ["Todas", "Sólo negativas", "Rango personalizado"],
    help=(
        "Filtra por la sumergencia relativa calculada por SAM Modificado. "
        "Las cartas sin cálculo válido se excluyen cuando está activo."
    ),
)
limite_sumergencia_min = None
limite_sumergencia_max = None
if modo_sumergencia == "Rango personalizado":
    columna_limite_izq, columna_limite_der = st.sidebar.columns(2)
    limite_sumergencia_min = columna_limite_izq.number_input(
        "Mínimo [%]", value=-100.0, step=5.0
    )
    limite_sumergencia_max = columna_limite_der.number_input(
        "Máximo [%]", value=100.0, step=5.0
    )

mascara_pozos = pd.Series(True, index=resumen_robusto_total.index)
if filtro_robusto:
    mascara_pozos &= resumen_robusto_total[
        "Diagnosticos_Robustos_Lista"
    ].map(
        lambda xs: any(d in xs for d in filtro_robusto)
    )
if filtro_pozo:
    mascara_pozos &= resumen_robusto_total["Pozo"].isin(filtro_pozo)
if busqueda.strip():
    mascara_pozos &= resumen_robusto_total["Pozo"].astype(str).str.contains(
        busqueda.strip(), case=False, regex=False
    )

# Los filtros de pozo y de robustez determinan primero el universo
# candidato. El filtro individual conserva solamente los pozos que
# tengan al menos una carta coincidente.
pozos_candidatos = sorted(
    resumen_robusto_total.loc[mascara_pozos, "Pozo"].dropna().unique()
)
cartas_candidatas = historico.loc[
    historico["Pozo"].isin(pozos_candidatos)
].copy()

if filtro_individual:
    cartas_filtradas = cartas_candidatas.loc[
        cartas_candidatas["Diagnosticos_Todos"].map(
            lambda xs: any(d in xs for d in filtro_individual)
        )
    ].copy()
else:
    cartas_filtradas = cartas_candidatas.copy()

if modo_sumergencia != "Todas":
    sumergencia_filtro = pd.to_numeric(
        cartas_filtradas.get(
            "Sumergencia_Relativa_SAM_Seleccionada_pct",
            pd.Series(np.nan, index=cartas_filtradas.index),
        ),
        errors="coerce",
    )
    if modo_sumergencia == "Sólo negativas":
        mascara_sumergencia = sumergencia_filtro < 0.0
    else:
        minimo_elegido = min(
            float(limite_sumergencia_min),
            float(limite_sumergencia_max),
        )
        maximo_elegido = max(
            float(limite_sumergencia_min),
            float(limite_sumergencia_max),
        )
        mascara_sumergencia = sumergencia_filtro.between(
            minimo_elegido, maximo_elegido, inclusive="both"
        )
    cartas_filtradas = cartas_filtradas.loc[
        mascara_sumergencia.fillna(False)
    ].copy()

pozos_filtrados = sorted(
    cartas_filtradas["Pozo"].dropna().unique()
)

# El resumen y el detalle conservan todas las cartas de los pozos
# resultantes. El explorador muestra solamente las cartas que cumplen
# los filtros individuales de diagnóstico y sumergencia.
cartas_contexto = historico.loc[
    historico["Pozo"].isin(pozos_filtrados)
].copy()
cartas_filtradas = cartas_filtradas.sort_values(
    ["Pozo", "Fecha"], ascending=[True, False]
)

tab_resumen, tab_explorador, tab_detalle, tab_descargas = st.tabs(
    [
        "Resumen por pozo",
        "Explorador de cartas",
        "Detalle del pozo",
        "Descargas",
    ]
)

with tab_resumen:
    ultimas = (
        cartas_contexto.sort_values("Fecha")
        .drop_duplicates("Pozo", keep="last")
    )
    robustos_por_pozo = []
    variables = 0
    for pozo, grupo in cartas_contexto.groupby("Pozo"):
        seis = grupo.nlargest(VENTANA_DIAGNOSTICO_ROBUSTO, "Fecha")
        estado, texto, variable = diagnostico_consolidado(seis)
        robustos_por_pozo.append({
            "Pozo": pozo,
            "Estado": estado,
            "Detalle": texto,
        })
        variables += int(variable)
    resumen_robusto = pd.DataFrame(robustos_por_pozo)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pozos del filtro", len(pozos_filtrados))
    c2.metric("Cartas del filtro individual", len(cartas_filtradas))
    c3.metric(
        "Pozos con diagnóstico robusto",
        int((resumen_robusto.get("Estado") == "Diagnóstico robusto").sum())
        if not resumen_robusto.empty else 0,
    )
    c4.metric("Pozos con variación", variables)

    st.subheader("Sumergencia SAM Modificado por diagnóstico")
    st.caption(
        "Distribución de sumergencia relativa para golpe/compresión, "
        "pozo subexplotado y pozo bien explotado."
    )
    # Todos los gráficos se construyen sobre las cartas que superaron cada
    # filtro activo (pozo, diagnóstico individual y sumergencia). El contexto
    # histórico completo queda reservado para el resumen y el detalle.
    analisis_sumergencia = cartas_filtradas.copy()
    for destino, origen in {
        "Sumergencia_SAM_m": "Sumergencia_SAM_Seleccionada_m",
        "Sumergencia_SAM_pct": "Sumergencia_Relativa_SAM_Seleccionada_pct",
        "Sumergencia_V2_m": "Sumergencia_SAM_V2_m",
        "Sumergencia_V2_pct": "Sumergencia_Relativa_SAM_V2_pct",
        "Peso_Fluido_V2_lbf": "Peso_Fluido_SAM_V2_lbf",
        "Confianza_V2": "Confianza_SAM_V2",
        "Diametro_Piston_pulg": "DiametroPistonBomba",
        "Profundidad_Bomba_m": "ProfundidadBomba",
        "GPM_Analisis": "GPM",
    }.items():
        analisis_sumergencia[destino] = pd.to_numeric(
            analisis_sumergencia.get(
                origen, pd.Series(np.nan, index=analisis_sumergencia.index)
            ), errors="coerce"
        )

    grupos_sumergencia = [
        ("Golpe de fluido / compresión",
         ["Posible golpe de fluido",
          "Posible compresión/interferencia de gas"],
         (0.0, 10.0), "#f97316"),
        ("Pozo subexplotado", ["Posible pozo subexplotado"],
         (15.0, None), "#16a34a"),
        ("Pozo bien explotado", ["Pozo bien explotado"],
         (0.0, 50.0), "#0ea5e9"),
    ]
    for columna, (nombre, diagnosticos, referencias, color) in zip(
        st.columns(3), grupos_sumergencia
    ):
        valores = analisis_sumergencia.loc[
            analisis_sumergencia["Diagnostico_Principal"].isin(diagnosticos),
            "Sumergencia_SAM_pct",
        ].dropna()
        figura = go.Figure(go.Histogram(
            x=valores, nbinsx=28, marker_color=color, opacity=0.82,
            hovertemplate="Sumergencia: %{x:.1f}%<br>Cartas: %{y}<extra></extra>",
        ))
        for referencia in referencias:
            if referencia is not None:
                figura.add_vline(
                    x=referencia, line_dash="dash", line_color="#6b7280"
                )
        figura.update_layout(
            title=(
                f"{nombre}<br><sup>n={len(valores)} · mediana="
                f"{valor_texto(valores.median(), '.1f', '%')}</sup>"
            ),
            xaxis_title="Sumergencia relativa [%]", yaxis_title="Cartas",
            height=360, margin=dict(l=15, r=15, t=65, b=45),
            showlegend=False, template="plotly_white",
        )
        with columna:
            st.plotly_chart(
                figura, use_container_width=True,
                key=f"hist_sumergencia_{nombre}"
            )

    diagnosticos_graficos = [
        diagnostico
        for _, diagnosticos, _, _ in grupos_sumergencia
        for diagnostico in diagnosticos
    ]
    relaciones = analisis_sumergencia.loc[
        analisis_sumergencia["Diagnostico_Principal"].isin(
            diagnosticos_graficos
        ) & analisis_sumergencia["Sumergencia_SAM_m"].notna()
    ].copy()
    colores_diagnostico = {
        "Posible golpe de fluido": "#f97316",
        "Posible compresión/interferencia de gas": "#2563eb",
        "Posible pozo subexplotado": "#16a34a",
        "Pozo bien explotado": "#0ea5e9",
    }

    def mostrar_relacion(
        campo_x, titulo_x, titulo, key,
        campo_y="Sumergencia_SAM_m",
        titulo_y="Sumergencia SAM Modificado [m]",
        color_por_sumergencia=False,
    ):
        figura = go.Figure()
        if color_por_sumergencia:
            parte = relaciones.dropna(
                subset=[campo_x, campo_y, "Sumergencia_SAM_m"]
            )
            figura.add_trace(go.Scatter(
                x=parte[campo_x], y=parte[campo_y], mode="markers",
                marker=dict(
                    size=8, opacity=0.72,
                    color=parte["Sumergencia_SAM_m"],
                    colorscale="RdYlGn",
                    colorbar=dict(title="Sumergencia [m]"),
                ),
                customdata=np.column_stack([
                    parte["Pozo"].astype(str),
                    parte["CartaId"].astype(str),
                    parte["Sumergencia_SAM_m"],
                    parte["Diagnostico_Principal"].astype(str),
                ]),
                hovertemplate=(
                    "Pozo: %{customdata[0]}<br>Carta: %{customdata[1]}<br>"
                    "Diagnóstico: %{customdata[3]}<br>"
                    + titulo_x + ": %{x:.2f}<br>"
                    + titulo_y + ": %{y:.1f}<br>"
                    "Sumergencia: %{customdata[2]:.1f} m<extra></extra>"
                ),
            ))
        else:
            for diagnostico, color in colores_diagnostico.items():
                parte = relaciones.loc[
                    relaciones["Diagnostico_Principal"].eq(diagnostico)
                ].dropna(subset=[campo_x, campo_y])
                figura.add_trace(go.Scatter(
                    x=parte[campo_x], y=parte[campo_y], mode="markers",
                    name=diagnostico.replace("Posible ", ""),
                    marker=dict(size=8, opacity=0.68, color=color),
                    customdata=np.column_stack([
                        parte["Pozo"].astype(str),
                        parte["CartaId"].astype(str),
                    ]),
                    hovertemplate=(
                        "Pozo: %{customdata[0]}<br>Carta: %{customdata[1]}<br>"
                        + titulo_x + ": %{x:.2f}<br>"
                        + titulo_y + ": %{y:.1f}<extra></extra>"
                    ),
                ))
        if campo_y == "Sumergencia_SAM_m":
            figura.add_hline(y=0, line_dash="dot", line_color="#6b7280")
        figura.update_layout(
            title=titulo, xaxis_title=titulo_x, yaxis_title=titulo_y,
            height=430, margin=dict(l=20, r=20, t=55, b=65),
            template="plotly_white",
            legend=dict(orientation="h", y=-0.22),
        )
        st.plotly_chart(figura, use_container_width=True, key=key)

    st.subheader("Relación con parámetros físicos y operativos")
    columnas = st.columns(2)
    with columnas[0]:
        mostrar_relacion(
            "Diametro_Piston_pulg", "Diámetro del pistón [pulg]",
            "Sumergencia vs diámetro de pistón", "sumergencia_vs_diametro"
        )
    with columnas[1]:
        mostrar_relacion(
            "Profundidad_Bomba_m", "Profundidad de bomba [m]",
            "Sumergencia vs profundidad de bomba",
            "sumergencia_vs_profundidad"
        )
    columnas = st.columns(2)
    with columnas[0]:
        mostrar_relacion(
            "Diametro_Piston_pulg", "Diámetro del pistón [pulg]",
            "Diámetro de pistón vs profundidad de bomba",
            "diametro_vs_profundidad",
            campo_y="Profundidad_Bomba_m",
            titulo_y="Profundidad de bomba [m]",
            color_por_sumergencia=True,
        )
    with columnas[1]:
        mostrar_relacion(
            "GPM_Analisis", "GPM", "Sumergencia vs GPM",
            "sumergencia_vs_gpm"
        )

    st.subheader("Validación del desplazamiento de bomba")
    st.caption(
        "La carrera geométrica propia se estima como la distancia entre los "
        "dos cruces de la carta real con la horizontal superior oculta. La "
        "carrera efectiva se calcula multiplicando esa carrera geométrica "
        "por el llenado operativo. El desplazamiento efectivo propio usa "
        "esa carrera efectiva, el diámetro del pistón y los GPM. La carrera "
        "efectiva y el desplazamiento informados por la API se usan solamente "
        "como referencia de comparación."
    )
    columnas_desplazamiento = [
        (
            "Desplazamiento efectivo",
            "Desplazamiento_Bruto_Efectivo_API_m3_d",
            "Desplazamiento_Bruto_Efectivo_Calculado_m3_d",
        ),
    ]
    resumen_desplazamiento = []
    for nombre, campo_api, campo_calculado in columnas_desplazamiento:
        # ``DataFrame.get`` devuelve ``None`` cuando una versión anterior
        # del pipeline no generó la columna. Convertir ese ``None`` produce
        # un escalar y luego falla al usar ``.notna()``. Conservamos siempre
        # el mismo índice para que una columna ausente sea simplemente una
        # serie completa de NaN y la comparación muestre cobertura cero.
        serie_vacia = pd.Series(
            np.nan,
            index=cartas_filtradas.index,
            dtype=float,
        )
        valor_api = pd.to_numeric(
            cartas_filtradas.get(campo_api, serie_vacia),
            errors="coerce",
        )
        valor_calculado = pd.to_numeric(
            cartas_filtradas.get(campo_calculado, serie_vacia),
            errors="coerce",
        )
        mascara = valor_api.notna() & valor_calculado.notna()
        delta = valor_calculado.loc[mascara] - valor_api.loc[mascara]
        resumen_desplazamiento.append({
            "Variable": nombre,
            "Cartas": int(mascara.sum()),
            "Sesgo calculado−API [m³/d]": delta.mean(),
            "MAE [m³/d]": delta.abs().mean(),
            "Correlación": (
                pd.concat(
                    [valor_api.loc[mascara], valor_calculado.loc[mascara]],
                    axis=1,
                ).corr().iloc[0, 1]
                if mascara.sum() >= 2
                else np.nan
            ),
        })
    st.dataframe(
        pd.DataFrame(resumen_desplazamiento).round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Pozos por diagnóstico robusto")
    conteo_robustos = (
        resumen_robusto_total.loc[
            resumen_robusto_total["Pozo"].isin(pozos_filtrados),
            ["Pozo", "Diagnosticos_Robustos_Lista"],
        ]
        .explode("Diagnosticos_Robustos_Lista")
        .dropna(subset=["Diagnosticos_Robustos_Lista"])
        .groupby("Diagnosticos_Robustos_Lista")["Pozo"]
        .nunique()
        .sort_values(ascending=False)
        .rename("Cantidad_de_pozos")
        .reset_index()
        .rename(columns={"Diagnosticos_Robustos_Lista": "Diagnóstico"})
    )
    fig_robustos = go.Figure(go.Bar(
        x=conteo_robustos["Cantidad_de_pozos"],
        y=conteo_robustos["Diagnóstico"],
        orientation="h",
        marker_color="#0874d1",
        text=conteo_robustos["Cantidad_de_pozos"],
        textposition="outside",
    ))
    fig_robustos.update_layout(
        height=max(320, 38 * len(conteo_robustos)),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Pozos únicos con diagnóstico robusto",
        margin=dict(l=20, r=30, t=20, b=35),
        template="plotly_white",
    )
    st.plotly_chart(
        fig_robustos,
        use_container_width=True,
        key="resumen_diagnosticos_robustos",
    )

    st.subheader("Pozos por diagnóstico individual de carta")
    conteo_diagnosticos = (
        cartas_filtradas[["Pozo", "Diagnosticos_Todos"]]
        .explode("Diagnosticos_Todos")
        .drop_duplicates(["Pozo", "Diagnosticos_Todos"])
        .groupby("Diagnosticos_Todos")["Pozo"]
        .nunique()
        .sort_values(ascending=False)
        .rename("Cantidad_de_pozos")
        .reset_index()
        .rename(columns={"Diagnosticos_Todos": "Diagnóstico"})
    )
    fig_conteo = go.Figure(go.Bar(
        x=conteo_diagnosticos["Cantidad_de_pozos"],
        y=conteo_diagnosticos["Diagnóstico"],
        orientation="h",
        marker_color="#0874d1",
        text=conteo_diagnosticos["Cantidad_de_pozos"],
        textposition="outside",
    ))
    fig_conteo.update_layout(
        height=max(360, 38 * len(conteo_diagnosticos)),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Pozos únicos",
        margin=dict(l=20, r=30, t=20, b=35),
        template="plotly_white",
    )
    st.plotly_chart(
        fig_conteo,
        use_container_width=True,
        key="resumen_diagnosticos_individuales",
    )

    st.subheader("Producción de los pozos filtrados")
    if ultimas.empty:
        st.info("No hay pozos para resumir.")
    else:
        suma_vfm = ultimas["VFM_Bruta_m3_d"].sum(min_count=1)
        promedio_vfm = ultimas["VFM_Bruta_m3_d"].mean()
        suma_neta = ultimas["VFM_Petroleo_m3_d"].sum(min_count=1)
        promedio_neto = ultimas["VFM_Petroleo_m3_d"].mean()
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("VFM bruto total", valor_texto(suma_vfm, ".1f", " m³/d"))
        p2.metric("VFM bruto promedio/pozo", valor_texto(promedio_vfm, ".2f", " m³/d"))
        p3.metric("VFM petróleo total", valor_texto(suma_neta, ".1f", " m³/d"))
        p4.metric("VFM petróleo promedio/pozo", valor_texto(promedio_neto, ".2f", " m³/d"))

    if not resumen_robusto.empty:
        st.subheader("Evaluación preliminar de repetición")
        st.dataframe(
            resumen_robusto,
            use_container_width=True,
            hide_index=True,
        )

with tab_explorador:
    st.caption(
        "Se muestran las cartas que cumplen el filtro individual, "
        "ordenadas por pozo y fecha. En Detalle se conserva el "
        "historial completo de cada pozo resultante."
    )
    cartas_por_pagina = st.selectbox(
        "Cartas por página",
        [10, 20, 40],
        index=1,
        key="explorador_tamano",
    )
    paginas = max(1, ceil(len(cartas_filtradas) / cartas_por_pagina))
    pagina = st.number_input(
        "Página",
        min_value=1,
        max_value=paginas,
        value=1,
        step=1,
        key="explorador_pagina",
    )
    inicio = (pagina - 1) * cartas_por_pagina
    lote = cartas_filtradas.iloc[inicio: inicio + cartas_por_pagina]

    for pozo, grupo in lote.groupby("Pozo", sort=False):
        st.markdown(f"### {pozo}")
        for i in range(0, len(grupo), 2):
            columnas = st.columns(2)
            for columna, (_, diag) in zip(columnas, grupo.iloc[i:i + 2].iterrows()):
                carta_id = int(diag["CartaId"])
                with columna:
                    st.plotly_chart(
                        figura_carta(
                            cartas_por_id[carta_id],
                            resultados_por_id.get(carta_id),
                            diag,
                            mostrar_horizontales_peso=(
                                mostrar_horizontales_peso
                            ),
                            mostrar_carta_patrones=(
                                mostrar_carta_patrones
                            ),
                            mostrar_sam_v2=mostrar_sam_v2,
                        ),
                        use_container_width=True,
                        key=f"explorador_{carta_id}_{pagina}",
                    )
                    alertas = " · ".join(diag["Diagnosticos_Todos"])
                    st.caption(f"Diagnóstico: {alertas}")
                    st.caption(
                        "Llenado "
                        f"{valor_texto(diag.get('Llenado_Diagnostico_pct'), '.1f', '%')} · "
                        "Sumergencia propia "
                        f"{valor_texto(diag.get('Sumergencia_SAM_Seleccionada_m'), '.1f', ' m')} "
                        "("
                        f"{valor_texto(diag.get('Sumergencia_Relativa_SAM_Seleccionada_pct'), '.1f', '%')}"
                        ") · "
                        "Carrera fondo "
                        f"{valor_texto(carrera_fondo_carta(cartas_por_id[carta_id]), '.1f', ' pulg')} · "
                        "VFM bruto "
                        f"{valor_texto(diag.get('VFM_Bruta_m3_d'), '.2f', ' m³/d')} · "
                        "VFM petróleo "
                        f"{valor_texto(diag.get('VFM_Petroleo_m3_d'), '.2f', ' m³/d')}"
                    )

with tab_detalle:
    if not pozos_filtrados:
        st.warning("No hay pozos que cumplan los filtros.")
    else:
        pozo = st.selectbox(
            "Seleccionar pozo",
            pozos_filtrados,
            key="detalle_pozo",
        )
        cartas_pozo = (
            historico.loc[historico["Pozo"] == pozo]
            .sort_values("Fecha", ascending=False)
            .reset_index(drop=True)
        )
        tendencias_pozo = (
            tendencias.loc[
                tendencias["Pozo_Clave"]
                == normalizar_pozo(pozo)
            ]
            .sort_values("Fecha")
            .copy()
            if not tendencias.empty
            else pd.DataFrame()
        )
        indicadores_15d = calcular_indicadores_moviles_15d(
            tendencias_pozo
        )
        paginas_pozo = max(
            1,
            ceil(len(cartas_pozo) / VENTANA_DIAGNOSTICO_ROBUSTO),
        )
        pagina_pozo = st.number_input(
            "Grupo de seis cartas (1 = más recientes)",
            min_value=1,
            max_value=paginas_pozo,
            value=1,
            step=1,
            key=f"detalle_pagina_{pozo}",
        )
        seis = cartas_pozo.iloc[
            (pagina_pozo - 1) * VENTANA_DIAGNOSTICO_ROBUSTO:
            pagina_pozo * VENTANA_DIAGNOSTICO_ROBUSTO
        ].copy()

        ventana_robusta = cartas_pozo.head(VENTANA_DIAGNOSTICO_ROBUSTO)
        estado, texto_estado, variable = diagnostico_consolidado(
            ventana_robusta
        )
        robustos_pozo = diagnosticos_robustos_ventana(ventana_robusta)
        accion_robusta = accion_para_diagnosticos_robustos(robustos_pozo)
        resumen_sumergencia = resumen_sumergencia_pozo(cartas_pozo)
        if (
            np.isfinite(resumen_sumergencia["representativa"])
            and resumen_sumergencia["representativa"] < 0
        ):
            accion_robusta = (
                "Validar datos cargados, diámetro/área de pistón y "
                "horizontales SAM antes de aplicar la acción del diagnóstico. "
                f"Luego: {accion_robusta.lower()}"
            )
        sumergencia_representativa_texto = valor_texto(
            resumen_sumergencia["representativa"], ".1f", " m"
        )
        sumergencia_representativa_pct_texto = valor_texto(
            resumen_sumergencia["representativa_pct"], ".1f", "%"
        )
        color_estado = "#16833b" if estado == "Diagnóstico robusto" else "#e87918"
        bloque_resumen_pozo = " ".join(
            linea.strip() for linea in dedent(f"""
            <div style="border-left:6px solid {color_estado};padding:10px 14px;
                        background:rgba(128,128,128,.08);border-radius:6px;">
                <div style="display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;">
                    <div style="flex:1;min-width:280px;">
                        <b>{estado}</b><br>{texto_estado}
                        {"<br><b>Atención:</b> existe variación relevante entre cartas o VFM." if variable else ""}
                    </div>
                    <div style="flex:1;min-width:280px;">
                        <b>Sumergencia representativa</b><br>
                        <span style="font-size:1.18rem;">{sumergencia_representativa_texto}
                        ({sumergencia_representativa_pct_texto})</span>
                        <small> · promedio de {resumen_sumergencia['muestras_representativa']}
                        cartas válidas dentro de las últimas seis</small><br>
                        <span style="color:{resumen_sumergencia['color']};">
                            <b>{resumen_sumergencia['estado']}</b>
                        </span> · {resumen_sumergencia['detalle']}<br>
                        <small>{resumen_sumergencia['advertencia']}</small>
                    </div>
                </div>
                <div style="margin-top:10px;padding-top:8px;
                            border-top:1px solid rgba(128,128,128,.25);">
                    <b>Acción recomendada:</b> {accion_robusta}
                </div>
            </div>
            """).splitlines() if linea.strip()
        )
        st.markdown(
            bloque_resumen_pozo,
            unsafe_allow_html=True,
        )

        if (
            estado == "Diagnóstico robusto"
            and "Posible pozo subexplotado" in texto_estado
        ):
            analisis_temporal = analizar_subexplotacion_temporal(
                indicadores_15d
            )
            evidencias_html = "".join(
                f"<li>{evidencia}</li>"
                for evidencia in analisis_temporal["evidencias"]
            )
            st.markdown(
                f"""
                <div style="border-left:6px solid {analisis_temporal['color']};
                            padding:10px 14px;margin-top:10px;
                            background:rgba(128,128,128,.08);border-radius:6px;">
                    <b>Análisis temporal de tendencias</b><br>
                    <span style="font-size:1.08rem;">
                        Posible pozo subexplotado — {analisis_temporal['estado'].lower()}
                    </span><br>
                    <small>Confianza temporal: {analisis_temporal['confianza']}</small>
                    <ul style="margin-top:8px;margin-bottom:0;">
                        {evidencias_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif estado == "Diagnóstico robusto":
            diagnosticos_aporte = [
                diagnostico
                for diagnostico in [
                    "Posible golpe de fluido",
                    "Posible compresión/interferencia de gas",
                ]
                if diagnostico in texto_estado
            ]
            if diagnosticos_aporte:
                etiqueta_aporte = " / ".join(diagnosticos_aporte)
                analisis_temporal = analizar_falta_aporte_temporal(
                    indicadores_15d,
                    diagnostico_robusto=etiqueta_aporte,
                )
                evidencias_html = "".join(
                    f"<li>{evidencia}</li>"
                    for evidencia in analisis_temporal["evidencias"]
                )
                st.markdown(
                    f"""
                    <div style="border-left:6px solid {analisis_temporal['color']};
                                padding:10px 14px;margin-top:10px;
                                background:rgba(128,128,128,.08);border-radius:6px;">
                        <b>Análisis temporal de admisión</b><br>
                        <span style="font-size:1.08rem;">
                            {etiqueta_aporte} — {analisis_temporal['estado'].lower()}
                        </span><br>
                        <small>Confianza temporal: {analisis_temporal['confianza']}</small>
                        <ul style="margin-top:8px;margin-bottom:0;">
                            {evidencias_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif "Posible sin trabajo de bomba" in texto_estado:
                analisis_temporal = analizar_bloqueo_temporal(
                    tendencias_pozo
                )
                evidencias_html = "".join(
                    f"<li>{evidencia}</li>"
                    for evidencia in analisis_temporal["evidencias"]
                )
                st.markdown(
                    f"""
                    <div style="border-left:6px solid {analisis_temporal['color']};
                                padding:10px 14px;margin-top:10px;
                                background:rgba(128,128,128,.08);border-radius:6px;">
                        <b>Análisis temporal de trabajo de bomba</b><br>
                        <span style="font-size:1.08rem;">
                            Posible sin trabajo de bomba —
                            {analisis_temporal['estado'].lower()}
                        </span><br>
                        <small>Confianza temporal:
                            {analisis_temporal['confianza']}</small>
                        <ul style="margin-top:8px;margin-bottom:0;">
                            {evidencias_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        ultima = cartas_pozo.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Régimen de bombeo", valor_texto(ultima.get("GPM"), ".2f", " GPM"))
        m2.metric(
            "Carrera de superficie (API)",
            valor_texto(carrera_superficie(ultima), ".1f", " pulg"),
        )
        m3.metric("Profundidad de bomba", valor_texto(ultima.get("ProfundidadBomba"), ".0f", " m"))
        m4.metric("Diámetro de pistón", valor_texto(ultima.get("DiametroPistonBomba"), ".2f", " in"))

        st.subheader("Evolución VFM y controles reales")
        vfm_pozo = (
            vfm_historico.loc[vfm_historico["Pozo"] == pozo]
            .sort_values("Fecha_Referencia_JSON")
            .drop_duplicates("Fecha_Referencia_JSON", keep="last")
        )
        controles_pozo = controles.loc[
            controles["Pozo_Clave"] == normalizar_pozo(pozo)
        ].sort_values("Fecha_Control") if not controles.empty else pd.DataFrame()

        fig = go.Figure()
        if not vfm_pozo.empty:
            fig.add_trace(go.Scatter(
                x=vfm_pozo["Fecha_Referencia_JSON"],
                y=vfm_pozo["VFM_Bruta_m3_d"],
                mode="lines+markers",
                name="VFM bruto",
                line=dict(color="#0874d1", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=vfm_pozo["Fecha_Referencia_JSON"],
                y=vfm_pozo["VFM_Petroleo_m3_d"],
                mode="lines+markers",
                name="VFM petróleo",
                line=dict(color="#16833b", width=3),
            ))
        if not controles_pozo.empty:
            fig.add_trace(go.Scatter(
                x=controles_pozo["Fecha_Control"],
                y=controles_pozo["Control_Bruta_m3_d"],
                mode="markers",
                name="Control bruto",
                marker=dict(color="#78bff2", size=12, symbol="diamond"),
            ))
            fig.add_trace(go.Scatter(
                x=controles_pozo["Fecha_Control"],
                y=controles_pozo["Control_Petroleo_m3_d"],
                mode="markers",
                name="Control petróleo",
                marker=dict(color="#7acb91", size=12, symbol="diamond"),
            ))
        fig.update_layout(
            height=430,
            hovermode="x unified",
            xaxis_title="Fecha",
            yaxis_title="Caudal [m³/d]",
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
            margin=dict(l=40, r=30, t=55, b=40),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"evolucion_{pozo}")

        if not controles_pozo.empty:
            st.caption(
                f"{len(controles_pozo)} control(es) real(es) disponible(s) para este pozo."
            )

        st.subheader("Tendencias históricas de operación")

        if tendencias_pozo.empty:
            st.info(
                "Cargá el CSV histórico de tendencias desde la barra "
                "lateral para visualizar estas variables."
            )
        else:
            fecha_minima = tendencias_pozo["Fecha"].min()
            fecha_maxima = tendencias_pozo["Fecha"].max()
            st.caption(
                f"{len(tendencias_pozo)} mediciones entre "
                f"{fecha_minima:%d/%m/%Y %H:%M} y "
                f"{fecha_maxima:%d/%m/%Y %H:%M}. "
                "Los gráficos muestran todas las mediciones recibidas."
            )

            st.markdown("#### Indicadores móviles de los últimos 15 días")
            if indicadores_15d.empty:
                st.info(
                    "No hay suficientes variables numéricas para calcular "
                    "los indicadores móviles."
                )
            else:
                st.dataframe(
                    indicadores_15d.round(2),
                    use_container_width=True,
                    hide_index=True,
                )
                with st.expander(
                    "Cómo interpretar estos indicadores",
                    expanded=False,
                ):
                    st.markdown(
                        """
- **Mediana 15d:** nivel central robusto de las medianas diarias.
- **Pendiente por día:** tendencia robusta Theil–Sen en unidades por día.
- **Pendiente relativa:** cambio diario respecto de la mediana de la ventana.
- **Volatilidad MAD:** dispersión robusta, menos sensible a valores extremos.
- **Cambio 3d vs 12d:** mediana de los últimos tres días menos la de los días anteriores.
- **Calidad:** depende de la cantidad de días distintos con información.

Los cálculos usan medianas diarias para que un día con muchas mediciones
no tenga más peso que otro. Todavía no se aplican umbrales diagnósticos.
                        """
                    )

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    figura_tendencia_diaria(
                        tendencias_pozo,
                        [
                            (
                                "Llenado_Bomba_API_pct",
                                "Llenado de bomba",
                                "#16833b",
                            ),
                            (
                                "Sumergencia_Relativa_API_pct",
                                "Sumergencia relativa",
                                "#0284c7",
                            ),
                        ],
                        "Llenado y sumergencia relativa",
                        "%",
                    ),
                    use_container_width=True,
                    key=f"tendencia_llenado_{pozo}",
                )
            with g2:
                st.plotly_chart(
                    figura_tendencia_diaria(
                        tendencias_pozo,
                        [
                            (
                                "Peso_Fluido_Promedio_lbf",
                                "Peso promedio",
                                "#2563eb",
                            ),
                            (
                                "Peso_Fluido_Max_lbf",
                                "Peso máximo",
                                "#7c3aed",
                            ),
                        ],
                        "Peso de fluido",
                        "lbf",
                    ),
                    use_container_width=True,
                    key=f"tendencia_peso_{pozo}",
                )

            g3, g4 = st.columns(2)
            with g3:
                st.plotly_chart(
                    figura_tendencia_diaria(
                        tendencias_pozo,
                        [
                            (
                                "Carga_Maxima_Fondo_lbf",
                                "Máxima fondo",
                                "#dc2626",
                            ),
                            (
                                "Carga_Minima_Fondo_lbf",
                                "Mínima fondo",
                                "#f59e0b",
                            ),
                        ],
                        "Cargas de fondo",
                        "lbf",
                    ),
                    use_container_width=True,
                    key=f"tendencia_cargas_fondo_{pozo}",
                )
            with g4:
                st.plotly_chart(
                    figura_tendencia_diaria(
                        tendencias_pozo,
                        [
                            (
                                "Carrera_Fondo_Total_pulg",
                                "Fondo total",
                                "#0f766e",
                            ),
                            (
                                "Carrera_Fondo_Efectiva_pulg",
                                "Fondo efectiva",
                                "#14b8a6",
                            ),
                            (
                                "Carrera_Superficie_pulg",
                                "Superficie",
                                "#0284c7",
                            ),
                        ],
                        "Carreras",
                        "pulg",
                    ),
                    use_container_width=True,
                    key=f"tendencia_carreras_{pozo}",
                )

            g5, g6 = st.columns(2)
            with g5:
                st.plotly_chart(
                    figura_tendencia_diaria(
                        tendencias_pozo,
                        [
                            (
                                "Torque_Reductor_pct",
                                "Torque reductor",
                                "#e11d48",
                            ),
                            (
                                "Carga_Estructural_pct",
                                "Carga estructural",
                                "#9333ea",
                            ),
                        ],
                        "Condición de superficie",
                        "%",
                    ),
                    use_container_width=True,
                    key=f"tendencia_superficie_pct_{pozo}",
                )
            with g6:
                st.plotly_chart(
                    figura_tendencia_diaria(
                        tendencias_pozo,
                        [
                            (
                                "Carga_Maxima_Superficie_lbf",
                                "Máxima superficie",
                                "#b91c1c",
                            ),
                            (
                                "Carga_Minima_Superficie_lbf",
                                "Mínima superficie",
                                "#ea580c",
                            ),
                        ],
                        "Cargas de superficie",
                        "lbf",
                    ),
                    use_container_width=True,
                    key=f"tendencia_cargas_superficie_{pozo}",
                )

        st.subheader(
            f"Cartas {1 + (pagina_pozo - 1) * VENTANA_DIAGNOSTICO_ROBUSTO}–"
            f"{min(pagina_pozo * VENTANA_DIAGNOSTICO_ROBUSTO, len(cartas_pozo))} "
            f"de {len(cartas_pozo)}"
        )
        for i in range(0, len(seis), 2):
            columnas = st.columns(2)
            for columna, (_, diag) in zip(columnas, seis.iloc[i:i + 2].iterrows()):
                carta_id = int(diag["CartaId"])
                with columna:
                    st.plotly_chart(
                        figura_carta(
                            cartas_por_id[carta_id],
                            resultados_por_id.get(carta_id),
                            diag,
                            mostrar_horizontales_peso=(
                                mostrar_horizontales_peso
                            ),
                            mostrar_carta_patrones=(
                                mostrar_carta_patrones
                            ),
                            mostrar_sam_v2=mostrar_sam_v2,
                        ),
                        use_container_width=True,
                        key=f"detalle_{pozo}_{carta_id}_{pagina_pozo}",
                    )
                    st.markdown(
                        f"**Diagnósticos:** {' · '.join(diag['Diagnosticos_Todos'])}"
                    )
                    st.caption(
                        f"Fecha: {pd.to_datetime(diag['Fecha']).strftime('%d/%m/%Y %H:%M')}  \n"
                        f"Carrera de fondo efectiva: "
                        f"{valor_texto(diag.get('Carrera_Efectiva_Fondo_Calculada_pulg'), '.1f', ' pulg')}  \n"
                        f"Peso de fluido: "
                        f"{valor_texto(diag.get('Peso_Fluido_SAM_Seleccionado_lbf'), '.0f', ' lbf')}  \n"
                        f"Sumergencia: "
                        f"{valor_texto(diag.get('Sumergencia_SAM_Seleccionada_m'), '.1f', ' m')} "
                        f"({valor_texto(diag.get('Sumergencia_Relativa_SAM_Seleccionada_pct'), '.1f', '%')})  \n"
                        f"Desplazamiento efectivo: "
                        f"{valor_texto(diag.get('Desplazamiento_Bruto_Efectivo_Calculado_m3_d'), '.2f', ' m³/d')}  \n"
                        f"Torque: {valor_texto(diag.get('Torque_Reductor_pct'), '.1f', '%')} · "
                        f"Carga estructural: {valor_texto(diag.get('Carga_Estructural_pct'), '.1f', '%')}  \n"
                        f"VFM: bruto {valor_texto(diag.get('VFM_Bruta_m3_d'), '.2f', ' m³/d')} · "
                        f"petróleo {valor_texto(diag.get('VFM_Petroleo_m3_d'), '.2f', ' m³/d')}  \n"
                        f"Comentario VFM: {diag.get('Comentario_VFM_Control', 'Sin comparación disponible')}  \n"
                        f"Acción recomendada: {diag.get('Accion_Sugerida', '—')}"
                    )

with tab_descargas:
    st.subheader("Descargar resultados")
    st.caption(
        "Las descargas respetan los filtros actuales. El CSV de cartas "
        "respeta también el filtro individual; el resumen por pozo conserva "
        "el contexto completo de los pozos seleccionados."
    )

    exportacion_cartas = construir_exportacion_cartas(
        cartas_filtradas
    )
    csv_cartas = preparar_csv_descarga(exportacion_cartas).to_csv(
        index=False,
        sep=";",
        decimal=",",
    ).encode("utf-8-sig")

    resumen_pozos = construir_exportacion_pozos(
        cartas_contexto
    )
    csv_pozos = preparar_csv_descarga(resumen_pozos).to_csv(
        index=False,
        sep=";",
        decimal=",",
    ).encode("utf-8-sig")

    d1, d2 = st.columns(2)
    with d1:
        st.metric("Cartas incluidas", len(cartas_filtradas))
        st.download_button(
            "Descargar CSV de cartas",
            data=csv_cartas,
            file_name="diagnostico_cartas_filtradas.csv",
            mime="text/csv",
            key="descargar_csv_cartas",
            use_container_width=True,
        )
    with d2:
        st.metric("Pozos incluidos", len(resumen_pozos))
        st.download_button(
            "Descargar CSV resumen por pozo",
            data=csv_pozos,
            file_name="diagnostico_resumen_pozos.csv",
            mime="text/csv",
            key="descargar_csv_pozos",
            use_container_width=True,
        )

if errores_archivos:
    with st.sidebar.expander("Archivos con errores"):
        for error in errores_archivos:
            st.caption(error)

st.sidebar.divider()
st.sidebar.caption(
    f"{len(archivos)} JSON · {historico['Pozo'].nunique()} pozos · "
    f"{len(historico)} cartas únicas"
)
