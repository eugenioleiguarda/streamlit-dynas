from __future__ import annotations

import ast
import json
from math import ceil

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
    "2026-08-13-hombros-posicion-peso-v44"
)

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
def ejecutar_pipeline(contenido: bytes, version: str):
    _ = version
    return procesar_json(contenido)


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


def construir_tabla_cartas(salida, produccion, controles):
    muestra = salida["muestra"].copy()
    diagnosticos = salida["diagnosticos_cartas"].copy()
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
):
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
        calculo_valido = bool(
            resultado.get("Calculo_Peso_Experimental_Valido", False)
        )
        carga_superior_peso = pd.to_numeric(
            resultado.get(
                "Carga_Superior_Peso_Experimental_lbf",
                np.nan,
            ),
            errors="coerce",
        )
        carga_inferior_peso = pd.to_numeric(
            resultado.get(
                "Carga_Inferior_Peso_Experimental_lbf",
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
                name="Peso fluido - superior",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=[x_min, x_max],
                y=[carga_inferior_peso, carga_inferior_peso],
                mode="lines",
                name="Peso fluido - inferior",
                line=dict(color="#06b6d4", width=2, dash="dot"),
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
    Regla preliminar:
    - robusto: aparece en al menos 3 de las últimas 5 cartas;
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
        [(d, n) for d, n in conteos.items() if n >= 3],
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
            "No hay un diagnóstico repetido en 3 cartas y existe variación entre mediciones.",
            True,
        )
    return (
        "Sin diagnóstico robusto",
        "Ningún diagnóstico aparece en al menos 3 de las cartas analizadas.",
        False,
    )


def tabla_diagnosticos_robustos(historico):
    """
    Una fila por pozo. Se consideran las últimas cinco cartas y un
    diagnóstico es robusto cuando aparece al menos tres veces.
    """
    filas = []
    for pozo, grupo in historico.groupby("Pozo", sort=True):
        ultimas = grupo.nlargest(5, "Fecha")
        conteos = {}
        for lista in ultimas["Diagnosticos_Todos"]:
            for diagnostico in set(lista_alertas(lista)):
                conteos[diagnostico] = conteos.get(diagnostico, 0) + 1
        robustos = sorted(
            diagnostico
            for diagnostico, cantidad in conteos.items()
            if cantidad >= 3
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
        "Sumergencia_Relativa_Peso_Experimental_pct",
        "Delta_Sumergencia_Peso_Experimental_vs_API_m",
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
        ultimas_cinco = grupo.head(5)
        ultima = grupo.iloc[0]

        conteos = {}
        for diagnosticos in ultimas_cinco["Diagnosticos_Todos"]:
            for diagnostico in set(lista_alertas(diagnosticos)):
                conteos[diagnostico] = conteos.get(diagnostico, 0) + 1

        robustos = sorted(
            (
                (diagnostico, cantidad)
                for diagnostico, cantidad in conteos.items()
                if cantidad >= 3
            ),
            key=lambda item: (-item[1], item[0]),
        )
        diagnosticos_robustos = " | ".join(
            f"{diagnostico} ({cantidad}/{len(ultimas_cinco)})"
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
            "Cartas_Analizadas_Robustez": int(len(ultimas_cinco)),
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
            "Sumergencia_Relativa_Peso_Experimental_pct",
            "Delta_Sumergencia_Peso_Experimental_vs_API_m",
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
    "Mostrar horizontales de peso de fluido",
    value=False,
    help=(
        "Muestra las rectas ocultas usadas para estimar peso de fluido y "
        "sumergencia. No modifica la carta patrones de cálculo ni los "
        "diagnósticos."
    ),
)
mostrar_carta_patrones = st.sidebar.checkbox(
    "Mostrar carta patrones de cálculo",
    value=True,
    help=(
        "Muestra u oculta la referencia geométrica usada por los cálculos. "
        "Ocultarla no modifica el llenado ni los diagnósticos."
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
            salida = ejecutar_pipeline(contenido, PIPELINE_CACHE_VERSION)
            produccion = excluir_vfm_cartas_invalidas(
                salida,
                ejecutar_vfm(contenido),
            )
            tabla = construir_tabla_cartas(salida, produccion, controles)
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
        "Sólo considera diagnósticos presentes en al menos tres "
        "de las últimas cinco cartas."
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
    pozos_filtrados = sorted(
        cartas_filtradas["Pozo"].dropna().unique()
    )
else:
    cartas_filtradas = cartas_candidatas.copy()
    pozos_filtrados = pozos_candidatos

# El resumen y el detalle conservan todas las cartas de los pozos
# resultantes. El explorador muestra solamente las cartas que cumplen
# el filtro individual.
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
        cinco = grupo.nlargest(5, "Fecha")
        estado, texto, variable = diagnostico_consolidado(cinco)
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

    st.subheader("Comparación de peso de fluido — criterio manual equivalente")
    st.caption(
        "Compara directamente PesoFluido informado por la API con la "
        "separación de las horizontales ocultas del método manual equivalente. "
        "El método fue calibrado con la carta ECh-277 del 12/08/2026 15:55 "
        "para reproducir 5764 lbf. Esta comparación no usa "
        "diámetro de pistón, profundidad, densidad ni presiones y, por lo "
        "tanto, permite detectar primero diferencias de carga o unidades."
    )
    st.caption(
        "Unidades interpretadas actualmente: cargas de fondo y PesoFluido "
        "en lbf; diámetro del pistón en pulgadas; profundidad de bomba en "
        "metros. Las dos últimas no intervienen en este gráfico."
    )

    comparacion_peso = cartas_contexto.copy()
    peso_api = pd.to_numeric(
        comparacion_peso.get(
            "Peso_Fluido_API_lbf",
            pd.Series(np.nan, index=comparacion_peso.index),
        ),
        errors="coerce",
    )
    peso_horizontales = pd.to_numeric(
        comparacion_peso.get(
            "Peso_Fluido_Experimental_lbf",
            pd.Series(np.nan, index=comparacion_peso.index),
        ),
        errors="coerce",
    )
    mascara_peso = (
        peso_api.notna()
        & peso_horizontales.notna()
        & (peso_api > 0)
        & (peso_horizontales > 0)
    )
    pesos_comparables = comparacion_peso.loc[mascara_peso].copy()
    pesos_comparables["Peso_API_lbf"] = peso_api.loc[mascara_peso]
    pesos_comparables["Peso_Horizontales_lbf"] = (
        peso_horizontales.loc[mascara_peso]
    )
    pesos_comparables["Delta_Peso_lbf"] = (
        pesos_comparables["Peso_Horizontales_lbf"]
        - pesos_comparables["Peso_API_lbf"]
    )
    pesos_comparables["Ratio_Horizontales_API"] = (
        pesos_comparables["Peso_Horizontales_lbf"]
        / pesos_comparables["Peso_API_lbf"]
    )

    total_peso = len(comparacion_peso)
    n_peso = len(pesos_comparables)
    cobertura_peso = 100.0 * n_peso / total_peso if total_peso else np.nan
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.caption("Cartas comparables")
    p1.markdown(f"### {n_peso} / {total_peso}")
    p2.caption("Cobertura")
    p2.markdown(f"### {valor_texto(cobertura_peso, '.1f', '%')}")

    if pesos_comparables.empty:
        for columna, titulo in zip(
            (p3, p4, p5, p6),
            ("Sesgo horiz.−API", "Error absoluto medio", "Ratio mediano", "Correlación"),
        ):
            columna.caption(titulo)
            columna.markdown("### —")
        st.info(
            "No hay cartas con ambos pesos positivos disponibles para el filtro."
        )
    else:
        sesgo_peso = pesos_comparables["Delta_Peso_lbf"].mean()
        mae_peso = pesos_comparables["Delta_Peso_lbf"].abs().mean()
        ratio_mediano = pesos_comparables["Ratio_Horizontales_API"].median()
        correlacion_peso = pesos_comparables[
            ["Peso_API_lbf", "Peso_Horizontales_lbf"]
        ].corr().iloc[0, 1]

        p3.caption("Sesgo método−API")
        p3.markdown(f"### {valor_texto(sesgo_peso, '.0f', ' lbf')}")
        p4.caption("Error absoluto medio")
        p4.markdown(f"### {valor_texto(mae_peso, '.0f', ' lbf')}")
        p5.caption("Ratio mediano")
        p5.markdown(f"### {valor_texto(ratio_mediano, '.3f')}")
        p6.caption("Correlación")
        p6.markdown(f"### {valor_texto(correlacion_peso, '.3f')}")

        minimo_peso = float(np.nanmin([
            pesos_comparables["Peso_API_lbf"].min(),
            pesos_comparables["Peso_Horizontales_lbf"].min(),
        ]))
        maximo_peso = float(np.nanmax([
            pesos_comparables["Peso_API_lbf"].max(),
            pesos_comparables["Peso_Horizontales_lbf"].max(),
        ]))
        fig_peso = go.Figure()
        fig_peso.add_trace(go.Scatter(
            x=pesos_comparables["Peso_API_lbf"],
            y=pesos_comparables["Peso_Horizontales_lbf"],
            mode="markers",
            name="Cartas",
            marker=dict(size=8, opacity=0.65, color="#0f9d58"),
            customdata=np.column_stack([
                pesos_comparables["Pozo"].astype(str),
                pesos_comparables["CartaId"].astype(str),
                pesos_comparables["Delta_Peso_lbf"],
                pesos_comparables["Ratio_Horizontales_API"],
            ]),
            hovertemplate=(
                "Pozo: %{customdata[0]}<br>"
                "Carta: %{customdata[1]}<br>"
                "API: %{x:.0f} lbf<br>"
                "Horizontales: %{y:.0f} lbf<br>"
                "Diferencia: %{customdata[2]:+.0f} lbf<br>"
                "Ratio: %{customdata[3]:.3f}<extra></extra>"
            ),
        ))
        fig_peso.add_trace(go.Scatter(
            x=[minimo_peso, maximo_peso],
            y=[minimo_peso, maximo_peso],
            mode="lines",
            name="Igualdad",
            line=dict(color="#6b7280", dash="dash"),
            hoverinfo="skip",
        ))
        fig_peso.update_layout(
            xaxis_title="Peso de fluido API [lbf]",
            yaxis_title="Peso por criterio manual equivalente [lbf]",
            height=470,
            margin=dict(l=20, r=20, t=20, b=45),
            template="plotly_white",
        )
        st.plotly_chart(
            fig_peso,
            use_container_width=True,
            key="comparacion_general_peso_fluido",
        )

    st.subheader("Comparación general de sumergencia")
    st.caption(
        "Comparación informativa entre la sumergencia informada por la API "
        "y la calculada con las horizontales ocultas del criterio manual "
        "equivalente calibrado. Solo incluye cartas "
        "con ambos valores disponibles; no modifica los diagnósticos."
    )
    st.caption(
        "Cálculo propio con las horizontales ocultas calibradas en ECh-277 "
        "(12/08/2026 15:55), gravedad específica fija 0,9904 y presiones "
        "de tubing/casing iguales. Con 5764 lbf y profundidad 1789 m, la "
        "sumergencia resulta aproximadamente 87,8 m; con los 1791 m del "
        "JSON resulta aproximadamente 89,8 m. No se fuerza el valor API."
    )

    comparacion_sumergencia = cartas_contexto.copy()
    api_sumergencia = pd.to_numeric(
        comparacion_sumergencia.get("Sumergencia_API_m"),
        errors="coerce",
    )
    propia_sumergencia = pd.to_numeric(
        comparacion_sumergencia.get("Sumergencia_Peso_Experimental_m"),
        errors="coerce",
    )
    mascara_comparable = api_sumergencia.notna() & propia_sumergencia.notna()
    comparables = comparacion_sumergencia.loc[mascara_comparable].copy()
    comparables["Sumergencia_API_m"] = api_sumergencia.loc[mascara_comparable]
    comparables["Sumergencia_Propia_m"] = propia_sumergencia.loc[mascara_comparable]
    comparables["Delta_Propia_API_m"] = (
        comparables["Sumergencia_Propia_m"]
        - comparables["Sumergencia_API_m"]
    )

    cantidad_total = len(cartas_contexto)
    cantidad_comparable = len(comparables)
    cobertura_comparacion = (
        100.0 * cantidad_comparable / cantidad_total
        if cantidad_total
        else np.nan
    )

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.caption("Cartas comparables")
    s1.markdown(f"### {cantidad_comparable} / {cantidad_total}")
    s2.caption("Cobertura")
    s2.markdown(
        f"### {valor_texto(cobertura_comparacion, '.1f', '%')}"
    )

    if comparables.empty:
        s3.caption("Sesgo propia−API")
        s3.markdown("### —")
        s4.caption("Error absoluto medio")
        s4.markdown("### —")
        s5.caption("Correlación")
        s5.markdown("### —")
        st.info(
            "No hay cartas con sumergencia API y propia disponibles "
            "simultáneamente para este filtro."
        )
    else:
        sesgo_sumergencia = comparables["Delta_Propia_API_m"].mean()
        mae_sumergencia = comparables["Delta_Propia_API_m"].abs().mean()
        correlacion_sumergencia = comparables[
            ["Sumergencia_API_m", "Sumergencia_Propia_m"]
        ].corr().iloc[0, 1]

        s3.caption("Sesgo propia−API")
        s3.markdown(
            f"### {valor_texto(sesgo_sumergencia, '.1f', ' m')}"
        )
        s4.caption("Error absoluto medio")
        s4.markdown(
            f"### {valor_texto(mae_sumergencia, '.1f', ' m')}"
        )
        s5.caption("Correlación")
        s5.markdown(
            f"### {valor_texto(correlacion_sumergencia, '.3f')}"
        )

        limite_minimo = float(np.nanmin([
            comparables["Sumergencia_API_m"].min(),
            comparables["Sumergencia_Propia_m"].min(),
        ]))
        limite_maximo = float(np.nanmax([
            comparables["Sumergencia_API_m"].max(),
            comparables["Sumergencia_Propia_m"].max(),
        ]))

        fig_sumergencia = go.Figure()
        fig_sumergencia.add_trace(go.Scatter(
            x=comparables["Sumergencia_API_m"],
            y=comparables["Sumergencia_Propia_m"],
            mode="markers",
            name="Cartas",
            marker=dict(size=8, opacity=0.65, color="#0874d1"),
            customdata=np.column_stack([
                comparables["Pozo"].astype(str),
                comparables["CartaId"].astype(str),
                comparables["Delta_Propia_API_m"],
            ]),
            hovertemplate=(
                "Pozo: %{customdata[0]}<br>"
                "Carta: %{customdata[1]}<br>"
                "API: %{x:.1f} m<br>"
                "Propia: %{y:.1f} m<br>"
                "Diferencia: %{customdata[2]:+.1f} m"
                "<extra></extra>"
            ),
        ))
        fig_sumergencia.add_trace(go.Scatter(
            x=[limite_minimo, limite_maximo],
            y=[limite_minimo, limite_maximo],
            mode="lines",
            name="Igualdad",
            line=dict(color="#6b7280", dash="dash"),
            hoverinfo="skip",
        ))
        fig_sumergencia.update_layout(
            xaxis_title="Sumergencia API [m]",
            yaxis_title="Sumergencia propia [m]",
            height=470,
            margin=dict(l=20, r=20, t=20, b=45),
            template="plotly_white",
        )
        st.plotly_chart(
            fig_sumergencia,
            use_container_width=True,
            key="comparacion_general_sumergencia",
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
            index=cartas_contexto.index,
            dtype=float,
        )
        valor_api = pd.to_numeric(
            cartas_contexto.get(campo_api, serie_vacia),
            errors="coerce",
        )
        valor_calculado = pd.to_numeric(
            cartas_contexto.get(campo_calculado, serie_vacia),
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
                        ),
                        use_container_width=True,
                        key=f"explorador_{carta_id}_{pagina}",
                    )
                    alertas = " · ".join(diag["Diagnosticos_Todos"])
                    st.caption(alertas)
                    st.caption(
                        "Llenado "
                        f"{valor_texto(diag.get('Llenado_Operativo_pct'), '.1f', '%')} · "
                        "Sumergencia API "
                        f"{valor_texto(diag.get('Sumergencia_Relativa_pct'), '.1f', '%')} · "
                        "propia "
                        f"{valor_texto(diag.get('Sumergencia_Relativa_Peso_Experimental_pct'), '.1f', '%')} · "
                        "Δ propia−API "
                        f"{valor_texto(diag.get('Delta_Sumergencia_Peso_Experimental_vs_API_m'), '+.1f', ' m')} · "
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
        paginas_pozo = max(1, ceil(len(cartas_pozo) / 5))
        pagina_pozo = st.number_input(
            "Grupo de cinco cartas (1 = más recientes)",
            min_value=1,
            max_value=paginas_pozo,
            value=1,
            step=1,
            key=f"detalle_pagina_{pozo}",
        )
        cinco = cartas_pozo.iloc[
            (pagina_pozo - 1) * 5: pagina_pozo * 5
        ].copy()

        estado, texto_estado, variable = diagnostico_consolidado(cinco)
        color_estado = "#16833b" if estado == "Diagnóstico robusto" else "#e87918"
        st.markdown(
            f"""
            <div style="border-left:6px solid {color_estado};padding:10px 14px;
                        background:rgba(128,128,128,.08);border-radius:6px;">
                <b>{estado}</b><br>{texto_estado}
                {"<br><b>Atención:</b> existe variación relevante entre cartas o VFM." if variable else ""}
            </div>
            """,
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
            f"Cartas {1 + (pagina_pozo - 1) * 5}–"
            f"{min(pagina_pozo * 5, len(cartas_pozo))} de {len(cartas_pozo)}"
        )
        for i in range(0, len(cinco), 2):
            columnas = st.columns(2)
            for columna, (_, diag) in zip(columnas, cinco.iloc[i:i + 2].iterrows()):
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
                        ),
                        use_container_width=True,
                        key=f"detalle_{pozo}_{carta_id}_{pagina_pozo}",
                    )
                    st.markdown(
                        f"**Diagnósticos:** {' · '.join(diag['Diagnosticos_Todos'])}"
                    )
                    st.caption(
                        f"Fecha: {pd.to_datetime(diag['Fecha']).strftime('%d/%m/%Y %H:%M')}  \n"
                        f"Carrera de fondo: "
                        f"{valor_texto(carrera_fondo_carta(cartas_por_id[carta_id]), '.1f', ' pulg')}  \n"
                        f"Llenado operativo: {valor_texto(diag.get('Llenado_Operativo_pct'), '.1f', '%')} · "
                        f"Sumergencia API: {valor_texto(diag.get('Sumergencia_API_m'), '.1f', ' m')} "
                        f"({valor_texto(diag.get('Sumergencia_Relativa_pct'), '.1f', '%')}) · "
                        f"calculada: "
                        f"{valor_texto(diag.get('Sumergencia_Peso_Experimental_m'), '.1f', ' m')} "
                        f"({valor_texto(diag.get('Sumergencia_Relativa_Peso_Experimental_pct'), '.1f', '%')})  \n"
                        f"Peso fluido API: "
                        f"{valor_texto(diag.get('Peso_Fluido_API_lbf'), '.0f', ' lbf')} · "
                        f"manual equivalente: "
                        f"{valor_texto(diag.get('Peso_Fluido_Experimental_lbf'), '.0f', ' lbf')}  \n"
                        f"SG usada: {valor_texto(diag.get('SG_Fluido_Peso_Experimental'), '.4f')} · "
                        f"carga hidráulica corregida: "
                        f"{valor_texto(diag.get('Carga_Hidraulica_Efectiva_Peso_Experimental_lbf'), '.0f', ' lbf')}  \n"
                        f"Carrera geométrica propia: "
                        f"{valor_texto(diag.get('Carrera_Geometrica_Fondo_Calculada_pulg'), '.1f', ' pulg')}  \n"
                        f"Carrera efectiva propia/API: "
                        f"{valor_texto(diag.get('Carrera_Efectiva_Fondo_Calculada_pulg'), '.1f', ' pulg')} / "
                        f"{valor_texto(diag.get('Carrera_Efectiva_Fondo_API_pulg'), '.1f', ' pulg')}  \n"
                        f"Desplazamiento efectivo propio/API: "
                        f"{valor_texto(diag.get('Desplazamiento_Bruto_Efectivo_Calculado_m3_d'), '.2f', ' m³/d')} / "
                        f"{valor_texto(diag.get('Desplazamiento_Bruto_Efectivo_API_m3_d'), '.2f', ' m³/d')}  \n"
                        f"Torque: {valor_texto(diag.get('Torque_Reductor_pct'), '.1f', '%')} · "
                        f"Carga estructural: {valor_texto(diag.get('Carga_Estructural_pct'), '.1f', '%')}  \n"
                        f"Acción: {diag.get('Accion_Sugerida', '—')}  \n"
                        f"VFM/control: {diag.get('Comentario_VFM_Control', 'Sin comparación disponible')}"
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
