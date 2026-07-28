from __future__ import annotations

import ast
import importlib
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

# Streamlit puede conservar módulos importados entre reruns. La recarga
# explícita garantiza que un cambio subido a pipeline_diagnostico.py se use
# inmediatamente y no quede mezclado con una versión anterior en memoria.
pipeline_diagnostico_modulo = importlib.reload(
    pipeline_diagnostico_modulo
)
a_array = pipeline_diagnostico_modulo.a_array
procesar_json = pipeline_diagnostico_modulo.procesar_json


st.set_page_config(
    page_title="Diagnóstico de pozos",
    page_icon="📈",
    layout="wide",
)

PIPELINE_CACHE_VERSION = "2026-07-28-sin-trabajo-sin-llenado-v10"

COLORES = {
    "Posible pozo subexplotado": "#16833b",
    "Posible sin trabajo de bomba": "#f59e0b",
    "Carta no válida - posible falla de medición o transmisión": "#b91c1c",
    "Posible golpe de fluido": "#e87918",
    "Posible compresión/interferencia de gas": "#2563eb",
    "Posible compresión/interferencia de gas suave": "#4f7fe5",
    "Posible pérdida en válvula viajera": "#8b5cf6",
    "Posible golpe de bomba": "#e11d48",
    "Posible tubing libre": "#795548",
    "Posible cierre tardío de válvula viajera": "#a855f7",
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


def figura_carta(carta, resultado, diagnostico):
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
    mostrar_carta_ideal = not sin_trabajo and not carta_no_valida
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
                name="Carta ideal",
                line=dict(color="#9063cd", width=3, dash="dash"),
            ))
    except Exception:
        pass
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
        margin=dict(l=35, r=20, t=72, b=35),
        xaxis_title="Posición",
        yaxis_title="Carga",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
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

    exclusiones = {"Pozo bien explotado"}
    conteos = {}
    for lista in cartas_ultimas["Diagnosticos_Todos"]:
        for diagnostico in set(lista):
            if diagnostico not in exclusiones:
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
            "No hay una anomalía repetida en 3 cartas y existe variación entre mediciones.",
            True,
        )
    return (
        "Sin diagnóstico robusto",
        "Ninguna anomalía aparece en al menos 3 de las cartas analizadas.",
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
                if diagnostico != "Pozo bien explotado":
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
    "Posible compresión/interferencia de gas suave":
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
        "Compresion_Gas_Suave",
        "Golpe_Bomba",
        "Tubing_Libre",
        "Pozo_Subexplotado",
        "Exceso_Torque",
        "Exceso_Carga_Estructural",
        "Llenado_Bruto_pct",
        "Llenado_Original_pct",
        "Llenado_Operativo_pct",
        "Sumergencia_Relativa_pct",
        "Torque_Reductor_pct",
        "Carga_Estructural_pct",
        "GPM",
        "Carrera_Fondo_Carta_pulg",
        "Carrera_Superficie_API_pulg",
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
                if diagnostico != "Pozo bien explotado":
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
archivo_controles = st.sidebar.file_uploader(
    "Actualizar controles reales (opcional)",
    type=["xlsx"],
    help="Si no se carga un Excel se utiliza controles_reales.xlsx del repositorio.",
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
                        ),
                        use_container_width=True,
                        key=f"explorador_{carta_id}_{pagina}",
                    )
                    alertas = " · ".join(diag["Diagnosticos_Todos"])
                    st.caption(alertas)
                    st.caption(
                        "Llenado "
                        f"{valor_texto(diag.get('Llenado_Operativo_pct'), '.1f', '%')} · "
                        "Sumergencia "
                        f"{valor_texto(diag.get('Sumergencia_Relativa_pct'), '.1f', '%')} · "
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
                        f"Sumergencia: {valor_texto(diag.get('Sumergencia_Relativa_pct'), '.1f', '%')}  \n"
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
