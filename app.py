from __future__ import annotations

import ast
import io
from math import ceil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline_diagnostico import a_array, procesar_json
from vfm_produccion import predecir_vfm
from controles_reales import (
    cruzar_controles,
    leer_controles,
    resumen_comparacion,
)


st.set_page_config(
    page_title="Diagnóstico de cartas dinamométricas",
    page_icon="📈",
    layout="wide",
)


COLORES = {
    "Exceso de carga estructural": "#991b1b",
    "Exceso de torque": "#dc2626",
    "Posible pozo subexplotado": "#16833b",
    "Posible sin trabajo de bomba": "#111827",
    "Posible golpe de fluido": "#e87918",
    "Posible compresión/interferencia de gas": "#2563eb",
    "Posible compresión/interferencia de gas suave": "#4f7fe5",
    "Posible pérdida en válvula viajera": "#8b5cf6",
    "Posible golpe de bomba": "#e11d48",
    "Posible tubing libre": "#795548",
    "Sin diagnóstico automático": "#6b7280",
}


def lista_alertas(valor):
    if isinstance(valor, list):
        return [str(x) for x in valor]
    if isinstance(valor, (tuple, set, np.ndarray)):
        return [str(x) for x in valor]
    if pd.isna(valor):
        return []
    if isinstance(valor, str):
        texto = valor.strip()
        if texto.startswith("["):
            try:
                parsed = ast.literal_eval(texto)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [texto]
    return [str(valor)]


@st.cache_data(show_spinner=False)
def ejecutar_pipeline(contenido: bytes):
    return procesar_json(contenido)


@st.cache_data(show_spinner=False)
def ejecutar_vfm(contenido: bytes):
    return predecir_vfm(contenido)


@st.cache_data(show_spinner=False)
def ejecutar_controles(contenido: bytes | None):
    return leer_controles(contenido)


def obtener_resultado(resultados, carta_id):
    fila = resultados.loc[resultados["CartaId"].astype(int) == int(carta_id)]
    return None if fila.empty else fila.iloc[0]


def figura_carta(carta, resultado, diagnostico, compacta=False):
    x = a_array(carta["Fondo_Posiciones"])
    y = a_array(carta["Fondo_Cargas"])
    principal = diagnostico.get("Diagnostico_Principal", "Sin diagnóstico automático")
    color = COLORES.get(principal, "#374151")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.append(x, x[0]), y=np.append(y, y[0]),
        mode="lines", name="Carta real",
        line=dict(color="#374151", width=2),
        fill="toself", fillcolor="rgba(92, 150, 190, 0.13)",
    ))

    vertices = resultado.get("Vertices_Ideal") if resultado is not None else None
    if vertices is not None:
        try:
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim == 2 and len(vertices) >= 4:
                cerrados = np.vstack([vertices[:4], vertices[0]])
                fig.add_trace(go.Scatter(
                    x=cerrados[:, 0], y=cerrados[:, 1], mode="lines",
                    name="Carta ideal", line=dict(color="#9063cd", width=3, dash="dash"),
                ))
        except Exception:
            pass

    fig.update_layout(
        title=dict(
            text=f"{carta['Pozo']} · Carta {int(carta['CartaId'])}<br><sup>{principal}</sup>",
            font=dict(color=color, size=15 if compacta else 19),
        ),
        height=330 if compacta else 560,
        margin=dict(l=35, r=20, t=65, b=35),
        xaxis_title="Posición",
        yaxis_title="Carga",
        hovermode="closest",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        template="plotly_white",
    )
    return fig


def tabla_exportable(df):
    salida = df.copy()
    for columna in salida.columns:
        if salida[columna].dtype == "object":
            salida[columna] = salida[columna].map(
                lambda x: " | ".join(lista_alertas(x)) if isinstance(x, (list, tuple, set, np.ndarray)) else x
            )
    return salida


def figura_barras_comparativas(tabla, titulo, eje_y):
    """Barras Real/VFM agrupadas, no apiladas."""
    fig = go.Figure()
    colores = {
        "Control real": "#78bff2",
        "VFM": "#0874d1",
    }
    for serie in ["Control real", "VFM"]:
        fig.add_trace(go.Bar(
            name=serie,
            x=tabla.index.astype(str),
            y=tabla[serie],
            marker_color=colores[serie],
            text=tabla[serie].map(lambda v: f"{v:,.1f}"),
            textposition="outside",
        ))
    fig.update_layout(
        title=titulo,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        yaxis_title=eje_y,
        legend=dict(
            orientation="h",
            y=1.12,
            x=1,
            xanchor="right",
        ),
        margin=dict(l=35, r=20, t=70, b=35),
        template="plotly_white",
    )
    return fig


def tarjeta_vfm_control(
    titulo,
    valor_vfm,
    valor_real,
    delta,
    unidad,
    decimales=2,
    unidad_delta=None,
):
    real_disponible = pd.notna(valor_real)
    texto_real = (
        f"{valor_real:.{decimales}f} {unidad}"
        if real_disponible
        else "Sin control"
    )
    texto_delta = (
        f"VFM − real: {delta:+.{decimales}f} {unidad_delta or unidad}"
        if real_disponible and pd.notna(delta)
        else "&nbsp;"
    )
    st.markdown(
        f"""
        <div style="
            border:1px solid rgba(128,128,128,.25);
            border-radius:10px;
            padding:12px 14px;
            min-height:122px;
        ">
            <div style="font-size:.95rem;font-weight:600;margin-bottom:8px;">
                {titulo}
            </div>
            <div style="display:flex;gap:22px;align-items:flex-start;">
                <div style="flex:1;">
                    <div style="font-size:.75rem;opacity:.65;">VFM</div>
                    <div style="font-size:1.45rem;font-weight:600;white-space:nowrap;">
                        {valor_vfm:.{decimales}f} {unidad}
                    </div>
                </div>
                <div style="flex:1;">
                    <div style="font-size:.75rem;opacity:.65;">Control real</div>
                    <div style="font-size:1.45rem;font-weight:600;white-space:nowrap;">
                        {texto_real}
                    </div>
                </div>
            </div>
            <div style="font-size:.78rem;margin-top:8px;color:#38b86b;">
                {texto_delta}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Diagnóstico de cartas dinamométricas")
st.caption("Exploración de oportunidades y alertas con reglas geométricas auditables")

archivo = st.sidebar.file_uploader(
    "Cargar respuesta JSON de la API",
    type=["json"],
    help="El archivo puede contener una lista directa o un objeto con la colección items.",
)

archivo_controles = st.sidebar.file_uploader(
    "Actualizar controles reales (opcional)",
    type=["xlsx"],
    help=(
        "Si no cargás un archivo se utiliza controles_reales.xlsx "
        "incluido en la aplicación."
    ),
)

if archivo is None:
    st.info("Cargá el JSON de la API desde la barra lateral para iniciar el análisis.")
    st.stop()

try:
    with st.spinner("Procesando cartas y calculando producción VFM…"):
        salida = ejecutar_pipeline(archivo.getvalue())
        produccion_vfm = ejecutar_vfm(archivo.getvalue())
        controles = ejecutar_controles(
            archivo_controles.getvalue()
            if archivo_controles is not None
            else None
        )
        comparacion_vfm = cruzar_controles(
            produccion_vfm,
            controles,
        )
except Exception as exc:
    st.error("No fue posible procesar el archivo.")
    st.exception(exc)
    st.stop()

muestra = salida["muestra"].copy()
resultados = salida["resultados_cartas"].copy()
diagnosticos = salida["diagnosticos_cartas"].copy()
diagnosticos["Alertas_lista"] = diagnosticos["Alertas"].map(lista_alertas)

df = diagnosticos.merge(
    muestra[["CartaId", "GPM", "ProfundidadBomba", "DiametroPistonBomba"]],
    on="CartaId", how="left", suffixes=("", "_API"),
)
df["Fecha_Dia"] = pd.to_datetime(
    df["Fecha"],
    errors="coerce",
).dt.normalize()
df = df.merge(
    comparacion_vfm.drop(columns=["Pozo_Clave"], errors="ignore"),
    on=["Pozo", "Fecha_Dia"],
    how="left",
    validate="many_to_one",
)

todos_principales = sorted(df["Diagnostico_Principal"].dropna().unique().tolist())
todas_alertas = sorted({alerta for lista in df["Alertas_lista"] for alerta in lista})

st.sidebar.header("Filtros")
filtro_principal = st.sidebar.multiselect("Diagnóstico principal", todos_principales)
filtro_alertas = st.sidebar.multiselect("Cualquier alerta", todas_alertas)
filtro_pozos = st.sidebar.multiselect("Pozo", sorted(df["Pozo"].dropna().unique().tolist()))
solo_multiples = st.sidebar.checkbox("Solo múltiples diagnósticos")
buscar = st.sidebar.text_input("Buscar pozo o CartaId")

filtrado = df.copy()
if filtro_principal:
    filtrado = filtrado[filtrado["Diagnostico_Principal"].isin(filtro_principal)]
if filtro_alertas:
    filtrado = filtrado[filtrado["Alertas_lista"].map(lambda xs: any(a in xs for a in filtro_alertas))]
if filtro_pozos:
    filtrado = filtrado[filtrado["Pozo"].isin(filtro_pozos)]
if solo_multiples:
    filtrado = filtrado[filtrado["Alertas_lista"].map(len) > 1]
if buscar.strip():
    q = buscar.strip().lower()
    filtrado = filtrado[
        filtrado["Pozo"].astype(str).str.lower().str.contains(q, regex=False)
        | filtrado["CartaId"].astype(str).str.contains(q, regex=False)
    ]

tab_resumen, tab_explorador, tab_detalle, tab_tabla = st.tabs(
    ["Resumen", "Explorador", "Detalle", "Tabla operativa"]
)

with tab_resumen:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cartas analizadas", len(df))
    c2.metric("Pozos únicos", df["Pozo"].nunique())
    c3.metric("Resultado del filtro", len(filtrado))
    c4.metric("Múltiples alertas", int(df["Alertas_lista"].map(len).gt(1).sum()))

    # El resumen debe responder a todos los filtros activos.
    # Se conserva una sola observación por pozo y día para evitar
    # duplicar caudales si hubiera más de una carta seleccionada.
    comparacion_filtrada = (
        filtrado
        .sort_values("Fecha")
        .drop_duplicates(
            subset=["Pozo", "Fecha_Dia"],
            keep="last",
        )
    )
    resumen_controles = resumen_comparacion(
        comparacion_filtrada
    )
    st.subheader("Producción VFM vs controles reales")
    r1, r2, r3 = st.columns(3)
    r1.metric(
        "Pozos con control comparable",
        f"{resumen_controles['cantidad']} / {resumen_controles['total']}",
    )
    r2.metric(
        "Cobertura",
        f"{resumen_controles['cobertura_pct']:.1f}%",
    )
    fechas_control_filtradas = pd.to_datetime(
        comparacion_filtrada.get("Fecha_Control"),
        errors="coerce",
    ).dropna()
    if not fechas_control_filtradas.empty:
        r3.metric(
            "Último control disponible",
            fechas_control_filtradas.max().strftime("%d/%m/%Y"),
        )
    if resumen_controles["cantidad"]:
        col_caudal, col_porcentaje = st.columns([2, 1])
        with col_caudal:
            st.caption("Suma sobre los mismos pozos coincidentes")
            st.plotly_chart(
                figura_barras_comparativas(
                    resumen_controles["caudales"],
                    "Caudales totales: control real vs VFM",
                    "Caudal [m³/d]",
                ),
                use_container_width=True,
            )
        with col_porcentaje:
            st.caption("Corte de agua ponderado por caudal bruto")
            st.plotly_chart(
                figura_barras_comparativas(
                    resumen_controles["porcentajes"],
                    "Corte de agua: control real vs VFM",
                    "Porcentaje [%]",
                ),
                use_container_width=True,
            )

        st.subheader("Promedio por pozo")
        col_promedio, col_promedio_pct = st.columns([2, 1])
        with col_promedio:
            st.plotly_chart(
                figura_barras_comparativas(
                    resumen_controles["promedios"],
                    "Caudal promedio por pozo: control real vs VFM",
                    "Caudal promedio [m³/d/pozo]",
                ),
                use_container_width=True,
            )
        with col_promedio_pct:
            st.plotly_chart(
                figura_barras_comparativas(
                    resumen_controles["porcentajes_promedio"],
                    "Corte de agua promedio por pozo",
                    "Porcentaje promedio [%]",
                ),
                use_container_width=True,
            )

        st.subheader("Estadísticas descriptivas comparativas")
        st.caption(
            "Media, mediana y desviación estándar calculadas sobre "
            "los mismos pozos con control comparable."
        )
        estadisticas = resumen_controles["estadisticas"].copy()
        for columna in ["Media", "Mediana", "Desviación estándar"]:
            estadisticas[columna] = estadisticas[columna].round(2)
        st.dataframe(
            estadisticas,
            use_container_width=True,
            hide_index=True,
        )

    conteo = df["Diagnostico_Principal"].value_counts().rename_axis("Diagnóstico").reset_index(name="Cantidad")
    st.subheader("Diagnóstico principal")
    st.bar_chart(conteo.set_index("Diagnóstico"))

    alertas_explotadas = df[["CartaId", "Alertas_lista"]].explode("Alertas_lista")
    alertas_explotadas = alertas_explotadas.dropna()
    conteo_alertas = alertas_explotadas["Alertas_lista"].value_counts().rename_axis("Alerta").reset_index(name="Cantidad")
    st.subheader("Todas las alertas, incluidas las secundarias")
    st.dataframe(conteo_alertas, use_container_width=True, hide_index=True)

with tab_explorador:
    st.subheader(f"Cartas encontradas: {len(filtrado)}")
    por_pagina = st.selectbox("Cartas por página", [6, 9, 12], index=1)
    paginas = max(1, ceil(len(filtrado) / por_pagina))
    pagina = st.number_input("Página", min_value=1, max_value=paginas, value=1)
    inicio = (pagina - 1) * por_pagina
    pagina_df = filtrado.iloc[inicio: inicio + por_pagina]

    for inicio_fila in range(0, len(pagina_df), 3):
        columnas = st.columns(3)
        for columna, (_, diag) in zip(columnas, pagina_df.iloc[inicio_fila: inicio_fila + 3].iterrows()):
            carta = muestra.loc[muestra["CartaId"].astype(int) == int(diag["CartaId"])].iloc[0]
            resultado = obtener_resultado(resultados, diag["CartaId"])
            with columna:
                st.plotly_chart(figura_carta(carta, resultado, diag, compacta=True), use_container_width=True)
                for alerta in diag["Alertas_lista"]:
                    st.caption(f"• {alerta}")

                vfm_bruta = diag.get("VFM_Bruta_m3_d", np.nan)
                vfm_neta = diag.get("VFM_Petroleo_m3_d", np.nan)
                real_bruta = diag.get("Control_Bruta_m3_d", np.nan)
                real_neta = diag.get("Control_Petroleo_m3_d", np.nan)

                st.markdown("**Caudales [m³/d]**")
                col_vfm, col_real = st.columns(2)
                with col_vfm:
                    st.caption("VFM")
                    st.write(
                        f"Bruta: **{vfm_bruta:.2f}**  \n"
                        f"Neta: **{vfm_neta:.2f}**"
                    )
                with col_real:
                    st.caption("Último control")
                    if pd.notna(real_bruta):
                        st.write(
                            f"Bruta: **{real_bruta:.2f}**  \n"
                            f"Neta: **{real_neta:.2f}**"
                        )
                        fecha_control = pd.to_datetime(
                            diag.get("Fecha_Control"),
                            errors="coerce",
                        )
                        if pd.notna(fecha_control):
                            st.caption(f"{fecha_control:%d/%m/%Y}")
                    else:
                        st.write("Sin control")

with tab_detalle:
    opciones = filtrado.apply(lambda r: f"{r['Pozo']} · Carta {int(r['CartaId'])}", axis=1).tolist()
    if not opciones:
        st.warning("No hay cartas que cumplan los filtros.")
    else:
        seleccion = st.selectbox("Seleccionar carta", opciones)
        posicion = opciones.index(seleccion)
        diag = filtrado.iloc[posicion]
        carta = muestra.loc[muestra["CartaId"].astype(int) == int(diag["CartaId"])].iloc[0]
        resultado = obtener_resultado(resultados, diag["CartaId"])
        st.plotly_chart(figura_carta(carta, resultado, diag), use_container_width=True)

        st.subheader("Virtual Flow Meter vs control real")
        t1, t2, t3 = st.columns(3)
        with t1:
            tarjeta_vfm_control(
                "Caudal bruto",
                diag.get("VFM_Bruta_m3_d", np.nan),
                diag.get("Control_Bruta_m3_d", np.nan),
                diag.get("Delta_Bruta_m3_d", np.nan),
                "m³/d",
                decimales=2,
            )
        with t2:
            tarjeta_vfm_control(
                "Petróleo neto",
                diag.get("VFM_Petroleo_m3_d", np.nan),
                diag.get("Control_Petroleo_m3_d", np.nan),
                diag.get("Delta_Petroleo_m3_d", np.nan),
                "m³/d",
                decimales=2,
            )
        with t3:
            tarjeta_vfm_control(
                "Corte de agua",
                diag.get("VFM_Agua_pct", np.nan),
                diag.get("Control_Agua_pct", np.nan),
                diag.get("Delta_Agua_pp", np.nan),
                "%",
                decimales=1,
                unidad_delta="pp",
            )

        if pd.notna(diag.get("Fecha_Control")):
            fecha_control = pd.to_datetime(diag["Fecha_Control"])
            st.caption(
                f"Control utilizado: {fecha_control:%d/%m/%Y} · "
                f"{int(diag.get('Control_Antiguedad_dias', 0))} días de antigüedad · "
                f"Estado: {diag.get('Estado_Control', '')}"
            )

        if pd.notna(diag.get("Fecha_Control")):
            st.info(
                "**Comentario de la comparación:** "
                + str(diag.get("Comentario_VFM_Control", ""))
            )

        izquierda, derecha = st.columns([1, 2])
        with izquierda:
            st.subheader("Diagnósticos")
            for alerta in diag["Alertas_lista"]:
                st.write(f"• {alerta}")
            st.metric("Llenado propio", f"{diag.get('Llenado_Original_pct', np.nan):.1f}%")
            st.metric("Llenado operativo", f"{diag.get('Llenado_Operativo_pct', np.nan):.1f}%")
            st.metric("Sumergencia relativa", f"{diag.get('Sumergencia_Relativa_pct', np.nan):.1f}%")
            st.metric("Torque reductor", f"{diag.get('Torque_Reductor_pct', np.nan):.1f}%")
            st.metric("Carga estructural", f"{diag.get('Carga_Estructural_pct', np.nan):.1f}%")

        with derecha:
            st.subheader("Métricas y evidencias")
            excluir = {"Alertas", "Alertas_lista"}
            detalle = pd.DataFrame(
                [(k, v) for k, v in diag.items() if k not in excluir and np.isscalar(v)],
                columns=["Variable", "Valor"],
            )
            st.dataframe(detalle, use_container_width=True, hide_index=True)
            evidencias = diag.get("Evidencias", [])
            if lista_alertas(evidencias):
                st.write("**Evidencias:**")
                for evidencia in lista_alertas(evidencias):
                    st.caption(f"• {evidencia}")

with tab_tabla:
    columnas_tabla = [
        "CartaId", "Pozo", "Fecha", "Diagnostico_Principal", "Alertas",
        "Llenado_Original_pct", "Llenado_Operativo_pct", "Sumergencia_Relativa_pct",
        "Torque_Reductor_pct", "Carga_Estructural_pct",
        "VFM_Bruta_m3_d", "VFM_Petroleo_m3_d", "VFM_Agua_pct",
        "VFM_Bruta_Via_Residuo_m3_d", "VFM_Petroleo_Via_Agua_m3_d",
        "Fecha_Control", "Control_Antiguedad_dias", "Estado_Control",
        "Control_Bruta_m3_d", "Control_Petroleo_m3_d", "Control_Agua_m3_d",
        "Control_Agua_pct", "Delta_Bruta_m3_d", "Error_Bruta_pct",
        "Delta_Petroleo_m3_d", "Error_Petroleo_pct", "Delta_Agua_pp",
        "Comentario_VFM_Control",
        "GPM", "ProfundidadBomba", "DiametroPistonBomba",
    ]
    columnas_tabla = [c for c in columnas_tabla if c in filtrado.columns]
    tabla = tabla_exportable(filtrado[columnas_tabla])
    st.dataframe(tabla, use_container_width=True, hide_index=True)
    csv = tabla.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar selección en CSV", csv, "diagnosticos_filtrados.csv", "text/csv")

st.sidebar.divider()
st.sidebar.caption(
    f"{len(muestra)} cartas válidas · {len(salida['invalidas'])} descartadas · "
    f"{len(salida['errores_cartas'])} errores técnicos"
)
