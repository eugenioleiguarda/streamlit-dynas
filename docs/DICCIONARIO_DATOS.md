# Diccionario de datos

## Convenciones

- Fondo y superficie se almacenan como pares de listas ordenadas.
- Las cargas de las cartas se tratan como `lbf`.
- Las posiciones/carreras se muestran en `pulg`.
- Profundidad y sumergencia se muestran en `m`.
- Producción se muestra en `m³/d`.
- Torque, carga estructural, llenado y corte de agua se muestran en `%`.

## Campos principales de la API

| Campo API | Significado | Unidad/forma |
|---|---|---|
| `IdCarta` | Identificador único de carta | entero |
| `Pozo` | Nombre del pozo | texto |
| `Fecha` | Fecha de adquisición | ISO datetime |
| `PosicionesFondo` | Posiciones secuenciales de fondo | lista, 80 puntos |
| `CargasFondo` | Cargas correspondientes de fondo | lista, 80 puntos, lbf |
| `PosicionesSuperficie` | Posiciones secuenciales de superficie | lista, 80 puntos |
| `CargasSuperficie` | Cargas correspondientes de superficie | lista, 80 puntos, lbf |
| `ProfundidadBomba` | Profundidad de bomba | m |
| `DiametroPistonBomba` | Diámetro de pistón | pulg |
| `GPM` | Régimen de bombeo | golpes/min |
| `LlenadoBomba` | Llenado informado por API | % |
| `PesoFluidoPromedio` | Peso de fluido API | lbf |
| `Sumergencia` | Sumergencia API | m |
| `PorcentajeTorqueReductorExistente` | Utilización de reductora | % |
| `PorcentajeCargaEstructural` | Utilización estructural | % |

La API puede aportar muchos otros campos usados por el VFM. La lista de
features efectiva se obtiene del bundle `modelos_finales.joblib.gz`.

## Campos normalizados

| Campo | Origen |
|---|---|
| `CartaId` | renombre de `IdCarta` |
| `Fondo_Posiciones` | `PosicionesFondo` |
| `Fondo_Cargas` | `CargasFondo` |
| `Superficie_Posiciones` | `PosicionesSuperficie` |
| `Superficie_Cargas` | `CargasSuperficie` |
| `Torque_Reductor_pct` | API |
| `Carga_Estructural_pct` | API |
| `Carta_Valida` | validación de listas y finitud |

## Métricas calculadas por carta

| Campo | Descripción |
|---|---|
| `Llenado_Bruto_pct` | área real/ideal antes de limitar área exterior |
| `Llenado_Original_pct` | porcentaje de área real dentro de la carta ideal |
| `Llenado_Operativo_pct` | llenado ajustado para diagnóstico |
| `Sumergencia_Relativa_pct` | sumergencia API/profundidad × 100 |
| `Compacidad_Carta` | área real/rectángulo envolvente |
| `Apertura_Central_Carta_pct` | separación mediana entre carreras en zona central |
| `Vacio_Superior_Izquierdo_pct` | área faltante por cuadrante |
| `Vacio_Superior_Derecho_pct` | área faltante por cuadrante |
| `Vacio_Inferior_Izquierdo_pct` | área faltante por cuadrante |
| `Vacio_Inferior_Derecho_pct` | área faltante por cuadrante |
| `Pendiente_Transferencia_Derecha` | inclinación de transferencia |
| `Curvatura_Transferencia_Derecha` | cambio de pendiente |
| `Ancho_Transferencia_20_80_pct` | recorrido para pasar de 20 a 80 % de carga |
| `Inicio_Transferencia_Derecha_pct` | inicio normalizado de transferencia |
| `Profundidad_Golpe_Inferior_pct` | profundidad del golpe de bomba |
| `Ancho_Golpe_Inferior_pct` | ancho del golpe respecto de carrera |
| `Angulo_Ideal_Izquierdo_deg` | ángulo lateral ideal |
| `Angulo_Ideal_Derecho_deg` | ángulo lateral ideal |
| `Carrera_Fondo_pulg` | rango de posiciones de fondo |

## Campos de diagnóstico

| Campo | Tipo |
|---|---|
| `Diagnostico_Principal` | texto |
| `Alertas` | lista de diagnósticos principales/secundarios |
| `Evidencias` | lista de razones auditables |
| `Accion_Sugerida` | texto |
| `Confianza` | número/etiqueta |
| `Carta_No_Valida` | booleano |
| `Sin_Trabajo_Bomba` | booleano |
| `Perdida_Valvula_Viajera` | booleano |
| `Cierre_Tardio_Valvula_Viajera` | booleano |
| `Golpe_Fluido` | booleano |
| `Compresion_Gas` | booleano |
| `Golpe_Bomba` | booleano |
| `Tubing_Libre` | booleano |
| `Pozo_Subexplotado` | booleano |
| `Exceso_Torque` | booleano |
| `Exceso_Carga_Estructural` | booleano |

## VFM

| Campo | Unidad |
|---|---|
| `VFM_Num_Cartas_Dia` | cantidad |
| `VFM_Bruta_m3_d` | m³/d |
| `VFM_Petroleo_m3_d` | m³/d |
| `VFM_Agua_pct` | % |
| `VFM_Bruta_Via_Residuo_m3_d` | m³/d |
| `VFM_Petroleo_Via_Agua_m3_d` | m³/d |

El VFM agrupa cartas por pozo y día y usa medianas de las features requeridas.

## Controles reales y comparación

| Campo | Descripción |
|---|---|
| `Fecha_Control` | fecha del control seleccionado |
| `Estado_Control` | estado operativo del control |
| `Control_Bruta_m3_d` | líquido real |
| `Control_Petroleo_m3_d` | petróleo real |
| `Control_Agua_m3_d` | agua real |
| `Control_Agua_pct` | corte de agua real |
| `Control_Antiguedad_dias` | diferencia contra fecha VFM |
| `Delta_Bruta_m3_d` | VFM menos real |
| `Error_Bruta_pct` | error relativo |
| `Delta_Petroleo_m3_d` | VFM menos real |
| `Error_Petroleo_pct` | error relativo |
| `Delta_Agua_pp` | diferencia de corte de agua |
| `Comentario_VFM_Control` | comentario automático |

## CSV de tendencias

Campos esperados:

- `Pozo`, `Fecha`;
- `Llenado_Bomba_API_pct`;
- `Sumergencia_API_m`;
- `Profundidad_Bomba_m`;
- `Peso_Fluido_Promedio_lbf`;
- `Carga_Maxima_Fondo_lbf`, `Carga_Minima_Fondo_lbf`;
- `Carga_Maxima_Superficie_lbf`, `Carga_Minima_Superficie_lbf`;
- `Carrera_Fondo_Total_pulg`, `Carrera_Superficie_pulg`;
- `Torque_Reductor_pct`, `Carga_Estructural_pct`.

Campos derivados:

- `Sumergencia_Relativa_API_pct`;
- `Rango_Carga_Fondo_lbf`;
- `Eficiencia_Carrera_pct`;
- indicadores estadísticos de 15 días.

