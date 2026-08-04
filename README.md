# Diagnóstico de pozos y evolución de cartas dinamométricas

Aplicación Streamlit para analizar cartas dinamométricas de fondo, consolidar
diagnósticos por pozo, estimar producción mediante un Virtual Flow Meter (VFM),
compararla con controles físicos y estudiar tendencias operativas.

> Estado documentado: 29 de julio de 2026  
> Versión visible de la aplicación: `2026-07-29-tendencias-bloqueo-v16`

## Qué resuelve

El proyecto recibe uno o más JSON de la API de EyesOn/Roch y:

1. valida la integridad y el orden de los puntos de cada carta;
2. separa las carreras ascendente y descendente;
3. identifica horizontales representativas;
4. construye una carta ideal;
5. calcula métricas geométricas y operativas;
6. aplica reglas diagnósticas independientes;
7. consolida diagnósticos robustos usando las últimas cinco cartas del pozo;
8. calcula VFM por pozo y día;
9. compara VFM con controles reales;
10. permite cargar tendencias históricas y analizar ventanas móviles de 15 días.

La salida es una ayuda para revisión técnica. Las recomendaciones son alertas y
no autorizan cambios automáticos de régimen, espaciamiento o intervención.

## Aplicaciones

- `app_pozos.py`: aplicación vigente, orientada a pozos, historia y tendencias.
- `app.py`: versión anterior conservada como referencia.
- `app_anterior.py`: respaldo adicional de una interfaz previa.

Para nuevas publicaciones debe usarse `app_pozos.py`.

## Estructura

```text
dashboard_dynas/
├── app_pozos.py
├── pipeline_diagnostico.py
├── vfm_produccion.py
├── controles_reales.py
├── modelos_finales.joblib.gz
├── controles_reales.xlsx
├── requirements.txt
├── instalar_tablero.bat
├── iniciar_tablero.bat
├── Script_JSON_Cartas.txt
├── Script_Tendencias.txt
├── README.md
└── docs/
    ├── ARQUITECTURA.md
    ├── DICCIONARIO_DATOS.md
    ├── ESTADO_ACTUAL.md
    ├── HISTORIAL_DECISIONES.md
    ├── OPERACION_Y_VERSIONADO.md
    ├── PENDIENTES.md
    └── REGLAS_DIAGNOSTICO.md
```

Los archivos `pipeline_diagnostico_*_v5.py` y `*_v6.py` son respaldos
intermedios. No deben importarse desde la aplicación vigente.

## Entradas

### JSON de cartas

Puede cargarse más de un JSON simultáneamente. Cada carta debe contener 80
posiciones y 80 cargas, en orden secuencial, tanto para fondo como para
superficie. El elemento `i` de posiciones corresponde al elemento `i` de
cargas.

Campos mínimos usados por el pipeline:

- `IdCarta`, `Pozo`, `Fecha`;
- `PosicionesFondo`, `CargasFondo`;
- `PosicionesSuperficie`, `CargasSuperficie`;
- `ProfundidadBomba`, `DiametroPistonBomba`, `GPM`.

La API aporta además variables como peso de fluido, sumergencia, llenado,
torque, carga estructural y datos requeridos por el VFM.

### CSV de tendencias

Se genera con `Script_Tendencias.txt` o con el script de PowerShell derivado de
ese archivo. La aplicación usa todas las mediciones y calcula indicadores sobre
los últimos 15 días calendario.

### Excel de controles reales

`controles_reales.xlsx` puede quedar junto a la aplicación o actualizarse desde
la barra lateral. Se usa para comparar producción bruta, petróleo y corte de
agua contra el VFM.

### Modelo VFM

`modelos_finales.joblib.gz` contiene el bundle de modelos y sus features. Debe
estar en el mismo directorio que `vfm_produccion.py`.

## Ejecución local

1. Instalar Python 3.11 o 3.12.
2. Ejecutar `instalar_tablero.bat` una sola vez.
3. Ejecutar `iniciar_tablero.bat`.
4. Cargar los JSON desde la barra lateral.
5. Opcionalmente cargar tendencias y controles reales.

Alternativa desde terminal:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app_pozos.py
```

## Streamlit Community Cloud

Configurar:

- repositorio y rama correctos;
- archivo principal: `app_pozos.py`;
- Python 3.12;
- `requirements.txt` en la raíz desplegada;
- modelo VFM y controles disponibles si se desean resultados precargados.

Después de modificar reglas:

1. actualizar `PIPELINE_CACHE_VERSION` en `app_pozos.py`;
2. hacer commit y push;
3. esperar el redeploy;
4. usar **Reprocesar todos los archivos**;
5. comprobar la versión visible en la barra lateral.

## Principios del diagnóstico

- Una carta inválida anula diagnósticos adicionales y resultados VFM.
- Los diagnósticos por carta pueden coexistir.
- “Pozo bien explotado” solo se asigna si no existe ninguna alerta de fondo,
  superficie o integridad.
- Un diagnóstico robusto es el que aparece al menos tres veces entre las
  últimas cinco cartas disponibles del pozo.
- La severidad de golpe de fluido/compresión puede calcularse internamente,
  pero actualmente no se muestra como clase separada.
- Los análisis temporales complementan al diagnóstico robusto; no lo
  reemplazan.

La especificación completa está en
[`docs/REGLAS_DIAGNOSTICO.md`](docs/REGLAS_DIAGNOSTICO.md).

## Documentación para retomar el proyecto

Antes de modificar el código, leer en este orden:

1. este `README.md`;
2. [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md);
3. [`docs/REGLAS_DIAGNOSTICO.md`](docs/REGLAS_DIAGNOSTICO.md);
4. [`docs/ESTADO_ACTUAL.md`](docs/ESTADO_ACTUAL.md);
5. [`docs/HISTORIAL_DECISIONES.md`](docs/HISTORIAL_DECISIONES.md);
6. [`docs/PENDIENTES.md`](docs/PENDIENTES.md).

Esto permite continuar el trabajo sin depender del historial del chat.

## Seguridad y datos

- La API key no debe guardarse en el repositorio, en scripts ni en capturas.
- Los JSON, CSV, Excel de controles y modelos pueden contener información
  confidencial del cliente.
- Antes de hacer público el repositorio, revisar permisos y eliminar o proteger
  todos los datos y modelos propietarios.
- La VPN y el endpoint privado solo deben usarse con las credenciales y
  autorizaciones del cliente.
