# Operación y versionado

## Regla de oro

No sobrescribir una versión estable sin antes crear un commit identificable.

## Flujo recomendado para un cambio

1. Guardar el caso que motiva el cambio:
   - `CartaId`;
   - pozo y fecha;
   - diagnóstico esperado;
   - métricas relevantes;
   - captura opcional.
2. Modificar la regla en `pipeline_diagnostico.py`.
3. Ejecutar pruebas de sintaxis.
4. Probar el caso objetivo.
5. Probar un conjunto de regresión de cartas anteriormente correctas.
6. Actualizar documentación si cambia una regla o umbral.
7. Cambiar `PIPELINE_CACHE_VERSION`.
8. Hacer commit.
9. Desplegar.
10. Reprocesar y comprobar la versión visible.

## Convención de commits

Ejemplos:

```text
fix(diagnostico): limitar carta invalida a cruces distribuidos
feat(tendencias): detectar bloqueo intermitente en ventana 15d
fix(app): aplicar filtro individual al detalle de pozos
docs: actualizar reglas de valvula viajera
```

## Tags sugeridos

Para estados validados:

```text
v2026.07.29-bloqueo-temporal
v2026.08.XX-calibracion-cliente
```

Un tag debe representar código, modelo y documentación compatibles.

## Pruebas mínimas

### Sintaxis

```powershell
python -m py_compile pipeline_diagnostico.py app_pozos.py `
    vfm_produccion.py controles_reales.py
```

### Inicio local

```powershell
python -m streamlit run app_pozos.py
```

Verificar:

- carga de uno y varios JSON;
- filtros robustos e individuales;
- resumen, explorador, detalle y descargas;
- paginación;
- tendencias opcionales;
- VFM y controles;
- ausencia de IDs duplicados de Plotly/Streamlit.

### Regresión diagnóstica

Mantener una tabla fuera del repositorio público:

| CartaId | Diagnóstico esperado | No debe aparecer | Motivo |
|---|---|---|---|
| ... | ... | ... | ... |

Antes de publicar, comparar el resultado nuevo contra la versión estable y
revisar todo cambio no intencional.

## Respaldo y recuperación

El repositorio debe contener:

- código vigente;
- documentación;
- scripts reproducibles;
- `requirements.txt`.

Los datos sensibles pueden guardarse en almacenamiento privado separado. Para
reproducir una corrida conviene registrar:

- hash/commit;
- versión de pipeline;
- nombre y fecha del modelo VFM;
- nombres y rangos de los JSON;
- rango del CSV de tendencias;
- fecha del Excel de controles.

## Streamlit Cloud

Si el tablero queda negro o parece no reprocesar:

1. abrir **Manage app** y revisar logs;
2. confirmar archivo principal y rama;
3. comprobar que el commit nuevo esté desplegado;
4. verificar la versión visible;
5. limpiar caché;
6. retirar y volver a cargar los JSON si el uploader perdió su estado;
7. pulsar reprocesar.

Si cambia el código pero no `PIPELINE_CACHE_VERSION`, los resultados anteriores
pueden seguir cacheados.

## Seguridad

No versionar:

- API keys;
- contraseñas VPN;
- configuraciones con secretos;
- JSON reales sin autorización;
- datos de producción o controles en repositorios públicos;
- modelos propietarios sin permiso.

