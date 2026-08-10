# Historial de decisiones

Este archivo conserva el razonamiento del proyecto. No es un changelog línea
por línea: registra decisiones que no deberían perderse al continuar en otro
chat o con otro desarrollador.

## 1. Reconstrucción de cartas

- El archivo original era un backup SQL Server.
- Se exportó primero una muestra de cuatro cartas por pozo y día.
- El CSV histórico incluía índices de punto que permitían reconstruir el orden.
- La primera API ordenaba los puntos por posición y destruía el recorrido.
- La API fue corregida: el orden de cada colección ahora representa el orden
  secuencial de adquisición; posición `i` corresponde a carga `i`.
- No se necesita una colección adicional de índices si la API respeta ese
  contrato.

## 2. Horizontales y carta ideal

- Las horizontales se usan para construir una referencia geométrica y analizar
  la forma, no como una medición exacta del peso de fluido.
- En compresión de gas, la descendente puede contener varias mesetas; se busca
  una referencia baja, tardía y persistente, evitando fricción puntual.
- Los laterales de la carta ideal se construyen con puntos extremos reales
  proyectados sobre las horizontales.
- El lateral derecho se mantiene paralelo al izquierdo cuando corresponde.
- Se preservó la proyección de horizontales: no debe limitarse al pequeño tramo
  usado para estimarlas.

## 3. Peso de fluido y sumergencia

- La diferencia entre horizontales de la carta verdadera incluye efectos
  dinámicos, fricción y flotación.
- Se probó un cálculo propio, pero produjo sumergencias negativas y baja
  concordancia.
- Decisión vigente: usar peso de fluido y sumergencia informados por la API
  para el diagnóstico operativo.
- La sumergencia relativa se calcula como sumergencia/profundidad de bomba.
- Las horizontales propias se conservan para carta ideal y métricas de forma.

## 4. Llenado

- El llenado bruto puede superar 100 % si se cuenta área exterior.
- Se corrigió para usar área real dentro de la carta ideal.
- En pérdida de válvula viajera, los vacíos superiores no deben reducir el
  llenado operativo.
- El llenado operativo es el usado para subexplotación.
- Para cartas sin trabajo o inválidas, el llenado no debe interpretarse ni
  mostrarse como resultado operativo.

## 5. Diagnósticos independientes

- No se fuerza una clase única: una carta puede tener golpe de bomba y gas, por
  ejemplo.
- Se conserva un diagnóstico principal por prioridad y todas las alertas
  secundarias.
- La pérdida de viajera requiere vacíos superiores en ambos extremos.
- El hueco exclusivamente izquierdo con demora de transferencia se separó como
  cierre tardío de válvula viajera.
- Golpe de fluido y compresión de gas comparten falta de aporte; se distinguen
  por transferencia abrupta frente a progresiva.
- La severidad leve/moderada dejó de mostrarse como categoría independiente,
  aunque puede mantenerse calculada.

## 6. Sin trabajo, bloqueo y carta inválida

- Se separaron dos conceptos:
  - carta inválida: medición/transmisión no confiable;
  - sin trabajo: carta válida pero bomba bloqueada o sin apertura hidráulica.
- Una carta delgada puede representar bloqueo con ambas válvulas cerradas.
- El área/compacidad y la apertura central se usan para detectar este patrón.
- El extremo izquierdo se excluye de la apertura central para no confundir un
  golpe de bomba fuerte con trabajo hidráulico.
- Cruces concentrados en el extremo izquierdo pueden ser golpe de bomba y no
  invalidan automáticamente la carta.
- Una carta inválida elimina diagnósticos y VFM.

## 7. Subexplotación

- Umbral vigente: llenado operativo `>= 85 %` y sumergencia relativa `>= 10 %`.
- Es incompatible con golpe de fluido o compresión de gas.
- La recomendación es evaluar aumento de régimen, nunca efectuarlo
  automáticamente.
- Se agregó contexto temporal para distinguir estabilidad, debilitamiento,
  deterioro y aproximación al equilibrio.

## 8. Robustez por pozo

- El resumen principal debe contar pozos, no cartas.
- Se usan las últimas cinco cartas disponibles.
- Un diagnóstico robusto aparece al menos tres veces.
- Los diagnósticos débiles siguen visibles en el explorador individual.
- “Pozo bien explotado” participa en la misma regla de tres de cinco.
- Los filtros robustos y por carta son distintos.

## 9. VFM y controles

- El modelo recibido se aisló en `vfm_produccion.py`.
- Se agrupa por pozo/día y se usan medianas de las features.
- El tablero compara bruto, petróleo y agua contra controles físicos.
- Los controles se alinean temporalmente usando el último control no posterior
  a la fecha evaluada.
- Se muestran sumas, promedios, estadísticas y comentarios.

## 10. Tendencias

- El archivo histórico conserva todas las mediciones; no se reduce a una sola
  observación diaria para los gráficos.
- Para los indicadores se consolidan medianas diarias y una ventana móvil de
  15 días.
- Se calculan pendiente Theil–Sen, MAD, cambio 3d contra 12d y cobertura.
- El análisis temporal complementa tres familias:
  - subexplotación;
  - falta de aporte;
  - bloqueo/sin trabajo.

## 11. Interfaz

- La aplicación vigente se organiza en:
  - resumen por pozo;
  - explorador de cartas;
  - detalle del pozo;
  - descargas.
- En detalle se muestran las últimas cinco cartas, con paginación hacia atrás.
- Se muestran carrera de fondo por carta y carrera de superficie como
  configuración/metadata cuando está disponible.
- Los CSV deben incluir todos los diagnósticos, evidencias, acciones y métricas,
  no solo el diagnóstico principal.

