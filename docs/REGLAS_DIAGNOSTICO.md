# Reglas de diagnóstico vigentes

## Alcance

Este documento resume la intención técnica y los umbrales efectivos más
importantes del código vigente. El código en `pipeline_diagnostico.py` sigue
siendo la fuente ejecutable.

Las reglas son preliminares y fueron calibradas con ejemplos revisados
visualmente. No constituyen un diagnóstico definitivo de campo.

## Orden lógico

1. validar integridad geométrica;
2. determinar si existe trabajo de bomba;
3. calcular horizontales y carta ideal;
4. extraer métricas independientes;
5. aplicar diagnósticos compatibles;
6. resolver el diagnóstico principal por prioridad;
7. conservar el resto como alertas secundarias.

## Carta no válida

Representa una posible falla de medición, transmisión u orden de puntos.

Se evalúan, entre otros:

- saltos grandes;
- reversiones de posición;
- cruces propios;
- rango de carga respecto del peso API;
- continuidad y plausibilidad del contorno.

Los cruces agrupados exclusivamente en el extremo izquierdo pueden responder a
un golpe de bomba fuerte y no deben invalidar por sí solos una carta
geométricamente coherente.

Si la carta es inválida:

- se eliminan los demás diagnósticos;
- no se informa llenado como resultado operativo;
- no se usa para VFM;
- la acción sugerida es revisar medición/transmisión.

## Posible sin trabajo de bomba

Se conserva la lógica histórica de horizontales no confiables y se agregan dos
indicadores geométricos:

- compacidad de la carta `<= 0.16`;
- apertura central normalizada `<= 0.24`, excluyendo el extremo izquierdo,
  combinada con sumergencia negativa o compacidad `<= 0.25`.

La exclusión del extremo izquierdo evita que un golpe de bomba profundo infle
artificialmente el área o la apertura hidráulica.

Este diagnóstico es incompatible con:

- golpe de fluido;
- compresión/interferencia de gas;
- pozo subexplotado;
- válvula viajera;
- tubing libre.

El golpe de bomba puede coexistir porque puede ser una estrategia operativa
para intentar evitar un bloqueo por gas.

## Pérdida en válvula viajera

Requiere:

- carta e horizontales válidas;
- ausencia de “sin trabajo”;
- vacíos en ambos cuadrantes superiores;
- vacío inferior derecho limitado;
- laterales ideales inclinados con ángulos menores a 90°;
- ausencia de cierre tardío.

Los valores base declarados en el pipeline son:

- vacío superior izquierdo: `15 %`;
- vacío superior derecho: `3 %`;
- vacío inferior derecho máximo: `35 %`.

Existe además una condición alternativa para vacíos superiores combinados, a
fin de estabilizar cartas visualmente equivalentes.

Cuando se identifica pérdida de viajera, los vacíos superiores se reintegran
al llenado operativo porque representan transferencia de carga, no pérdida
real de llenado.

## Posible cierre tardío de válvula viajera

Diagnóstico separado de pérdida.

Busca que, al inicio de la carrera ascendente:

- la carga permanezca próxima a la carga inferior;
- se demore la transferencia del 50 %;
- se demore la transferencia del 80 %;
- exista principalmente vacío superior izquierdo y no los dos vacíos
  superiores típicos de una pérdida.

Puede indicar suciedad, falla de cierre o un dispositivo/pin antibloqueo.

## Golpe de fluido

Requiere una transferencia sostenida en el sector inferior derecho y una
geometría compatible con llenado incompleto.

Condición marcada:

- vacío superior derecho `>= 20 %`;
- vacío inferior derecho `>= 30 %`;
- llenado `< 90 %`.

Condición suave/fronteriza:

- vacío superior derecho `>= 4 %`;
- vacío inferior derecho `>= 10 %`;
- llenado `< 92 %`.

La transferencia se considera abrupta principalmente cuando:

- comienza antes del extremo final;
- el ancho 20–80 % es reducido.

Hay respaldos acotados para cartas con llenado entre `85–92 %` y para
coexistencia con golpe de bomba. La severidad se calcula internamente como
leve/moderada, pero no se presenta como diagnóstico distinto.

## Compresión/interferencia de gas

Comparte los requisitos geométricos de admisión con golpe de fluido, pero la
transferencia derecha es progresiva o redondeada.

Se contemplan:

- transferencia progresiva mensurable;
- transferencia progresiva inferida cuando las métricas no están disponibles;
- transferencia tardía suave con vacío inferior apreciable.

La severidad leve/moderada queda calculada internamente y se muestra bajo una
única denominación.

## Golpe de bomba

Busca una excursión:

- bajo la horizontal inferior;
- breve y profunda;
- localizada exclusivamente en el extremo izquierdo;
- con mínimo dentro del primer tramo de carrera.

El umbral base de profundidad es aproximadamente `12 %` de la altura útil. La
medición usa ambas ramas porque el punto de separación puede ubicar los puntos
del impacto en cualquiera de ellas.

Puede coexistir con otros diagnósticos, incluido “sin trabajo de bomba”.

## Posible pozo subexplotado

Condición efectiva:

- carta e horizontales válidas;
- trabajo de bomba;
- datos operativos válidos;
- llenado operativo `>= 85 %`;
- sumergencia relativa `>= 10 %`;
- ausencia de golpe de fluido;
- ausencia de compresión de gas;
- ausencia de transferencia inferior sostenida.

Es deliberadamente incompatible con falta de aporte: si la carta evidencia
golpe de fluido o gas, no se recomienda aumentar régimen aunque la sumergencia
API sea elevada.

## Posible tubing libre

Usa los laterales de la carta ideal:

- ángulo ideal izquierdo `< 83°`;
- ángulo ideal derecho `>= 90°`;
- carta válida y con trabajo;
- sin pérdida ni cierre tardío de válvula viajera.

## Alertas de superficie

- `Exceso de torque`: torque de reductora `> 105 %`.
- `Exceso de carga estructural`: carga en viga `> 100 %`.

Estas alertas impiden clasificar la carta como “Pozo bien explotado”.

El proyecto contempla el diagnóstico de riesgo por Goodman cuando el JSON
aporte la estructura de tramos correspondiente. Debe verificarse en cada nueva
versión de la API que los datos estén presentes y correctamente normalizados.

## Pozo bien explotado

Solo se asigna si:

- la carta es válida;
- hay trabajo de bomba;
- no hay golpe de fluido ni gas;
- no hay golpe de bomba;
- no hay problemas de válvula;
- no hay tubing libre;
- no está subexplotado;
- no hay exceso de torque ni carga estructural;
- no existe ninguna otra alerta.

No significa que el pozo esté probado como óptimo; significa que el sistema no
encontró ninguna anomalía bajo las reglas actuales.

## Diagnósticos robustos por pozo

Se analizan las últimas cinco cartas disponibles. Un diagnóstico es robusto
cuando aparece en al menos tres. Se contabilizan diagnósticos principales y
secundarios, sin duplicar un diagnóstico dentro de la misma carta.

El filtro robusto trabaja a nivel pozo. El filtro individual trabaja a nivel
carta y, en detalle, conserva los pozos que tengan al menos una carta
coincidente.

## Análisis temporal

Los análisis temporales se ejecutan solo cuando existe el diagnóstico robusto
correspondiente.

### Indicadores móviles de 15 días

Para cada variable se calcula:

- último valor;
- mediana diaria y mediana de 15 días;
- pendiente robusta Theil–Sen;
- pendiente relativa;
- volatilidad MAD;
- diferencia entre últimos 3 días y 12 días anteriores;
- cantidad de días y mediciones;
- calidad de cobertura.

Variables principales:

- llenado API;
- sumergencia relativa API;
- peso de fluido;
- cargas máxima, mínima y apertura;
- carrera de fondo y superficie;
- eficiencia fondo/superficie;
- torque y carga estructural.

### Subexplotación

Clasifica la evolución como:

- condición estable;
- oportunidad debilitándose;
- condición de extracción deteriorándose;
- aproximándose al equilibrio operativo;
- comportamiento volátil;
- no concluyente/insuficiente.

### Falta de aporte

Para golpe de fluido o gas:

- restricción agravándose;
- restricción estable;
- recuperación;
- posible deterioro/recuperación;
- volátil/no concluyente.

### Bloqueo

Para “sin trabajo de bomba” robusto:

- bloqueo reciente;
- bloqueo persistente en 15 días;
- bloqueo intermitente;
- posible desbloqueo reciente;
- inicio temporal incierto;
- evidencia insuficiente.

La clasificación diaria combina llenado, peso de fluido, apertura de cargas y
relación de carrera fondo/superficie.

