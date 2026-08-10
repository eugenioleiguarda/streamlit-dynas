# Pendientes y criterios por calibrar

## Prioridad alta

- Validar los umbrales con diagnósticos confirmados por especialistas de campo.
- Crear un conjunto etiquetado de cartas correctas y casos problemáticos.
- Medir falsos positivos y falsos negativos por diagnóstico.
- Incorporar pruebas automatizadas con cartas emblemáticas y sus `CartaId`.
- Verificar disponibilidad y estructura de Goodman por tramo en la API.
- Revisar unidades contractuales de todas las cargas y variables API.

## Diagnóstico

- Calibrar mejor pérdida de válvula viajera frente a fricción y cierre tardío.
- Determinar si “cierre tardío” debe denominarse apertura tardía de fija según
  criterio operativo del cliente.
- Investigar agarres/fricción como diagnóstico separado.
- Desarrollar pérdida de tubing con apoyo de carta de superficie e historia.
- Desarrollar posible pesca de varillas usando superficie, carga, carrera e
  historia.
- Definir severidad útil para golpe de fluido y gas antes de volver a mostrarla.
- Revisar comportamiento de cartas con sumergencia API contradictoria.

## Temporal

- Calibrar el puntaje diario de bloqueo con pozos bloqueados confirmados.
- Definir fecha de inicio con tolerancia a faltantes de datos.
- Evaluar ventanas de 7, 15 y 30 días.
- Incorporar persistencia temporal directamente al nivel de confianza.
- Comparar tendencias de peso, apertura, carrera y llenado contra eventos
  operativos conocidos.

## VFM

- Versionar formalmente el modelo y registrar fecha/dataset de entrenamiento.
- Validar el VFM por batería, rango de producción y antigüedad del control.
- Separar error de modelo de cambios reales ocurridos desde el último control.
- Evitar publicar el modelo en repositorios públicos sin autorización.

## Datos y automatización

- Automatizar descarga diaria solo desde un entorno con VPN y credenciales
  seguras.
- Corregir/confirmar paginación cuando `totalRecords` cambia durante la consulta.
- Implementar descarga incremental de tendencias sin duplicados.
- Guardar un manifiesto con rango temporal, cantidad de páginas y registros.
- Nunca incluir la API key en archivos versionados.

## Interfaz

- Mostrar claramente cuándo una métrica no aplica por carta inválida/sin trabajo.
- Hacer visible la versión del modelo VFM además de la versión del pipeline.
- Añadir una vista auditable de evidencias y umbrales para casos puntuales.
- Permitir exportar la serie temporal y los indicadores móviles por pozo.
- Evaluar autenticación antes de compartir datos sensibles mediante URL pública.

## Deuda técnica

- El pipeline contiene funciones anidadas heredadas del notebook; conviene
  separarlas en módulos de geometría, métricas y reglas.
- Existen respaldos históricos en la carpeta principal; moverlos a `archive/`
  cuando se confirme que no son importados.
- Corregir caracteres mojibake que persisten en algunos comentarios/textos
  heredados.
- Centralizar umbrales efectivos en una configuración versionada.
- Agregar type hints y pruebas unitarias.

