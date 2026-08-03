param(
    [Parameter(Mandatory = $true)]
    [string]$JsonPozos,

    [int]$Dias = 30,

    [string]$CarpetaSalida = "$env:USERPROFILE\Downloads",

    [string]$BaseUrl = "http://10.17.12.70:8075/api/datos-pozo/cartas",

    [int]$PageSize = 500
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $JsonPozos)) {
    throw "No existe el JSON indicado: $JsonPozos"
}

if ($Dias -lt 1) {
    throw "Dias debe ser mayor o igual que 1."
}

if (-not (Test-Path -LiteralPath $CarpetaSalida)) {
    New-Item `
        -ItemType Directory `
        -Path $CarpetaSalida `
        -Force | Out-Null
}

# Obtener la lista de pozos del JSON que se está analizando en Streamlit.
$jsonBase = Get-Content `
    -LiteralPath $JsonPozos `
    -Raw `
    -Encoding UTF8 | ConvertFrom-Json

if ($null -ne $jsonBase.items) {
    $itemsBase = @($jsonBase.items)
}
elseif ($jsonBase -is [System.Array]) {
    $itemsBase = @($jsonBase)
}
else {
    throw "El JSON no contiene una colección 'items' ni una lista de cartas."
}

$pozosObjetivo = New-Object `
    "System.Collections.Generic.HashSet[string]" `
    ([System.StringComparer]::OrdinalIgnoreCase)

foreach ($item in $itemsBase) {
    $pozo = [string]$item.Pozo
    if (-not [string]::IsNullOrWhiteSpace($pozo)) {
        [void]$pozosObjetivo.Add($pozo.Trim())
    }
}

if ($pozosObjetivo.Count -eq 0) {
    throw "No se encontraron nombres de pozos en el JSON."
}

Write-Host "Pozos objetivo: $($pozosObjetivo.Count)"

# La API key se solicita de forma oculta y no queda escrita en el script.
$claveSegura = Read-Host `
    "Ingresá la API key" `
    -AsSecureString

$apiKey = [Net.NetworkCredential]::new(
    "",
    $claveSegura
).Password.Trim()

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "La API key quedó vacía."
}

$headers = @{
    "EYESON-DATA-API_KEY" = $apiKey
}

# Se consulta por jornadas para evitar respuestas demasiado grandes.
$hastaGlobal = Get-Date
$desdeGlobal = $hastaGlobal.AddDays(-$Dias)

# CartaId evita duplicados en los límites entre jornadas.
$registrosPorCarta = @{}

for ($dia = 0; $dia -lt $Dias; $dia++) {
    $desdeDia = $desdeGlobal.Date.AddDays($dia)
    $hastaDia = $desdeDia.AddDays(1)

    if ($desdeDia -lt $desdeGlobal) {
        $desdeDia = $desdeGlobal
    }

    if ($hastaDia -gt $hastaGlobal) {
        $hastaDia = $hastaGlobal
    }

    if ($desdeDia -ge $hastaGlobal) {
        break
    }

    Write-Host ""
    Write-Host (
        "Consultando {0:yyyy-MM-dd HH:mm} a {1:yyyy-MM-dd HH:mm}" `
        -f $desdeDia, $hastaDia
    )

    $pagina = 1

    while ($true) {
        $desdeTexto = $desdeDia.ToString(
            "yyyy-MM-ddTHH:mm:ss"
        )
        $hastaTexto = $hastaDia.ToString(
            "yyyy-MM-ddTHH:mm:ss"
        )

        $url = (
            "$BaseUrl" +
            "?page_size=$PageSize" +
            "&page=$pagina" +
            "&desde=" +
            [uri]::EscapeDataString($desdeTexto) +
            "&hasta=" +
            [uri]::EscapeDataString($hastaTexto)
        )

        $respuesta = Invoke-RestMethod `
            -Uri $url `
            -Headers $headers `
            -Method Get `
            -TimeoutSec 120

        $itemsPagina = @($respuesta.items)

        if ($itemsPagina.Count -eq 0) {
            break
        }

        foreach ($carta in $itemsPagina) {
            $pozo = [string]$carta.Pozo

            if (-not $pozosObjetivo.Contains($pozo)) {
                continue
            }

            $id = [string]$carta.IdCarta

            # Se guardan valores originales y variables derivadas.
            $registrosPorCarta[$id] = [pscustomobject]@{
                CartaId = $carta.IdCarta
                Pozo = $carta.Pozo
                Fecha = $carta.Fecha
                Peso_Fluido_Promedio_lbf = $carta.PesoFluidoPromedio
                Peso_Fluido_Max_lbf = $carta.PesoFluidoMax
                Llenado_Bomba_API_pct = $carta.LlenadoBomba
                Carga_Maxima_Fondo_lbf = $carta.CargaMaximaBomba
                Carga_Minima_Fondo_lbf = $carta.CargaMinimaBomba
                Carrera_Fondo_Total_pulg = (
                    [double]$carta.CarreraMaximaBomba -
                    [double]$carta.CarreraMinimaBomba
                )
                Carrera_Fondo_Efectiva_pulg = (
                    [double]$carta.CarreraEfectivaBombaFin -
                    [double]$carta.CarreraEfectivaBombaInicio
                )
                Carga_Maxima_Superficie_lbf = $carta.CargaMaximaSuperficie
                Carga_Minima_Superficie_lbf = $carta.CargaMinimaSuperficie
                Carrera_Superficie_pulg = (
                    [double]$carta.CarreraMaximaSuperficie -
                    [double]$carta.CarreraMinimaSuperficie
                )
                Torque_Reductor_pct = $carta.PorcentajeTorqueReductorExistente
                Carga_Estructural_pct = $carta.PorcentajeCargaEstructural
                GPM = $carta.GPM
                Profundidad_Bomba_m = $carta.ProfundidadBomba
                Diametro_Piston_pulg = $carta.DiametroPistonBomba
                Sumergencia_API_m = $carta.Sumergencia
            }
        }

        Write-Host (
            "  Página ${pagina}: " +
            "$($itemsPagina.Count) cartas recibidas; " +
            "$($registrosPorCarta.Count) objetivo acumuladas"
        )

        $totalRecords = [int]$respuesta.totalRecords

        if (
            $itemsPagina.Count -lt $PageSize -or
            ($pagina * $PageSize) -ge $totalRecords
        ) {
            break
        }

        $pagina++
    }
}

$salida = Join-Path `
    $CarpetaSalida `
    (
        "tendencias-pozos-" +
        $desdeGlobal.ToString("yyyyMMdd") +
        "-a-" +
        $hastaGlobal.ToString("yyyyMMdd-HHmm") +
        ".csv"
    )

$registros = @(
    $registrosPorCarta.Values |
    Sort-Object Pozo, Fecha
)

$registros |
    Export-Csv `
        -LiteralPath $salida `
        -NoTypeInformation `
        -Encoding UTF8 `
        -Delimiter ";"

$apiKey = $null
$claveSegura = $null

Write-Host ""
Write-Host "Descarga terminada."
Write-Host "Registros exportados: $($registros.Count)"
Write-Host "Archivo: $salida"
