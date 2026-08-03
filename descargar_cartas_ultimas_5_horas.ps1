param(
    [int]$Horas = 5,

    [string]$CarpetaSalida = "$env:USERPROFILE\Downloads",

    [string]$BaseUrl = "http://10.17.12.70:8075/api/datos-pozo/cartas",

    [int]$PageSize = 500
)

$ErrorActionPreference = "Stop"

if ($Horas -lt 1) {
    throw "Horas debe ser mayor o igual que 1."
}

if (-not (Test-Path -LiteralPath $CarpetaSalida)) {
    New-Item `
        -ItemType Directory `
        -Path $CarpetaSalida `
        -Force | Out-Null
}

$hastaFecha = Get-Date
$desdeFecha = $hastaFecha.AddHours(-$Horas)

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

$todasLasCartas = New-Object System.Collections.ArrayList
$pagina = 1
$totalRecords = $null

try {
    do {
        # Se usa el mismo formato sin milisegundos que ya funcionó
        # correctamente en la descarga histórica.
        $desdeTexto = $desdeFecha.ToString(
            "yyyy-MM-ddTHH:mm:ss"
        )
        $hastaTexto = $hastaFecha.ToString(
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

        Write-Host "Descargando página $pagina..."

        $respuesta = Invoke-RestMethod `
            -Uri $url `
            -Headers $headers `
            -Method Get `
            -TimeoutSec 120

        if ($null -eq $respuesta.items) {
            throw (
                "La respuesta de la API no contiene la colección 'items'."
            )
        }

        $itemsPagina = @($respuesta.items)
        $totalRecords = [int]$respuesta.totalRecords

        foreach ($carta in $itemsPagina) {
            [void]$todasLasCartas.Add($carta)
        }

        Write-Host (
            "Página ${pagina}: " +
            "$($itemsPagina.Count) cartas. " +
            "Acumuladas: $($todasLasCartas.Count) de $totalRecords"
        )

        $pagina++

    } while (
        $itemsPagina.Count -eq $PageSize -and
        $todasLasCartas.Count -lt $totalRecords
    )

    if ($null -eq $totalRecords) {
        throw "No se obtuvo una respuesta válida de la API."
    }

    if ($todasLasCartas.Count -ne $totalRecords) {
        Write-Warning (
            "La API declaró $totalRecords registros, pero entregó " +
            "$($todasLasCartas.Count). Se guardará la colección " +
            "efectivamente devuelta por la API."
        )
    }

    $resultado = [ordered]@{
        totalRecords = $todasLasCartas.Count
        totalRecordsApi = $totalRecords
        desde = $desdeFecha.ToString("o")
        hasta = $hastaFecha.ToString("o")
        items = @($todasLasCartas)
    }

    $nombre = (
        "cartas-" +
        $desdeFecha.ToString("yyyyMMdd-HHmm") +
        "-a-" +
        $hastaFecha.ToString("HHmm") +
        ".json"
    )

    $salida = Join-Path `
        $CarpetaSalida `
        $nombre

    $json = $resultado |
        ConvertTo-Json `
            -Depth 100 `
            -Compress

    [System.IO.File]::WriteAllText(
        $salida,
        $json,
        [System.Text.UTF8Encoding]::new($false)
    )

    Write-Host ""
    Write-Host "JSON descargado correctamente:"
    Write-Host $salida
    Write-Host "Cartas descargadas: $($todasLasCartas.Count)"
    Write-Host "Desde: $desdeFecha"
    Write-Host "Hasta: $hastaFecha"
}
finally {
    $apiKey = $null
    $claveSegura = $null
}
