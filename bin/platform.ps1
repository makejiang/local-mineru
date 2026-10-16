$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot
try {
    if ($args.Count -ne 1 -or $args[0] -cne '--is-aipc') {
        [Console]::Error.WriteLine('Usage: platform.ps1 --is-aipc')
        exit 2
    }

    try {
        $modelName = Get-CimInstance -Query 'SELECT Name FROM Win32_Processor' |
            ForEach-Object { ([string]$_.Name).Trim() } |
            Where-Object { $_ } |
            Select-Object -First 1
    }
    catch {
        [Console]::Error.WriteLine('error: failed to query WMI')
        exit 1
    }

    $isAipc = $false
    if ($modelName) {
        $lastSpace = $modelName.LastIndexOf([char]' ')
        $code = if ($lastSpace -lt 0) {
            $modelName
        }
        else {
            $modelName.Substring($lastSpace + 1)
        }

        if ($code.Length -gt 0) {
            $firstCharacter = [int][char]$code[0]
            $startsWithDigit = $firstCharacter -ge [int][char]'0' -and
                $firstCharacter -le [int][char]'9'

            if ($startsWithDigit -and $code[0] -ne [char]'1') {
                $isAipc = $modelName.Contains('Ultra') -or
                    $firstCharacter -ge [int][char]'3'
            }
        }
    }

    if ($isAipc) {
        Write-Output '1'
    }
    else {
        Write-Output '0'
    }
}
finally {
    Pop-Location
}
