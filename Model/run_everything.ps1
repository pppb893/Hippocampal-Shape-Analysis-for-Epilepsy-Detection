$ErrorActionPreference = "Continue"

$folders = @(
    "All_Augment_tain",
    "combind_methode",
    "Ds004469",
    "Ds004469Train_Ds005602test",
    "Ds005602",
    "Ds005602Train_Ds004469test"
)

$base_dir = $PSScriptRoot

foreach ($folder in $folders) {
    Write-Host "========================================="
    Write-Host "Running models in $folder"
    Write-Host "========================================="
    
    $folder_path = Join-Path $base_dir $folder
    $script_path = Join-Path $folder_path "run_all_models.ps1"
    
    if (Test-Path $script_path) {
        Set-Location $folder_path
        & $script_path
    } else {
        Write-Host "Could not find run_all_models.ps1 in $folder"
    }
}

Set-Location $base_dir
Write-Host "All datasets finished processing."
