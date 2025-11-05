param(
  [string]$Root = "C:\Users\Asus TUF\Desktop\proy-ml-dashboard"
)
$env:PYTHONPATH = $Root
$python = Join-Path $Root "venv\Scripts\python.exe"
$logDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

& $python "$Root\etl\capa_bronze\main_bronze.py"  *> "$logDir\bronze.log"
& $python "$Root\etl\capa_silver\main_silver.py"  *> "$logDir\silver.log"
& $python "$Root\etl\capa_gold\main_gold.py"      *> "$logDir\gold.log"


#schtasks /Create /TN "MLD_Batch_Pipelines" /TR "powershell.exe -ExecutionPolicy Bypass -File C:\Users\Asus TUF\Desktop\proy-ml-dashboard\run_pipelines.ps1" /SC HOURLY /MO 1 /F