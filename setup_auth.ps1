# Script de configuración del sistema de autenticación
# ML Dashboard - Predictive Maintenance

Write-Host "🚀 Configurando sistema de autenticación para ML Dashboard..." -ForegroundColor Green

# Verificar si Python está instalado
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python encontrado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python no está instalado. Por favor instala Python 3.8+" -ForegroundColor Red
    exit 1
}

# Verificar si Node.js está instalado
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Node.js encontrado: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js no está instalado. Por favor instala Node.js 18+" -ForegroundColor Red
    exit 1
}

# Verificar si PostgreSQL está instalado
try {
    $psqlVersion = psql --version 2>&1
    Write-Host "✅ PostgreSQL encontrado: $psqlVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ PostgreSQL no está instalado. Por favor instala PostgreSQL" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Dependencias básicas verificadas" -ForegroundColor Green

# Configurar backend
Write-Host "📦 Configurando backend..." -ForegroundColor Yellow
Set-Location backend

# Crear entorno virtual si no existe
if (-not (Test-Path "venv")) {
    Write-Host "Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv venv
}

# Activar entorno virtual
Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Instalar dependencias
Write-Host "Instalando dependencias de Python..." -ForegroundColor Yellow
pip install -r requirenments.txt

# Configurar variables de entorno
Write-Host "Configurando variables de entorno..." -ForegroundColor Yellow
$env:DATABASE_URL = "postgresql+asyncpg://postgres:2608@localhost:5432/db-proy-ml"
$env:MODEL_DIR = "modelo/artifacts_anomalia"
$env:CORS_ALLOW_ORIGINS = "http://localhost:3000"

# Crear usuario administrador
Write-Host "Creando usuario administrador inicial..." -ForegroundColor Yellow
python create_admin.py

Write-Host "✅ Backend configurado" -ForegroundColor Green

# Configurar frontend
Write-Host "📦 Configurando frontend..." -ForegroundColor Yellow
Set-Location ../frontend

# Instalar dependencias
Write-Host "Instalando dependencias de Node.js..." -ForegroundColor Yellow
npm install

# Crear archivo de configuración
Write-Host "Creando archivo de configuración..." -ForegroundColor Yellow
@"
NEXT_PUBLIC_API_URL=http://localhost:8000
"@ | Out-File -FilePath ".env.local" -Encoding UTF8

Write-Host "✅ Frontend configurado" -ForegroundColor Green

Write-Host ""
Write-Host "🎉 ¡Configuración completada!" -ForegroundColor Green
Write-Host ""
Write-Host "Para ejecutar el sistema:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Backend (en una terminal):" -ForegroundColor Yellow
Write-Host "   cd backend" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "2. Frontend (en otra terminal):" -ForegroundColor Yellow
Write-Host "   cd frontend" -ForegroundColor White
Write-Host "   npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "3. Acceder a: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Credenciales por defecto:" -ForegroundColor Cyan
Write-Host "   Email: admin@example.com" -ForegroundColor White
Write-Host "   Password: admin123" -ForegroundColor White
Write-Host ""
Write-Host "¡Disfruta tu sistema de autenticación! 🔐" -ForegroundColor Green
