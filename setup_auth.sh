#!/bin/bash

# Script de configuración del sistema de autenticación
# ML Dashboard - Predictive Maintenance

echo "🚀 Configurando sistema de autenticación para ML Dashboard..."

# Verificar si Python está instalado
if ! command -v python &> /dev/null; then
    echo "❌ Python no está instalado. Por favor instala Python 3.8+"
    exit 1
fi

# Verificar si Node.js está instalado
if ! command -v node &> /dev/null; then
    echo "❌ Node.js no está instalado. Por favor instala Node.js 18+"
    exit 1
fi

# Verificar si PostgreSQL está instalado
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL no está instalado. Por favor instala PostgreSQL"
    exit 1
fi

echo "✅ Dependencias básicas verificadas"

# Configurar backend
echo "📦 Configurando backend..."
cd backend

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python -m venv venv
fi

# Activar entorno virtual
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Instalar dependencias
echo "Instalando dependencias de Python..."
pip install -r requirenments.txt

# Configurar variables de entorno
echo "Configurando variables de entorno..."
export DATABASE_URL="postgresql+asyncpg://postgres:2608@localhost:5432/db-proy-ml"
export MODEL_DIR="modelo/artifacts_anomalia"
export CORS_ALLOW_ORIGINS="http://localhost:3000"

# Crear usuario administrador
echo "Creando usuario administrador inicial..."
python create_admin.py

echo "✅ Backend configurado"

# Configurar frontend
echo "📦 Configurando frontend..."
cd ../frontend

# Instalar dependencias
echo "Instalando dependencias de Node.js..."
npm install

# Crear archivo de configuración
echo "Creando archivo de configuración..."
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF

echo "✅ Frontend configurado"

echo ""
echo "🎉 ¡Configuración completada!"
echo ""
echo "Para ejecutar el sistema:"
echo ""
echo "1. Backend (en una terminal):"
echo "   cd backend"
echo "   source venv/bin/activate  # En Windows: venv\\Scripts\\activate"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "2. Frontend (en otra terminal):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. Acceder a: http://localhost:3000"
echo ""
echo "Credenciales por defecto:"
echo "   Email: admin@example.com"
echo "   Password: admin123"
echo ""
echo "¡Disfruta tu sistema de autenticación! 🔐"
