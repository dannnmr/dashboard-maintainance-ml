# Sistema de Autenticación - ML Dashboard

Este documento describe el sistema de autenticación implementado para el dashboard de mantenimiento predictivo.

## Características

- **3 Roles de usuario**: Administrador, Viewer y Técnico
- **Autenticación JWT**: Tokens seguros con expiración
- **Gestión de usuarios**: Solo administradores pueden crear usuarios
- **Protección de rutas**: Control de acceso basado en roles
- **Base de datos PostgreSQL**: Almacenamiento seguro de usuarios

## Instalación

### Backend

1. **Instalar dependencias**:
```bash
cd backend
pip install -r requirenments.txt
```

2. **Configurar variables de entorno**:
```bash
export DATABASE_URL="postgresql+asyncpg://postgres:2608@localhost:5432/db-proy-ml"
export MODEL_DIR="modelo/artifacts_anomalia"
export CORS_ALLOW_ORIGINS="http://localhost:3000"
```

3. **Crear usuario administrador inicial**:
```bash
python create_admin.py
```

4. **Ejecutar el servidor**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

1. **Instalar dependencias**:
```bash
cd frontend
npm install
```

2. **Configurar variables de entorno**:
Crear archivo `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. **Ejecutar el servidor de desarrollo**:
```bash
npm run dev
```

## Uso

### Credenciales por defecto

- **Email**: admin@example.com
- **Password**: admin123
- **Rol**: Administrador

### Funcionalidades por rol

#### Administrador
- Acceso completo al dashboard
- Crear, editar y desactivar usuarios
- Cambiar contraseñas de usuarios
- Ver todos los datos del sistema

#### Técnico
- Acceso al dashboard
- Ver datos de mantenimiento predictivo
- No puede gestionar usuarios

#### Viewer
- Acceso de solo lectura al dashboard
- Ver datos de mantenimiento predictivo
- No puede gestionar usuarios

## Endpoints de la API

### Autenticación
- `POST /auth/login` - Iniciar sesión
- `POST /auth/register` - Registrar usuario (solo admin)
- `GET /auth/me` - Obtener usuario actual
- `POST /auth/logout` - Cerrar sesión

### Gestión de usuarios (solo admin)
- `GET /users` - Listar usuarios
- `GET /users/{id}` - Obtener usuario por ID
- `PUT /users/{id}` - Actualizar usuario
- `POST /users/{id}/change-password` - Cambiar contraseña
- `DELETE /users/{id}` - Desactivar usuario

## Estructura del proyecto

### Backend
```
backend/
├── app/
│   ├── auth.py              # Lógica de autenticación JWT
│   ├── auth_routes.py       # Rutas de autenticación
│   ├── database.py          # Configuración de base de datos
│   ├── models.py            # Modelos de usuario y roles
│   ├── user_routes.py       # Rutas de gestión de usuarios
│   ├── user_service.py      # Lógica de negocio de usuarios
│   └── main.py              # Aplicación principal
├── create_admin.py          # Script para crear admin inicial
└── requirenments.txt        # Dependencias Python
```

### Frontend
```
frontend/
├── src/
│   ├── app/
│   │   ├── login/           # Página de login
│   │   ├── dashboard/       # Dashboard principal
│   │   └── users/           # Gestión de usuarios
│   ├── components/
│   │   ├── LoginForm.tsx    # Formulario de login
│   │   ├── UserManagement.tsx # Gestión de usuarios
│   │   ├── ProtectedRoute.tsx # Protección de rutas
│   │   └── Navbar.tsx       # Barra de navegación
│   ├── contexts/
│   │   └── AuthContext.tsx  # Contexto de autenticación
│   ├── lib/
│   │   ├── api.ts           # Cliente API
│   │   └── auth.ts          # Servicios de autenticación
│   └── types/
│       └── auth.ts          # Tipos TypeScript
└── package.json
```

## Seguridad

- **Contraseñas**: Hasheadas con bcrypt
- **JWT**: Tokens con expiración de 30 minutos
- **Validación**: Email y contraseñas con validación estricta
- **CORS**: Configurado para dominios específicos
- **Roles**: Control de acceso granular

## Desarrollo

### Agregar nuevos roles
1. Actualizar enum `UserRole` en `backend/app/models.py`
2. Actualizar tipos TypeScript en `frontend/src/types/auth.ts`
3. Actualizar validaciones en el frontend

### Modificar permisos
1. Actualizar decoradores en `backend/app/auth.py`
2. Actualizar `ProtectedRoute` en el frontend
3. Actualizar lógica de navegación

## Troubleshooting

### Error de conexión a base de datos
- Verificar que PostgreSQL esté ejecutándose
- Verificar credenciales en `DATABASE_URL`
- Verificar que la base de datos `db-proy-ml` exista

### Error de autenticación
- Verificar que el token JWT no haya expirado
- Verificar que el usuario esté activo
- Verificar permisos de rol

### Error de CORS
- Verificar `CORS_ALLOW_ORIGINS` en el backend
- Verificar que el frontend esté en el puerto correcto
