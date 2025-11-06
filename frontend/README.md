## Frontend (Next.js)

### Descripción
Aplicación web del ML Dashboard con rutas de panel, predicciones, alertas, reportes y usuarios. Consume la API del backend mediante `NEXT_PUBLIC_API_URL`.

### Stack
- Next.js 15 (App Router)
- React 19
- Tailwind CSS 4
- Axios, React Hook Form, React Hot Toast, Recharts

### Configuración
- `next.config.ts` define `env.NEXT_PUBLIC_API_URL` (por defecto `http://localhost:8000`).
  - En desarrollo puedes exportar la variable:
    ```powershell
    $env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
    ```

### Scripts
```bash
npm install
npm run dev      # http://localhost:3000
npm run build
npm start
```

### Rutas principales (app router)
- `app/dashboard` → visión general.
- `app/login` → autenticación (JWT vía `/auth/login`).
- `app/predicciones` → consulta de predicciones.
- `app/alertas` → gestión/visualización de alertas.
- `app/reportes` → reportes.
- `app/transformadores` → activos.
- `app/users` → administración de usuarios.
- `app/test-predictions` → página de prueba.

### Autenticación
- `AuthContext` maneja sesión y token (cookies con `js-cookie`).
- Proteger vistas con `ProtectedRoute`.

### Consumo de API
- `src/lib/api.ts` centraliza `axios` con baseURL = `NEXT_PUBLIC_API_URL`.
- Requiere enviar `Authorization: Bearer <token>` tras login.

### Debug de endpoints
- Archivo `debug_frontend_predictions.html` (estático) para probar `login`, `GET /predictions/` y `GET /predictions/stats` contra el backend local.

### Buenas prácticas
- Restringir CORS en el backend para dominios del frontend en producción.
- Manejar expiración/renovación de JWT y errores globales en `axios`.

