# Despliegue en Railway (Frontend)

Este proyecto usa `railway.toml` para autoconfigurar el despliegue del frontend (Streamlit) en [Railway](https://railway.app/).

### Pasos Rápidos
1. **Crear Proyecto:** En Railway, haz clic en **New Project** > **Deploy from GitHub repo** y selecciona tu repositorio.
2. **Variables de Entorno:** Ve a la pestaña **Variables** y agrega:
   - `API_URL`: URL pública de tu backend (ej. `http://<ip-ec2>:8000`).
   - `DASHBOARD_TOKEN`: Clave secreta para proteger los reportes.
   - `MODEL_VERSION`: Versión actual del modelo (ej. `v3.0-sigmoid-10classes`).
3. **Desplegar:** Railway construirá la imagen Docker automáticamente. Revisa el progreso en **Deployments**.
4. **Generar Dominio:** Una vez desplegado, ve a **Settings > Networking > Generate Domain** para obtener la URL pública de tu aplicación.

> **Solución de problemas:** Si la construcción falla por configuración, verifica en *Settings* que **Root Directory** sea `/` y **Dockerfile Path** sea `frontend/Dockerfile`.
