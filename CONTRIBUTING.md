# Guía de Contribución

## Convención de Nombres de Ramas

```
main              → Código listo para producción (protegida)
feature/*         → Nuevas funcionalidades (ej: feature/agregar-eda)
fix/*             → Corrección de errores
docs/*            → Actualizaciones de documentación
data/*            → Cambios relacionados con datos
```

## Flujo de Trabajo

1. **Crear una rama de feature**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/nombre-de-tu-feature
   ```

2. **Hacer commits con mensajes descriptivos**
   ```bash
   git add .
   git commit -m "feat: agregar exploración inicial de datos"
   ```

3. **Subir y crear Pull Request**
   ```bash
   git push origin feature/nombre-de-tu-feature
   ```
   Luego crear PR en GitHub y solicitar revisión.

4. **Después de la aprobación, hacer merge a main**

## Formato de Mensajes de Commit

Usar commits convencionales:

| Prefijo | Descripción |
|---------|-------------|
| `feat:` | Nueva funcionalidad |
| `fix:` | Corrección de error |
| `docs:` | Documentación |
| `data:` | Cambios de datos |
| `refactor:` | Refactorización de código |
| `test:` | Pruebas |
| `chore:` | Mantenimiento |

**Ejemplos:**
```
feat: agregar notebook de EDA para transcripciones médicas
data: agregar dataset original de Kaggle
docs: actualizar README con instrucciones de configuración
fix: corregir ruta de carga de datos
```

## Guías para Pull Request

- Describir qué cambios realizaste
- Referenciar issues relacionados si los hay
- Solicitar al menos 1 revisor
- Asegurar que todas las verificaciones pasen antes de hacer merge
