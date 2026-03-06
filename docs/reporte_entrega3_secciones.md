# Secciones para el Reporte Final (Entrega 3)

---

## 1. Cambios Arquitectónicos frente a la Entrega 2

En la Entrega 2, el modelo y la API corrían en entornos locales o instancias estáticas (EC2), y los datos se almacenaban sin una segregación clara entre producción y experimentación. Para esta Entrega 3, la arquitectura evolucionó hacia un modelo **Cloud-Native y Serverless**:
1. **Despliegue Serverless:** Transición de entornos manuales a contenedores Docker gestionados automáticamente por AWS ECS Fargate.
2. **Data Lake Estructurado:** Migración del almacenamiento local de datos a un esquema de zonas (Golden y Analítica) en Amazon S3, unificando el control de versiones (DVC) y el monitoreo de ML.

---

## 2. Justificación del Servicio Cloud: AWS ECS (Fargate)

Evaluamos EKS, EC2 y ECS (Fargate) para desplegar la API. Elegimos **ECS con Fargate** por tres razones clave:

1. **Baja Complejidad Operativa:** Fargate elimina la necesidad de gestionar servidores subyacentes (parches, OS). EKS era desproporcionadamente complejo (gestión de clústeres Kubernetes) para las necesidades actuales del prototipo.
2. **Eficiencia en Costos:** EC2 cobra por hora encendida y EKS tiene un costo base mensual elevado ($73 USD). Fargate sigue un modelo *pay-as-you-go*, cobrando estrictamente por los recursos (CPU/RAM) consumidos durante la ejecución del contenedor.
3. **Escalabilidad y Ecosistema:** Integración nativa sin fricción con servicios de MLOps de AWS (ECR, S3, CloudWatch) mediante IAM Task Roles, facilitando el escalamiento automático en picos de demanda.

---

## 3. Arquitectura de Datos: S3 Data Lake y DVC

Implementamos un bucket unificado en Amazon S3 (`mlops-medical-project-uniandes-2026`) bajo el estándar de *Medallion Architecture*, dividido en dos zonas estrictas:

- **Zona Golden (`/golden`):** Entorno inmutable y de solo lectura para la API. Almacena el dataset curado y los modelos entrenados (`.joblib`). Se controla explícitamente usando **DVC**.
- **Zona Analítica (`/analytics`):** Entorno de escritura. Un middleware en la API intercepta asíncronamente cada request HTTP y guarda la entrada del usuario junto con la predicción del modelo en formato JSON para futuro monitoreo de *Data Drift*.

**Acceso Downstream para el Dashboard (Streaming de Históricos):**
Para desacoplar el frontend del almacenamiento subyacente y proteger las credenciales de AWS, se expone un endpoint dedicado en la API (`GET /dashboard-data`). En lugar de que Frontend consulte S3 directamente, este endpoint funciona como un puente: lee los registros históricos en la Zona Analítica, los consolida y los retorna al Frontend en un formato estructurado y unificado.

El endpoint está protegido por Autenticación Simple. El dashboard debe enviar:
`Headers: { "Authorization": "Bearer <API_SECRET_KEY_DEL_PROYECTO>" }`

> **Nota para Pruebas (Swagger UI o Terminal):**
> Dado que el endpoint requiere un *Bearer Token*, acceder desde la barra de direcciones arrojará un error `401 Unauthorized`. 
> - **Opción A (Visual):** Entre a `http://<IP_API>:8000/docs`, haga clic en "Authorize", pegue su token secreto y testee el `GET /dashboard-data` dando clic a "Try it out".
> - **Opción B (cURL):** Ejecute en su consola: 
>   `curl -s -H "Authorization: Bearer <API_SECRET>" http://<IP_API>:8000/dashboard-data`

**Estructura de la Respuesta (`/dashboard-data`):**
Devuelve un JSON con una lista consolidada `data` que contiene el histórico de predicciones. Cada elemento respeta el esquema `ApiCallLog`:

```json
{
  "data": [
    {
      "request_id": "b57e067c-cb6e-40c6...",
      "timestamp": "2026-03-06T06:35:55.072Z",
      "method": "POST",
      "path": "/predict",
      "request_body": "{\"transcription\": \"Patient presents with chest pain...\"}",
      "response_body": "{\"specialty\":\"Cardiology\",\"confidence\":0.85...}",
      "status_code": 200,
      "response_time_ms": 4.52
    }
  ]
}
```
*Con esta estructura, el frontend puede deserializar fácilmente `request_body` y `response_body` usando `pandas.json_normalize()` para graficar uso de modelos, confianzas promedio y distribuciones de especialidades.*

### Diagrama de Flujo MLOps (Estructura)

La arquitectura sigue el patrón **API Gateway** con almacenamiento desacoplado:

1. **Interacción de los Clientes:**
   - **Usuarios / Médicos:** Envían la transcripción usando el método `POST /predict`. Las peticiones llegan Directamente a la IP Pública Expuesta por AWS.
   - **Dashboard (Streamlit):** Consume el histórico usando el método `GET /dashboard-data`, enviando el token de autenticación a la misma IP.

2. **Capa de Cómputo (AWS ECS Fargate):**
   - El contenedor FastAPI recibe el tráfico gracias a su Interfaz de Red Elástica (ENI) pública asignada por Fargate.
   - Al encender, el contenedor descarga los modelos de Machine Learning pre-entrenados desde la *Zona Golden* de S3.

3. **Capa de Almacenamiento (S3 Data Lake) y DVC:**
   - **Zona Golden (`/golden`):** Contiene los datos crudos (`mtsamples.csv`) versionados por el remoto de **DVC** y los modelos serializados (`.joblib`). Es de **solo lectura** para la API.
   - **Zona Analítica (`/analytics`):** Cada vez que la API hace una predicción, un *middleware* guarda silenciosamente el texto original y la predicción en formato JSON.
   - Cuando el Dashboard pide los datos, la API descarga y unifica estos JSONs de la *Zona Analítica* para enviarlos listos para graficar.
