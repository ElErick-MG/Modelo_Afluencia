# 🇪🇨 API Predicción Turística Ecuador

API REST para predecir afluencia turística en Ecuador.

## Endpoints

- `GET /health` - Estado de la API
- `POST /predict/simple` - Predicción simplificada para N8N

## Uso N8N

**URL:** `POST https://tu-app.railway.app/predict/simple`

**Body:**
```json
{
  "Es_Feriado": 0,
  "Es_Vacaciones": 1,
  "Mes": 8,
  "Dia_Semana_Encoded": 6,
  "Trimestre": 3,
  "Temporada_Turistica_Encoded": 2,
  "provincia": "PICHINCHA"
}
```

**Respuesta:**
```json
{
  "afluencia": 28.5,
  "categoria": "ALTA",
  "success": true
}
```
