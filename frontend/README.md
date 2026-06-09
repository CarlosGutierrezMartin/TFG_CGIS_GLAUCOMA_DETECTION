# Frontend — Cribado de glaucoma

## Descripción

Interfaz single page en HTML5, CSS inline y JavaScript vanilla para la API REST de detección temprana de glaucoma.

La pantalla principal está orientada a cribado clínico serio: muestra el porcentaje calibrado, la banda de riesgo, una explicación prudente, el aviso clínico y las imágenes principal/original + predicción. La información técnica queda en capas secundarias.

## Características

- Página única sin dependencias externas ni paso de build.
- Consulta aleatoria de casos Test400 con un botón.
- Resultado principal con:
  - probabilidad de signos compatibles con sospecha glaucomatosa;
  - banda baja/intermedia/moderada;
  - recomendación no diagnóstica;
  - imagen original y overlay de predicción.
- Botón `Más` con modal de biomarcadores, GT, acierto, score, umbral, calidad y overlay GT.
- Botón `Info` con panel lateral explicativo sobre la página, el porcentaje, bandas y biomarcadores.
- Cierre de modal/drawer por botón, clic fuera y tecla `Escape`.
- Diseño tipo dashboard en escritorio, ajustado a una sola vista sin scroll principal.
- Diseño responsive para móvil, donde se permite scroll para preservar legibilidad.

## Requisitos

1. **API ejecutando:** `GLAUCOMA_API_MODE=demo uvicorn api.app:app --reload` para probar la interfaz, o `uvicorn api.app:app --reload` para el pipeline real.
2. **Navegador moderno:** Chrome, Firefox, Safari o Edge.
3. **CORS habilitado:** `api/app.py` incluye `CORSMiddleware` para permitir el frontend local.

## Uso

### Opción 1: Servir desde el backend

```bash
GLAUCOMA_API_MODE=demo uvicorn api.app:app --reload
# Abrir http://127.0.0.1:8000/
```

### Opción 2: Abrir directamente

```bash
open frontend/index.html
```

### Opción 3: Servir localmente

```bash
cd frontend/
python -m http.server 8001
# Abrir http://127.0.0.1:8001
```

## Flujo

1. La página consulta `/health` y activa el botón si la API responde.
2. `Caso aleatorio` llama a `/api/test-cases/random`.
3. La pantalla principal muestra resultado, aviso e imágenes.
4. `Más` abre los datos técnicos del caso.
5. `Info` abre la explicación clínica y de biomarcadores.

## Personalización

La URL de la API se calcula en `frontend/index.html`: si la página se sirve desde
FastAPI usa el mismo origen; si se abre como archivo usa `http://127.0.0.1:8000`.

```javascript
const API_BASE = window.location.protocol === "file:"
    ? "http://127.0.0.1:8000"
    : window.location.origin;
```

Los textos clínicos, bandas y definiciones de biomarcadores están inline en el mismo archivo para mantener el frontend autocontenido.
