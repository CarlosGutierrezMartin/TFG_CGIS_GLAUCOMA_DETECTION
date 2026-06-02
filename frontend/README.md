# Frontend — Detección Temprana de Glaucoma

## Descripción

Interfaz web profesional (HTML5 + vanilla JavaScript) para la API REST de detección temprana de glaucoma.

**Características:**

- ✓ Interfaz sobria, estilo académico TFG.
- ✓ Consulta aleatoria de casos Test400 con un botón.
- ✓ Visualización en tiempo real de:
  - Imagen original, overlay de ground-truth, overlay de predicción.
  - Probabilidad calibrada en %.
  - Banda de riesgo (baja/intermedia/moderada).
  - Biomarcadores (vCDR, hCDR, rCDR, area_CDR, rim_to_disc, ISNT-like).
  - Etiqueta real vs predicción + acierto/fallo.
  - Métricas de calidad (entropía, anatomía válida).
  - Disclaimer clínico.
- ✓ Indicador de conexión a API (estado en tiempo real).
- ✓ Responsivo (mobile + desktop).
- ✓ Sin dependencias externas (CSS inline, JS vanilla).

## Requisitos

1. **API ejecutando:** `uvicorn api.app:app --reload` en http://127.0.0.1:8000
2. **Navegador moderno:** Chrome, Firefox, Safari, Edge (soporte ES6).
3. **CORS habilitado en API:** por defecto FastAPI permite CORS locales.

## Instalación y Uso

### Opción 1: Abrir directamente (recomendado)

```bash
# Abre el archivo HTML en el navegador
open frontend/index.html
# o desde navegador: Ctrl+O → seleccionar frontend/index.html
```

### Opción 2: Servir con servidor local (si CORS falla)

```bash
cd frontend/
python -m http.server 8001
# Abre http://127.0.0.1:8001
```

## Estructura

```
frontend/
  index.html       # Interfaz completa (HTML5 + CSS inline + JS vanilla)
  README.md        # Este archivo
```

## Flujo de Uso

1. **Página carga:** conecta automáticamente a `/health` de la API.
   - Si exitoso: muestra modelos cargados, casos Test400, umbral.
   - Si falla: muestra error de conexión, botón deshabilitado.

2. **Haz clic en "Caso Aleatorio":**
   - Spinner de carga.
   - API devuelve caso aleatorio con predicción, overlays, biomarcadores.

3. **Visualiza resultados:**
   - 3 imágenes lado a lado (original, GT, predicción).
   - Tabla de biomarcadores (vCDR predicho vs GT, diferencia).
   - Banda de riesgo coloreada (verde=bajo, amarillo=intermedio, rojo=moderado).
   - Caja de decisión (umbral, score, acierto sí/no).
   - Métricas de calidad (entropía, anatomía válida).

4. **Información de estado:**
   - Panel izquierdo muestra todos los valores extracción:
     - Nombre de imagen.
     - Probabilidad % y banda de riesgo.
     - Predicción vs etiqueta real.
     - Validez de disco/copa.

## Personalización

### Colores (Risk Bands)

En `<style>`:

```css
.risk-band.low {
    background: #d5f4e6;      /* Verde suave */
    color: #27ae60;
    border-left-color: #27ae60;
}

.risk-band.intermediate {
    background: #fff3cd;      /* Amarillo suave */
    color: #856404;
    border-left-color: #ffc107;
}

.risk-band.moderate {
    background: #f8d7da;      /* Rojo suave */
    color: #721c24;
    border-left-color: #dc3545;
}
```

### URL de API

Si la API está en otro puerto o servidor, cambia `API_BASE`:

```javascript
const API_BASE = "http://127.0.0.1:8000";  // Cambiar aquí
```

### Textos y Mensajes

Todos los textos son en línea en el HTML. Para traducir o cambiar mensajes, busca strings en el archivo
(ej., "Caso Aleatorio", "Hallazgos intermedios...").

## Problemas Comunes

### "No se puede conectar a la API"

- Verifica que `uvicorn api.app:app --reload` está ejecutando en http://127.0.0.1:8000
- Si frontend está en otro servidor (no localhost), la API debe tener CORS configurado:

  ```python
  # En api/app.py, asegurate de:
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
  ```

### Las imágenes no se muestran

- Verifica que los overlays base64 se generan correctamente (decodifica manualmente en console).
- Navega a DevTools (F12) → Network → verifica respuesta de `/api/test-cases/random`.

### CORS error

- Si ves `Access-Control-Allow-Origin` en console, la API no permite cross-origin requests.
- Asegurate que FastAPI tiene CORSMiddleware habilitado.
- Alternativa: sirve frontend y API en mismo localhost (ej., ambos en 127.0.0.1, diferentes puertos).

## Notas Técnicas

### Sin Frameworks Externos

- **HTML5:** semántica moderna, viewport meta, charset UTF-8.
- **CSS:** grid/flexbox, media queries, animations (@keyframes).
- **JavaScript:** vanilla ES6, fetch API, event listeners.

**Ventajas:**
- Sin dependencias de npm, webpack, bundlers.
- Carga instantánea (< 50 KB total).
- Fácil de mantener, sin breaking changes de librerías.

### Accesibilidad

- Contraste de colores: WCAG AA (4.5:1 ratio).
- Textos descriptivos para controles.
- Estructura semántica (header, section, footer).

### Responsive Design

- Breakpoint principal: 1024px (columna única en mobile).
- Grid 3 columnas en desktop, 1 en mobile.
- Fuentes y espaciados escalables.

---

**Desarrollado como parte del TFG de Detección Temprana de Glaucoma.**
