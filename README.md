# ForecastingVentas

Proyecto de Machine Learning para pronosticar ventas.

## Estructura

- `data/raw/`: datos originales, sin modificar.
- `data/processed/`: datos limpios y transformados.
- `notebooks/`: exploracion, preparacion, entrenamiento y evaluacion.
- `src/`: codigo reutilizable del proyecto.
- `models/`: modelos entrenados y artefactos generados.
- `app/`: aplicacion interactiva en Streamlit.
- `docs/`: documentacion tecnica y decisiones del proyecto.
- `tests/`: pruebas automatizadas.

## Inicio rapido

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app\app.py
```

Coloca los archivos de entrada en `data/raw/`. Los datos y modelos generados se mantienen fuera del repositorio mediante `.gitignore`.
