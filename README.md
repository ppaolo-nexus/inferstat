# InferStat U — Estadística Inferencial interactiva

Aplicación web para cursos universitarios de Estadística Inferencial: muestreo y
pruebas de hipótesis con gráfico interactivo, desarrollo paso a paso en LaTeX y
conclusión académica automática.

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q            # pruebas unitarias del núcleo matemático
```

## Arquitectura

```
inferstat/
├── app.py                  # Configuración global + registro/enrutamiento de páginas
├── core/                   # CAPA MATEMÁTICA (sin Streamlit, 100 % testeable)
│   ├── common.py           #   Step, formato, distribuciones, críticos y p-valor
│   ├── sampling.py         #   Tamaño muestral, estratificado, sistemático
│   ├── hypothesis.py       #   Pruebas Z, t, proporciones, χ² + TEST_REGISTRY
│   └── data_loader.py      #   Lectura CSV/Excel y estadísticos resumidos
├── viz/
│   └── charts.py           # CAPA GRÁFICA (Plotly): campana, regiones, marcador
├── ui/                     # CAPA DE PRESENTACIÓN (Streamlit)
│   ├── components.py       #   Entradas comunes, veredicto, pasos, navegación
│   ├── help_texts.py       #   Todos los tooltips en un solo lugar
│   ├── page_wizard.py      #   Asistente «¿Qué deseas calcular?»
│   ├── page_sampling.py    #   Módulo 1
│   └── page_hypothesis.py  #   Módulo 2
├── tests/test_core.py      # Validación contra SciPy
└── sample_data/            # CSV y Excel de ejemplo
```

Flujo de datos: `ui` recoge entradas → `core` devuelve un objeto resultado
(`TestResult`, `SampleSizeResult`, …) con estadístico, p-valor, críticos y pasos
LaTeX → `viz` lo dibuja → `ui` lo muestra. Ninguna capa inferior importa a una superior.

## Cómo añadir ANOVA, no paramétricas o regresión

1. **Núcleo**: crea `core/anova.py`. Si usa una distribución nueva (F), añádela
   en `common.get_dist`; `critical_values` y `p_value` funcionan sin cambios para
   pruebas de cola derecha.
2. **Resultado**: reutiliza `TestResult` (ya incluye `steps`, `notes`, `conclusion`)
   o crea un dataclass propio si necesitas tablas (p. ej. tabla ANOVA).
3. **Gráfico**: `plot_hypothesis_test` sirve para cualquier `TestResult` cuya
   distribución exista en `get_dist`.
4. **Interfaz**: crea `ui/page_anova.py` con `render()` y regístrala en `PAGES` (app.py).
5. **Pruebas**: compara contra `scipy.stats.f_oneway`, `mannwhitneyu`, `linregress`, etc.

## Archivos de datos
Una columna por muestra con encabezado. Separador `,` `;` o tabulador (se detecta
automáticamente). Para proporciones, la columna puede contener categorías
(«Si/No») y se elige qué valor cuenta como éxito.
