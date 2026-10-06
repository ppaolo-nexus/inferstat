"""
ui/help_texts.py
================
Textos de los tooltips informativos. Centralizarlos permite revisarlos,
traducirlos o ampliarlos sin tocar la lógica de la interfaz.
"""

HELP = {
    # ---------------- Muestreo ----------------
    "confidence": (
        "**Nivel de confianza (1 − α)**: proporción de intervalos que, construidos con este "
        "método en muestras repetidas, contendrían al parámetro verdadero. A mayor confianza, "
        "mayor Z y mayor tamaño de muestra."),
    "e_mean": (
        "**¿Qué es el margen de error e?** Es la máxima diferencia que aceptas entre la media "
        "muestral y la media poblacional verdadera. Se expresa en las MISMAS unidades de la "
        "variable (p. ej. e = 2 kg). Reducir e a la mitad cuadruplica n."),
    "e_prop": (
        "**¿Qué es el margen de error e?** Diferencia máxima tolerada entre la proporción "
        "muestral y la verdadera, en tanto por uno: 0.05 equivale a ±5 puntos porcentuales."),
    "sigma": (
        "**Desviación estándar poblacional σ**: si no la conoces, usa la de un estudio previo, "
        "una prueba piloto, o la aproximación σ ≈ Rango / 4."),
    "p_prop": (
        "**Proporción esperada p**: estimación previa de la proporción de éxitos. Si no tienes "
        "información, usa p = 0.5 (máxima variabilidad → muestra más grande y segura)."),
    "population": (
        "Usa **población finita** cuando conoces el tamaño total N y la muestra representa una "
        "fracción apreciable (regla práctica: n/N > 5 %). La corrección reduce n."),
    "N": "**N**: número total de elementos de la población (marco muestral).",
    "n_strat": "Tamaño total de la muestra que se repartirá entre los estratos.",
    "strata": (
        "Escribe el nombre y el tamaño N_i de cada estrato. Agrega o elimina filas con los "
        "controles de la tabla. En afijación proporcional, cada estrato recibe una parte de la "
        "muestra igual a su peso W_i = N_i / N."),
    "start": (
        "**Arranque A**: primer elemento seleccionado, elegido al azar entre 1 y k. Fijar una "
        "semilla hace que el sorteo sea reproducible (útil para informes)."),
    "seed": "Con la misma semilla obtendrás siempre el mismo arranque aleatorio.",

    # ---------------- Pruebas de hipótesis ----------------
    "test": (
        "**¿Cuándo usar t-Student vs Z?**\n\n"
        "• **Z**: σ poblacional conocida, o muestra grande (n ≥ 30) donde s aproxima bien a σ.\n\n"
        "• **t-Student**: σ desconocida y muestra pequeña (n < 30), con población "
        "aproximadamente normal. Con n grande, t y Z casi coinciden.\n\n"
        "• **χ²**: para contrastar una varianza.\n\n"
        "¿Dudas? Usa el Asistente de selección."),
    "tail": (
        "• **Bilateral**: H₁ dice «es diferente» (≠).\n\n"
        "• **Unilateral izquierda**: H₁ dice «es menor» (<).\n\n"
        "• **Unilateral derecha**: H₁ dice «es mayor» (>).\n\n"
        "Elige la cola según la pregunta de investigación ANTES de mirar los datos."),
    "alpha": (
        "**Nivel de significancia α**: probabilidad de rechazar H₀ cuando es verdadera "
        "(error tipo I). Valores habituales: 0.05 (ciencias sociales), 0.01 (más exigente)."),
    "data_mode": (
        "**Datos resumidos**: ingresas x̄, s y n ya calculados.\n\n"
        "**Archivo**: subes un CSV/Excel con los datos crudos (una columna por muestra) y la app "
        "calcula los resúmenes automáticamente."),
    "xbar": "**Media muestral x̄**: promedio de los datos observados en la muestra.",
    "s": "**Desviación estándar muestral s** (con denominador n − 1).",
    "sigma_pop": "**σ poblacional**: desviación estándar conocida de la población (dato del problema).",
    "sigma_known": (
        "Marca esta casilla si el enunciado proporciona σ de la POBLACIÓN. Si no la conoces, "
        "la prueba Z solo es válida como aproximación con n ≥ 30."),
    "n": "**Tamaño de muestra n**: número de observaciones.",
    "mu0": "**Valor hipotético μ₀**: valor de la media que afirma la hipótesis nula.",
    "d0": "**Diferencia hipotética (μ₁ − μ₂)₀**: normalmente 0 (las medias son iguales).",
    "equal_var": (
        "**Varianzas iguales**: usa la varianza combinada (pooled). Si las desviaciones difieren "
        "mucho (razón mayor que 2), desmarca para usar la corrección de Welch."),
    "x_success": "**Éxitos x**: número de elementos de la muestra que tienen la característica.",
    "p0": "**Proporción hipotética p₀**: valor de la proporción que afirma H₀.",
    "sigma0_sq": "**Varianza hipotética σ₀²**: varianza poblacional que afirma H₀ (¡varianza, no desviación!).",
    "success_value": "Valor de la columna que se contará como «éxito» (p. ej. «Sí», 1, «Aprobado»).",
}
