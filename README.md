# GaugeViewer

Herramienta de línea de comandos para detectar el aro y la aguja de manómetros analógicos
utilizando visión clásica con OpenCV. El script `detect_gauge_center.py` procesa imágenes,
recorta la carátula, estima el ángulo de la aguja y genera visualizaciones para depurar el
pipeline.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido

Procesar los ejemplos incluidos y guardar las salidas en `output/` (comportamiento por defecto):

```bash
python detect_gauge_center.py
```

### Parámetros principales

```
usage: detect_gauge_center.py [-h] [--output OUTPUT] [--no-write]
                              [--calibration CALIBRATION]
                              [inputs ...]
```

- `inputs`: rutas individuales, directorios o patrones glob. Si se omiten se
  utilizan las imágenes del directorio `samples/`.
- `--output/-o`: define la carpeta donde se escribirán las imágenes anotadas,
  las rectificaciones y las máscaras de la aguja. Se crean los subdirectorios
  `annotated/`, `rectified/` y `mask/` si no existen.
- `--no-write`: evita generar archivos. Solo se imprimirán las lecturas por consola,
  útil para ejecutar pruebas sin ensuciar el repositorio.
- `--calibration/-c`: ruta a un archivo JSON con calibraciones para convertir
  ángulos en valores físicos. Se puede indicar múltiples veces para combinar
  tablas; las últimas entradas sobrescriben a las anteriores.

### Ejemplos

Procesar un directorio propio y guardar resultados en otra carpeta:

```bash
python detect_gauge_center.py ~/datasets/gauges --output ~/tmp/gauge_debug
```

Leer imágenes individuales sin generar archivos:

```bash
python detect_gauge_center.py samples/20.png samples/75.jpg --no-write
```

Agregar una tabla de calibraciones personalizada:

```bash
python detect_gauge_center.py --calibration configs/gauges.json
```

Formato esperado del archivo JSON de calibraciones:

```json
{
  "75.jpg": {
    "angle_min": 40.0,
    "angle_max": 320.0,
    "value_min": 0.0,
    "value_max": 160.0,
    "clockwise": true
  }
}
```

Cada clave corresponde al nombre del archivo procesado (sin ruta) y sus valores
permiten transformar el ángulo detectado en una lectura física a través de la
interpolación lineal descrita en el código.

## Cómo funciona

1. **Detección del dial**: se realza el contraste, se aplica un banco compacto de
   parámetros de Hough para localizar el aro y luego se refina con elipses
   ajustadas sobre contornos filtrados morfológicamente.
2. **Rectificación**: a partir de la elipse se construye una homografía que
   normaliza la carátula, recortando un patch cuadrado centrado en el gauge.
3. **Aguja**: se ensaya una cascada de detectores (LSD, Hough probabilístico y un
   perfil polar) que comparten una función de puntaje geométrico para validar la
   línea candidata.
4. **Calibración opcional**: si existe una entrada en la tabla de calibraciones
   para el nombre de la imagen, el ángulo se convierte en valor físico.
5. **Anotaciones y máscaras**: cuando no se usa `--no-write` el script genera las
   imágenes anotadas del original y la vista rectificada, además de una máscara
   binaria de la aguja.

## Datos de ejemplo

El directorio `samples/` contiene un conjunto reducido de manómetros para evaluar
el pipeline. Las salidas ejemplo se almacenan en `output/`.
