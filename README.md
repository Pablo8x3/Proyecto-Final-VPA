# Proyecto de Detección de Zonas en Trenes - YOLOv8

**Proyecto de Visión Artificial para Detección de Cabinas, Salones y Otras Zonas en Planos de Trenes usando YOLOv8**

## 📋 Descripción General

Este proyecto implementa un sistema completo de detección de objetos basado en YOLOv8 para identificar y analizar diferentes zonas (cabinas, salones, vestíbulos, aseos, etc.) en imágenes de planos de trenes. Incluye herramientas para:

- **Entrenamiento** de modelos YOLOv8 con división automática de datos
- **Aumento de datos** (data augmentation con flip horizontal)
- **Anotación y edición** de bounding boxes (etiquetado manual)
- **Análisis de resultados** con métricas IoU detalladas
- **Inferencia** con visualización de predicciones y análisis de sensores

## 🚀 Requisitos Previos

### Sistema Operativo
- Linux (Ubuntu 18.04+, Debian, etc.)
- macOS (versión reciente)
- Windows (con WSL2 recomendado)

### Requisitos del Sistema
- **Python 3.8+** (recomendado 3.10 o superior)
- **pip** (gestor de paquetes de Python)
- **GPU NVIDIA** (recomendado para entrenamiento rápido, aunque es opcional)
  - CUDA 11.8+ (si se va a usar GPU)
  - cuDNN 8.x (si se va a usar GPU)
- **Tesseract OCR** (para detección automática de escala)
- **OpenCV** (incluido en dependencias)

### Instalación de Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Descargar e instalar desde: https://github.com/UB-Mannheim/tesseract/wiki

## 📦 Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Pablo8x3/Proyecto-FInal-VPA.git
cd Proyecto-FInal-VPA
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno (Linux/macOS)
source venv/bin/activate

# Activar entorno (Windows)
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
# Asegurar que pip está actualizado
pip install --upgrade pip

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
pro_vision/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── prueba_numero_escala.py           # Script de prueba para detección de escala
├── yolov8m.pt                        # Modelo YOLOv8 mediano (descargable)
├── yolov8s.pt                        # Modelo YOLOv8 pequeño (descargable)
│
├── scripts/
│   ├── train_yolov8.py              # Entrenamiento de modelo con IoU
│   ├── flip_images.py               # Aumento de datos (flip horizontal)
│   ├── etiquetador_bbox.py          # Herramienta interactiva para etiquetar
│   ├── bbox_editor.py               # Editor de bounding boxes
│   ├── analizar_results.py          # Análisis de resultados con IoU
│   ├── resultados_bbox.py           # Visualización de predicciones
│   └── resultados_sensores.py       # Análisis avanzado con sensores
│
└── planos/
    ├── data_trenes.yaml             # Configuración del dataset
    ├── all_images/                  # Dataset completo
    │   ├── images/                  # Imágenes originales
    │   ├── labels/                  # Anotaciones YOLO (formato txt)
    │   ├── split/                   # Dataset dividido (train/val/test)
    │   ├── models/                  # Modelos entrenados
    │   │   ├── entrenamiento_1/     # Primer entrenamiento
    │   │   ├── entrenamiento_2/     # Segundo entrenamiento
    │   │   └── entrenamiento_3/     # Tercer entrenamiento (best.pt)
    │   └── comprobar_manual/        # Imágenes de prueba manual
    ├── train/                        # Datos de entrenamiento
    ├── val/                          # Datos de validación
    └── models/                       # Modelos adicionales
```

## 🎯 Guía de Uso

### 1. Preparación de Datos

#### A. Anotación de Imágenes (Si tienes imágenes nuevas)

```bash
# Etiquetador interactivo: dibuja bounding boxes manualmente
python scripts/etiquetador_bbox.py

# Editor de bounding boxes: corrige anotaciones existentes
python scripts/bbox_editor.py
```

**Clases disponibles:**
- cabina
- salon
- vestibulo
- wc_normal
- wc_pmr
- bufet
- fuelles
- anexo
- bicicletas
- personal
- corredor

#### B. Aumento de Datos

```bash
# Genera copias con flip horizontal de imágenes y etiquetas
python scripts/flip_images.py
```

### 2. Entrenamiento del Modelo

```bash
# Entrena YOLOv8 con división automática 70/20/10
python scripts/train_yolov8.py
```

**Qué hace este script:**
- Divide automáticamente el dataset en train (70%), val (20%), test (10%)
- Entrena el modelo YOLOv8
- Calcula métricas IoU en el conjunto test
- Guarda resultados en `planos/all_images/models/entrenamiento_N/`
- Genera reportes CSV y TXT con análisis detallado

**Salida esperada:**
- `entrenamiento_N/entrenamiento_N/weights/best.pt` - Mejor modelo
- `entrenamiento_N/iou_per_image.csv` - IoU por imagen
- `entrenamiento_N/iou_per_class.csv` - IoU por clase
- `entrenamiento_N/iou_summary.txt` - Resumen textual

### 3. Análisis de Resultados

```bash
# Análisis detallado de un modelo entrenado
python scripts/analizar_results.py
```

**Genera:**
- Métricas IoU por imagen y por clase
- Resumen JSON con estadísticas
- Reportes en CSV

### 4. Inferencia y Visualización

#### A. Visualización de Bounding Boxes

```bash
# Prueba el modelo en imágenes y visualiza predicciones con colores
python scripts/resultados_bbox.py
```

#### B. Análisis Avanzado con Sensores

```bash
# Detección automática de escala + análisis de zonas + sensores
# Genera PDF multipágina con resultados
python scripts/resultados_sensores.py
```

**Características:**
- Detección automática de línea de cota (escala)
- Fallback manual con clics si falla OCR
- Posicionamiento geométrico de sensores
- Salida en PDF multipágina

### 5. Prueba de Escala (Test)

```bash
# Script de prueba para detección de escala y OCR
python prueba_numero_escala.py
```

## ⚙️ Configuración

### Ajuste de Rutas (si es necesario)

Si tu estructura de carpetas es diferente, edita las siguientes variables en cada script:

**En `train_yolov8.py`:**
```python
DATASET_IMAGES = "ruta/a/tus/imagenes"
DATASET_LABELS = "ruta/a/tus/etiquetas"
PROJECT_RESULTS_BASE = "ruta/donde/guardar/modelos"
```

**En `resultados_bbox.py`:**
```python
model_path = "ruta/al/modelo/best.pt"
img_folder = "ruta/a/imagenes/test"
output_folder = "ruta/donde/guardar/resultados"
```

**En `resultados_sensores.py`:**
```python
MODEL_PATH = "ruta/al/modelo/best.pt"
IMG_FOLDER = "ruta/a/imagenes/test"
OUTPUT_FOLDER = "ruta/donde/guardar/pdfs"
```

### Parámetros de Entrenamiento

En `train_yolov8.py` puedes ajustar:

```python
EPOCHS = 50              # Número de épocas
IMG_SIZE = 640          # Tamaño de imagen (640, 960, etc.)
BATCH_SIZE = 16         # Tamaño del batch
PATIENCE = 10           # Early stopping patience
DEVICE = 0              # GPU ID (0 para primera GPU, -1 para CPU)
```

### Parámetros de Detección de Escala

En `resultados_sensores.py`:

```python
TOP_CROP_RATIO = 0.20   # % de imagen para buscar línea (arriba)
HSV_MIN = np.array([35, 20, 40])      # Rango HSV mínimo (verde)
HSV_MAX = np.array([95, 200, 255])    # Rango HSV máximo
HOUGH_THRESHOLD = 50    # Sensibilidad de detección de líneas
```

## 📊 Formato de Datos

### Formato YOLO de Anotaciones

Las etiquetas están en formato YOLO (un archivo `.txt` por imagen):

```
<class_id> <x_center> <y_center> <width> <height>
```

Donde:
- `class_id`: ID de la clase (0-10)
- Todas las coordenadas están **normalizadas a [0, 1]** (relativas al tamaño de la imagen)

**Ejemplo:**
```
0 0.5 0.3 0.2 0.1
5 0.7 0.6 0.15 0.25
```

### Configuración YAML

El archivo `data_trenes.yaml` define el dataset:

```yaml
path: /absolute/path/to/planos/split
train: images  # relativo a path
val: images    # relativo a path
test: images   # relativo a path

nc: 11  # número de clases
names:  # nombres de clases
  0: cabina
  1: salon
  2: vestibulo
  3: wc_normal
  4: bufet
  5: fuelles
  6: anexo
  7: bicicletas
  8: personal
  9: wc_pmr
  10: corredor
```

## 🔧 Solución de Problemas

### "ModuleNotFoundError: No module named 'ultralytics'"

```bash
# Reinstala las dependencias
pip install -r requirements.txt
# O directamente
pip install ultralytics opencv-python pytorch
```

### "CUDA out of memory"

Si tienes GPU pero se queda sin memoria:

```python
# En train_yolov8.py, reduce el batch size
BATCH_SIZE = 8  # cambiar de 16 a 8 o menor
```

O usa CPU:
```python
DEVICE = -1  # usar CPU en lugar de GPU
```

### Tesseract no encontrado

```bash
# Linux
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Después, en Python:
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

### GPU no detectada

```bash
# Verifica que CUDA esté instalado
nvidia-smi

# Reinstala PyTorch con CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Métricas y Evaluación

El proyecto genera varios tipos de métricas:

### IoU (Intersection over Union)
- **Por imagen**: Precisión del modelo en cada imagen
- **Por clase**: Rendimiento del modelo para cada tipo de zona
- **Global**: Métrica agregada de todo el conjunto

### Archivos de Salida

1. **iou_per_image.csv**: Una fila por imagen con IoU
2. **iou_per_class.csv**: Una fila por clase con estadísticas
3. **iou_summary.json**: Resumen estructurado en JSON
4. **iou_summary.txt**: Resumen legible para humanos

## 🎓 Ejemplo Completo

```bash
# 1. Crear entorno
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Aumentar datos (opcional)
python scripts/flip_images.py

# 4. Entrenar
python scripts/train_yolov8.py

# 5. Analizar resultados
python scripts/analizar_results.py

# 6. Visualizar predicciones
python scripts/resultados_bbox.py

# 7. Análisis avanzado con sensores
python scripts/resultados_sensores.py
```

## 📝 Notas Importantes

- **Rutas absolutas vs relativas**: Los scripts usan rutas relativas basadas en el directorio `planos/`. Asegúrate de ejecutar los scripts desde la raíz del proyecto.
- **GPU opcional**: El entrenamiento es más rápido con GPU, pero funciona con CPU.
- **Espacio en disco**: Asegúrate de tener suficiente espacio (mínimo 5-10 GB para modelos y resultados).
- **Tiempo de entrenamiento**: El primer entrenamiento puede tardar 30 minutos a varias horas dependiendo de hardware.

## 👥 Información de Contacto

Proyecto desarrollado por: **Pablo8x3**

Repositorio: https://github.com/Pablo8x3/Proyecto-FInal-VPA

## 📄 Licencia

Este proyecto está bajo licencia (especificar licencia si aplica).

## 🔗 Recursos Útiles

- [YOLOv8 Documentación](https://docs.ultralytics.com/)
- [OpenCV Documentación](https://docs.opencv.org/)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [PyTorch](https://pytorch.org/)

---

**Última actualización:** Diciembre 2025
