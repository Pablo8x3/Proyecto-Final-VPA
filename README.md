# Proyecto de Detección de Zonas en Trenes - YOLOv8

**Sistema de Visión Artificial para Detección de Zonas (Cabinas, Salones, etc.) en Planos de Trenes**

## 📋 Descripción

Este proyecto utiliza YOLOv8 para detectar y analizar diferentes zonas en imágenes de planos de trenes. Proporciona dos herramientas principales para usuarios finales:

1. **`main.py`** - Genera PDFs con predicciones de zonas y análisis de sensores
2. **`analizar_results.py`** - Evalúa el rendimiento del modelo con métricas IoU

## 🚀 Requisitos Previos

### Instalación Base

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip tesseract-ocr
```

**macOS:**
```bash
brew install python3 tesseract
```

**Windows:** Descargar Python desde https://www.python.org/ e instalar Tesseract desde https://github.com/UB-Mannheim/tesseract/wiki

### Requisitos del Sistema
- Python 3.8+
- pip
- Tesseract OCR (para detección automática de escala)

## 📦 Instalación Rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/Pablo8x3/Proyecto-FInal-VPA.git
cd Proyecto-FInal-VPA

# 2. Crear entorno virtual
python3 -m venv venv
# Linux/macOS
source venv/bin/activate          
# o en Windows:
venv\Scripts\activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

## 🎯 Uso

### 1. Análisis de Imágenes con main.py

Genera un PDF multipágina con predicciones de zonas (cabinas, salones, etc.) y sensores.

```bash
python scripts_uso/main.py
```

**Qué hace:**
- Carga automáticamente el modelo entrenado
- Detecta zonas en las imágenes usando YOLOv8
- Detecta automáticamente la escala de la imagen (OCR)
- Posiciona sensores de temperatura y humedad
- Genera PDF con visualización de resultados

**Salida:**
- PDFs en la carpeta `outputs/` (una por imagen analizada)

**Si necesitas cambiar las imágenes de entrada**, edita en `main.py`:
```python
IMG_FOLDER = "planos/all_images/split/test/images/"  # Carpeta de imágenes
OUTPUT_FOLDER = "outputs"                            # Donde guardar PDFs
MODEL_PATH = "planos/all_images/models/entrenamiento_3/entrenamiento_3/weights/best.pt"  # Modelo
```

### 2. Evaluar Rendimiento del Modelo con analizar_results.py

Analiza qué tan bien detecta el modelo las zonas (calcula métrica IoU).

```bash
python scripts_uso/analizar_results.py
```

**Qué hace:**
- Carga el modelo entrenado
- Evalúa en conjuntos de entrenamiento, validación y prueba
- Calcula precisión IoU (Intersection over Union) por imagen y por clase
- Genera reportes detallados

**Salida:**
- `iou_per_image.csv` - Precisión por imagen
- `iou_per_class.csv` - Precisión por clase
- `iou_summary.json` - Resumen en JSON
- `iou_summary.txt` - Resumen legible

**Si necesitas cambiar los datos evaluados**, edita en `analizar_results.py`:
```python
MODEL_PT = Path("planos/models/yolo_trenes/weights/best.pt")  # Modelo a evaluar

# Datos a evaluar:
TRAIN_IMAGES = Path("planos/train/images")
VAL_IMAGES   = Path("planos/val/images")
TEST_IMAGES  = Path("planos/comprobar_manual/images/")

# Donde guardar resultados:
OUTPUT_DIR = Path("planos/models/yolo_trenes/results")
```

## 📁 Estructura de Datos Esperada

```
pro_vision/
├── README.md
├── requirements.txt
├── outputs/                         # PDFs generados por main.py
│
├── scripts_uso/
│   ├── main.py                      # Análisis con sensores (PDF)
│   └── analizar_results.py          # Evaluación del modelo (IoU)
│
└── planos/
    ├── all_images/
    │   ├── images/                  # Imágenes originales
    │   ├── labels/                  # Anotaciones (formato YOLO)
    │   ├── split/
    │   │   ├── train/images/
    │   │   ├── val/images/
    │   │   └── test/images/
    │   └── models/
    │       ├── entrenamiento_3/
    │       │   └── entrenamiento_3/
    │       │       └── weights/
    │       │           └── best.pt  # Modelo usado por main.py
    │       └── yolo_trenes/
    │           ├── weights/
    │           │   └── best.pt      # Modelo usado por analizar_results.py
    │           └── results/         # Salida de analizar_results.py
    │
    ├── train/images/
    ├── train/labels/
    ├── val/images/
    ├── val/labels/
    ├── comprobar_manual/
    │   ├── images/
    │   └── labels/
    │
    └── data_trenes.yaml             # Config del dataset
```

## 🔧 Problemas Comunes

### Error: "ModuleNotFoundError: No module named 'ultralytics'"

```bash
# Reinstala dependencias
pip install -r requirements.txt
```

### Error: "Tesseract is not installed"

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Descarga e instala desde: https://github.com/UB-Mannheim/tesseract/wiki

### El PDF sale vacío o sin sensores

1. Verifica que el archivo `best.pt` existe en la ruta especificada
2. Revisa que las imágenes están en la carpeta `IMG_FOLDER`
3. Si la OCR falla, el script te pedirá hacer clics manuales para definir la escala

### Error: "CUDA out of memory"

Si el script va lento, cambia el modelo a CPU en `main.py` (línea ~500):
```python
model = YOLO(MODEL_PATH)
model.to('cpu')  # Usar CPU en lugar de GPU
```

## 📊 Clases Detectadas

El modelo detecta 11 tipos de zonas en trenes:

1. **Cabina** - Cabina de conducción
2. **Salón** - Área principal de pasajeros
3. **Vestíbulo** - Entrada/Pasillo
4. **WC Normal** - Aseo estándar
5. **WC PMR** - Aseo para personas con movilidad reducida
6. **Búfet** - Área de servicio de alimentos
7. **Fuelles** - Fuelles entre vagones
8. **Anexo** - Áreas anexas
9. **Bicicletas** - Área de bicicletas
10. **Personal** - Área de personal
11. **Corredor** - Pasillos

## 📈 Interpretación de Resultados

### Métrica IoU (Intersection over Union)

Mide qué tan preciso es el modelo al detectar zonas. Rango: 0 a 1 (o 0% a 100%)

- **IoU > 0.9**: Excelente detección
- **0.7 < IoU < 0.9**: Muy buena detección
- **0.5 < IoU < 0.7**: Buena detección
- **IoU < 0.5**: Detección deficiente

### Archivos de Salida de analizar_results.py

1. **iou_per_image.csv**
   ```
   image,train_iou,val_iou,test_iou
   10.jpg,0.95,0.92,0.88
   13.jpg,0.87,0.85,0.82
   ```

2. **iou_summary.txt**
   ```
   ========== RESUMEN GLOBAL ==========
   IoU TRAIN: 0.91 ± 0.05
   IoU VAL:   0.89 ± 0.07
   IoU TEST:  0.85 ± 0.10
   ```

## ❓ Preguntas Frecuentes

**P: ¿Puedo usar mis propias imágenes?**
R: Sí, coloca las imágenes en `IMG_FOLDER` en `main.py` y ejecuta el script.

**P: ¿Qué pasa si la escala OCR falla?**
R: El script te pedirá hacer dos clics en la imagen para marcar la línea de escala manualmente.

**P: ¿Necesito GPU para ejecutar esto?**
R: No, pero es más rápido con GPU. Para CPU, el tiempo de análisis es de 10-30 segundos por imagen.

**P: ¿Dónde encuentro los resultados?**
R: Los PDFs se guardan en `outputs/`, los reportes de IoU en la carpeta especificada en `OUTPUT_DIR`.

## 📝 Notas

- Ejecuta los scripts desde la carpeta raíz del proyecto
- Las imágenes deben ser en formato JPG, PNG, BMP, etc.
- El modelo requiere imágenes de planos de trenes para precisión óptima

## 👤 Información de Contacto

**Desarrollado por:** Pablo8x3

**Repositorio:** https://github.com/Pablo8x3/Proyecto-FInal-VPA

---

**Última actualización:** Diciembre 2025
