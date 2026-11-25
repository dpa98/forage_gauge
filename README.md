# ForageGauge

Estimación de biomasa de pasto seco —fracciones verde y muerta, en g/m²— a partir de una única fotografía cenital de pastura mediante un modelo de visión por computador basado en DINOv3 con adaptación eficiente de parámetros.

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

## 1. Alcance del repositorio

Este repositorio contiene la versión open-source anterior del proyecto. Las métricas reportadas en la sección 6 provienen de una ejecución interna (v1, informe técnico del 15/11/2026) realizada sobre datos y pesos no distribuidos públicamente; los comandos de esta versión no las reproducen exactamente.

## 2. Descripción

La cuantificación de la biomasa de pastura mediante muestreo de campo es laboriosa, destructiva y difícil de escalar. Este repositorio aborda dicho problema mediante regresión directa sobre imágenes RGB tomadas desde arriba, con el fin de predecir la materia verde seca y la materia muerta seca por metro cuadrado, así como su suma.

El conjunto de datos consta de 80 imágenes etiquetadas: 72 de entrenamiento, registradas en `data/raw/train_wide.csv` y almacenadas en `data/train/`, y 8 independientes de test, registradas en `data/test/test_wide.csv` y almacenadas en `data/test/`, que se mantuvieron separadas para la verificación del modelo. Según el informe de la ejecución interna, los objetivos se derivan del peso húmedo, la materia seca y la proporción verde/muerta; las medias son 123 ± 85 g/m² para verde (rango 23–426), 94 ± 68 g/m² para muerta (5–270) y 216,7 g/m² para el total. La validación se organiza sobre las 72 imágenes de entrenamiento en tres particiones estratificadas por intervalos de igual anchura sobre las dos variables objetivo (entre el 29 % y el 36 % de validación por fold), con semilla fija 16999.

El modelo emplea el backbone `facebook/dinov3-vitl16-pretrain-lvd1689m`, adaptado con LoRA/DoRA sobre las proyecciones `q`, `k`, `v` y `o` (rango 32, alfa 64, dropout 0,01). Sobre los tokens de salida se aplican dos bloques `LocalMambaBlock`, un promedio global y dos cabezas de regresión con activación positiva. La optimización utiliza pérdida de Tweedie con p = 1,2 y pesos 0,4 y 0,6; normalización de etiquetas según el máximo del conjunto de entrenamiento de cada fold multiplicado por 1,1; precisión mixta; media móvil exponencial del modelo; calentamiento inicial; reducción de la tasa de aprendizaje por meseta y parada temprana.

Según el informe técnico de la versión interna, el backbone se mantiene congelado y se adapta con LoRA/DoRA (r = 32, alfa = 64) en `q`, `k`, `v` y `o`, con 314,8 M parámetros totales y 11,65 M entrenables (3,7 %). La cabeza de esa versión consiste en dos bloques convolucionales ligeros con compuerta (LayerNorm → compuerta sigmoide → convolución depthwise k = 5 → proyección → residual), pooling global y dos MLPs con activación positiva (ELU + 1.001); es decir, difiere de los bloques `LocalMambaBlock` del código público. La optimización reportada emplea AdamW con tasa de aprendizaje 1e-4 en el backbone y 1e-3 en las cabezas, weight decay 1e-3, grad clipping 1.0, batch 2 con acumulación 4 (batch efectivo 8), AMP fp16, EMA 0,99 y gradient checkpointing; incluye warmup de 5 épocas, `ReduceLROnPlateau` sobre BR2 (factor 0,1, paciencia 10), un máximo de 100 épocas y parada temprana con paciencia 20. El checkpoint seleccionado es el EMA de la mejor época según BR2.

El preprocesamiento reportado redimensiona las imágenes a 1024×1024: en entrenamiento aplica volteos horizontales y verticales, rotación de 90°, rotación ±5° y color jitter, con normalización ImageNet; en validación solo redimensionado y normalización.

Ejemplo de imagen de entrada utilizada por el modelo:

![Fotografía cenital de pastura utilizada como entrada del modelo.](docs/figures/IMG_20260811_155344_n.jpg)

La selección del checkpoint se realiza con BR2, definido como la media de los R² por objetivo menos 0,1 veces su desviación típica; GR2 corresponde al R² global ponderado. El aporte del código consiste en integrar en un flujo reproducible la construcción de folds, el entrenamiento validado, la inferencia con test-time augmentation de cuatro volteos y el promedio de los pesos `best_ema.pt` de todos los folds, además del registro de métricas por intervalo de peso.

## 3. Publicación

No se dispone de publicación asociada a este repositorio. En consecuencia, no se incluyen cita, DOI ni enlaces oficiales.

## 4. Stack / herramientas

- Python
- PyTorch
- Hugging Face Transformers (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
- PEFT (LoRA/DoRA)
- timm (`ModelEmaV2`)
- Albumentations y OpenCV
- iterstrat (`MultilabelStratifiedKFold`)
- pandas y NumPy
- tqdm
- Matplotlib para la visualización de las curvas de entrenamiento

## 5. Qué hace el código

Todos los comandos se ejecutan desde la raíz del repositorio; los módulos se importan de forma plana, no como paquete.

- `python download_weights.py [HF_TOKEN]` descarga el backbone desde Hugging Face y lo almacena en `backbone_weights/`. Si ya existe `backbone_weights/config.json`, el modelo se carga desde esa ruta local.
- `python create_folds.py --input data/raw/train_wide.csv --output_dir ./data/folds/folds_v1 --n_splits 3` genera los archivos `fold_*.csv` con columnas `id`, `Dry_Green_g`, `Dry_Dead_g` y `split`. El valor por defecto de `--n_splits` es 5; los resultados reportados corresponden a 3 folds. También acepta `--n_bins` y `--seed`.
- `python train.py --version v1 --epochs 100 --batch_size 2 --img_size 1024` entrena todos los folds. Las opciones adicionales son `--folds_dir`, `--only_fold`, `--seed` y `--deterministic`. Los resultados se escriben en `checkpoints/<version>/fold_<id>/` (`best_ema.pt`, `final_ema.pt`, `last_checkpoint.pth`) y en `logs/<version>/fold_<id>/` (`training.log`, `metrics.csv`).
- `python predict.py --csv data/test/test_wide.csv --images_dir data/test --weights_dir checkpoints/v1 --output predictions.csv` genera predicciones promediando los cuatro volteos y todos los pesos `best_ema.pt` encontrados. El CSV de entrada debe contener una columna `id`. Las opciones adicionales son `--batch_size`, `--img_size` y `--token`. La salida contiene `id`, `Dry_Green_g`, `Dry_Dead_g` y `Dry_Total_g`.
- `python generar_diagramas.py` genera un diagrama estático de la arquitectura; no interviene en el entrenamiento ni en la inferencia.

El script `scripts/plot_metrics.py`, citado en versiones anteriores de este README, no está incluido en la versión pública. Las curvas de la sección 6 se entregan ya generadas en `docs/figures/Figure_1.png`.

## 6. Resultados

La figura siguiente presenta las curvas del fold 0. El panel superior muestra la pérdida de entrenamiento y las métricas BR2/GR2 por época; los paneles inferiores muestran, respectivamente, el R² y el MAE por objetivo. La línea vertical discontinua señala la época con el BR2 máximo.

![Curvas de entrenamiento del fold 0: pérdida de entrenamiento y BR2/GR2 por época (panel superior), R² por objetivo (inferior izquierdo) y MAE por objetivo (inferior derecho); la línea vertical marca la mejor época según BR2.](docs/figures/Figure_1.png)

Las métricas siguientes proceden del informe técnico de la ejecución interna (15/11/2026) y se calculan sobre imágenes de validación no vistas durante el entrenamiento. La figura ilustra las curvas del fold 0 de esa ejecución.

| Fold | Mejor época (parada) | BR2 | GR2 | MAE Total (g/m²) | RMSE Total (g/m²) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 55 (75) | 0,679 | 0,767 | 49,6 | 60,9 |
| 1 | 100 (sin parada) | 0,729 | 0,865 | 33,2 | 43,6 |
| 2 | 86 (100) | 0,632 | 0,703 | 56,1 | 74,8 |

| Métrica (3 folds) | Media ± desviación |
| --- | ---: |
| BR2 | 0,680 ± 0,048 |
| GR2 | 0,778 ± 0,082 |
| R² verde | 0,727 ± 0,175 |
| R² muerta | 0,621 ± 0,089 |
| R² total | 0,721 ± 0,101 |
| MAE verde (g/m²) | 30,2 ± 9,2 |
| MAE muerta (g/m²) | 28,2 ± 2,8 |
| MAE total (g/m²) | 46,3 ± 11,8 |
| RMSE total (g/m²) | 59,7 ± 15,6 |
| RMSLE total | 0,266 ± 0,043 |

El entorno fue Kaggle con GPU T4 ×2 (el script entrena en una sola GPU) y 16 GB de RAM; el entrenamiento de los 3 folds tardó 9 h 39 m 10 s. Los promedios del conjunto de datos empleados en el título de la figura son 123,1 g para Green, 93,6 g para Dead y 216,7 g para Total. El MAE total equivale a aproximadamente el 21 % de la biomasa media; la materia muerta es el objetivo más variable (R² 0,52–0,69) y el error absoluto crece con la biomasa. Con 80 imágenes, estos resultados son provisionales.

## 7. Limitaciones y próximos pasos

**Limitaciones**

1. N = 80: alta varianza entre folds (BR2 0,63–0,73; R² verde 0,54–0,88). Los resultados no son concluyentes.
2. Resolución y coste: 1024×1024 es adecuada para la textura del pasto, pero poco realista para procesar miles de imágenes en producción; conviene evaluar 512×512 o 768×768.
3. Altura del pasto: es una variable que el modelo no conoce y solo puede inferir referencias visuales; puede ser determinante cuando el pasto no deja ver el suelo.
4. Clima: el modelo no fue entrenado con heladas, escarcha, varios días sin llover ni lluvia reciente. Aprende a partir de color, textura, densidad y sombras, por lo que es crítico evaluar esas condiciones.
5. Drones: quedan por validar el volumen de datos a procesar, la calidad y resolución de las imágenes y el hardware necesario.

**Próximos pasos**

1. Ampliar el dataset a 800–1000 imágenes e incluir potreros con distintas condiciones de humedad.
2. Reducir la resolución a 768×768 o 512×512 y medir el impacto en precisión y latencia.
3. Evaluar imágenes con distintas alturas y sin suelo visible antes de recolectar datos de altura.
4. Analizar la robustez ante lluvia, heladas y periodos secos: comprobar si la misma coloración implica la misma densidad de materia seca.
5. Validar la calidad y representatividad de los datos antes de escalar a imágenes de dron.

## 8. Instalación y uso

Se requiere Python con `pip`. El entrenamiento y la inferencia funcionan en CPU, aunque el uso de GPU CUDA permite activar precisión mixta. Para cargar el backbone desde Hugging Face es necesario aceptar la licencia del modelo y definir `HF_TOKEN`; este paso no es necesario si `backbone_weights/` ya contiene el modelo. No se incluye `requirements.txt`: `install_deps.py` instala las dependencias principales y el resto se añade según los scripts que se vayan a ejecutar.

```bash
python install_deps.py
pip install numpy tqdm iterstrat matplotlib
python download_weights.py
```

Flujo mínimo de reproducción:

```bash
python create_folds.py --input data/raw/train_wide.csv --output_dir ./data/folds/folds_v1 --n_splits 3
python train.py --version v1 --epochs 100 --batch_size 2 --img_size 1024
python predict.py --csv data/test/test_wide.csv --images_dir data/test --weights_dir checkpoints/v1 --output predictions.csv
```

## 9. Estructura del proyecto

```text
.
├── GUIDE.md
├── MODEL_ANALYSIS.md
├── README.md
├── README_SETUP.md
├── create_folds.py
├── dataset.py
├── docs/
│   ├── download_weights.py
│   └── figures/
│       ├── Figure_1.png
│       └── IMG_20260811_155344_n.jpg
├── download_weights.py
├── generar_diagramas.py
├── inference_dataset.py
├── install_deps.py
├── loss.py
├── merge_weight.py
├── metrics.py
├── model.py
├── predict.py
├── prepare_data.py
├── train.py
├── transforms.py
└── utils/
    └── log_setup.py
```

No incluidos en esta versión pública: `requirements.txt`, `config.py`, `trainer.py`, `scripts/plot_metrics.py`, `utils/paths.py`, `utils/plotting.py` y los directorios generados o locales `data/`, `checkpoints/`, `logs/` y `backbone_weights/`. El código actual no importa `config.py` ni `trainer.py`; aparecían en la estructura de versiones anteriores del README.

## 10. Autor y contacto

Autor: Daniel Polo Alcoba.  
Correo: dpoloalcoba@gmail.com.
