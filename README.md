# 🛍️ Amazon Reviews NLP - Sentiment Classification with PyTorch

Proyecto de Procesamiento de Lenguaje Natural (NLP) desarrollado como trabajo final del curso **Data Science III - NLP**, cuyo objetivo es clasificar reseñas de productos mediante un modelo de Deep Learning implementado en PyTorch.

El proyecto sigue un flujo completo de Machine Learning, desde la exploración del conjunto de datos hasta el entrenamiento y evaluación del modelo.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-green)
![License](https://img.shields.io/badge/Status-Completed-success)

## 📖 Proyecto

El flujo de trabajo se encuentra dividido en cuatro notebooks principales:

| Notebook | Descripción |
|----------|-------------|
| **01_dataset_exploration** | Exploración del dataset y análisis exploratorio de datos (EDA). |
| **02_preprocessing** | Pipeline de limpieza, normalización, tokenización, eliminación de stopwords y lematización (NLTK y comparación con spaCy). |
| **03_modeling** | Representación mediante TF-IDF, construcción del modelo en PyTorch y entrenamiento. |
| **04_evaluation** | Evaluación del modelo, métricas, matriz de confusión y análisis de resultados. |

---

## ⚙️ Pipeline de procesamiento

El texto pasa por las siguientes etapas:

- Conversión a minúsculas
- Expansión de contracciones
- Tokenización
- Eliminación de stopwords
- Lematización (NLTK)
- Comparación con spaCy
- Reconstrucción del texto
- Vectorización mediante TF-IDF

---

## 🧠 Modelo

El clasificador fue implementado utilizando **PyTorch** e incluye:

- TF-IDF como representación numérica
- Red neuronal totalmente conectada (MLP)
- Dos capas densas
- Batch Normalization
- Dropout (p = 0.5)
- Función de pérdida CrossEntropyLoss
- Optimizador Adam
- Early Stopping

---

## 📊 Evaluación

El proyecto incorpora:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Matriz de confusión
- Curvas de entrenamiento y validación

---

## 🛠️ Tecnologías utilizadas

- Python
- Pandas
- NumPy
- NLTK
- spaCy
- Scikit-learn
- PyTorch
- Matplotlib
- Joblib

---

## 🚀 Cómo ejecutar

1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/amazon_reviews_nlp.git
cd amazon_reviews_nlp
```

2. Crear un entorno virtual

```bash
python -m venv .venv
```

3. Activarlo

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

4. Instalar dependencias

```bash
pip install -r requirements.txt
```

5. Descargar los recursos de NLTK

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

6. Descargar el modelo de spaCy

```bash
python -m spacy download en_core_web_sm
```

---

## 📌 Dataset

Women's Clothing E-Commerce Reviews

Contiene más de **22.000 reseñas** de clientes con su correspondiente valoración en una escala de 1 a 5 estrellas.

---

## 🎯 Objetivos del proyecto

- Aplicar técnicas clásicas de NLP.
- Construir un pipeline modular de procesamiento de texto.
- Comparar distintas estrategias de lematización.
- Representar texto mediante TF-IDF.
- Implementar un clasificador de Deep Learning con PyTorch.
- Evaluar el desempeño mediante métricas de clasificación.

---

## 👤 Autor

**Lautaro Ezequiel Rodriguez**

Proyecto desarrollado como parte de la especialización en Data Science y Machine Learning.