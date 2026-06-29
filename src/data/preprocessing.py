"""
Funciones de preprocesamiento para NLP.
"""

import re
import string

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Inicialización
lemmatizer = WordNetLemmatizer()

STOPWORDS = set(stopwords.words("english"))

def normalize_text(text: str) -> str:
    """
    Convierte el texto a minúsculas
    y elimina espacios sobrantes.
    """

    text = text.lower()

    text = text.strip()

    return text

import contractions

def expand_contractions(text: str) -> str:
    """
    Expande contracciones del inglés.

    Ejemplo:
        don't -> do not
        it's -> it is
    """

    return contractions.fix(text)

def remove_punctuation(text: str) -> str:
    """
    Elimina los signos de puntuación.
    """

    return text.translate(
        str.maketrans("", "", string.punctuation)
    )

def remove_numbers(text: str) -> str:
    """
    Elimina todos los dígitos del texto.
    """

    return re.sub(r"\d+", "", text)

from nltk.tokenize import word_tokenize

def tokenize(text: str) -> list[str]:
    """
    Divide el texto en una lista de tokens.
    """

    return word_tokenize(text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Elimina las stopwords de una lista de tokens.
    """

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

def lemmatize(tokens: list[str]) -> list[str]:
    """
    Convierte cada token a su lema.
    """

    return [lemmatizer.lemmatize(token) for token in tokens]

def join_tokens(tokens: list[str]) -> str:
    """
    Une nuevamente la lista de tokens en un texto.
    """

    return " ".join(tokens)

def preprocess_text(text: str) -> str:
    """
    Ejecuta el pipeline completo de preprocesamiento.
    """

    if not isinstance(text, str):
        return ""

    text = normalize_text(text)

    text = expand_contractions(text)

    tokens = tokenize(text)

    tokens = remove_stopwords(tokens)

    tokens = lemmatize(tokens)

    text = join_tokens(tokens)

    return text