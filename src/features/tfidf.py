"""
Funciones para la representación numérica mediante TF-IDF.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(max_features: int = 5000):
    """
    Crea un vectorizador TF-IDF.
    """

    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True
    )