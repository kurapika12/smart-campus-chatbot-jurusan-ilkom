import re
import string

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ======================
# STOPWORDS
# ======================

stop_words = set(stopwords.words("indonesian"))

# ======================
# STEMMER
# ======================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ======================
# PREPROCESSING
# ======================

def clean_text(text):

    # Case Folding
    text = str(text).lower()

    # Hapus URL
    text = re.sub(r"http\S+|www\S+", "", text)

    # Hapus angka
    text = re.sub(r"\d+", "", text)

    # Hapus tanda baca
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Hapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # ======================
    # TOKENIZING
    # ======================

    tokens = text.split()

    # ======================
    # STOPWORD REMOVAL
    # ======================

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # ======================
    # STEMMING
    # ======================

    stemmed_tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return " ".join(stemmed_tokens)