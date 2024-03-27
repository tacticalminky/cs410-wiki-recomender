import nltk
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# import nltk libs and stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# Define global constants
MAX_NUM_DOCS = 25000    # -> 100k
VOCAB_SIZE   = 1000     # -> 8k

DOC_INFO_FILE   = './data/doc_info.csv'
INV_IDX_FILE    = './data/inv_idx.csv'
VOCAB_FILE      = './data/vocab.csv'

def load_doc_info() -> pd.DataFrame:
    """Loads the stored document info

    :returns: document info as a DataFrame
        (docid -> url, title, len)
    """

    doc_info = pd.read_csv(DOC_INFO_FILE,
                           names=['docid', 'url', 'title', 'len'],
                           index_col='docid')

    return doc_info

def load_inv_idx() -> pd.DataFrame:
    """Loads the stored inveted index

    :returns: inveted index as a DataFrame
        (term -> docid -> frequency)
    """

    inv_idx = pd.read_csv(INV_IDX_FILE,
                          names=['term', 'docid', 'frequency'],
                          index_col=['term', 'docid'])

    return inv_idx

def load_vocab() -> pd.DataFrame:
    """Loads the stored vocab

    :returns: vocab as a DataFrame
        (term -> frequency)
    """

    vocab = pd.read_csv(VOCAB_FILE,
                        names=['term', 'frequency'],
                        index_col='term')

    return vocab

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the stored data files

    :returns: (doc info, inv idx, vocab)
    """

    doc_info = load_doc_info()
    inv_idx  = load_inv_idx()

    vocab    = pd.read_csv(VOCAB_FILE,
                           names=['term', 'frequency'],
                           index_col='term')

    return (doc_info, inv_idx, vocab)


def parse_text(text: str) -> list[str]:
    """Parse text into a list of terms

    :param text: text to be parsed
    :returns: list of terms in text
    """

    # remove new line chars, non-alphanumeric chars, and nums
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r"'[\w]*|[^\w\s]+", ' ', text)
    text = re.sub(r'\d+[\w]*', ' ', text)

    # remove extra whitespace and strip
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    text = text.lower()

    # tokenize and remove stopwords
    word_tokens = word_tokenize(text)
    filtered = [ stemmer.stem(w) for w in word_tokens if not w in stop_words ]

    return filtered
