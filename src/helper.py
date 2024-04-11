import nltk
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

# import nltk libs and stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

_stemmer = EnglishStemmer()

# Define global constants
NUM_DOCS    = 2500
VOCAB_SIZE  = 2000

ALIAS_FILE      = './data/aliases.parquet'
ADJ_LIST_FILE   = './data/adj_list.parquet'
DOC_INFO_FILE   = './data/doc_info.parquet'
INV_IDX_FILE    = './data/inv_idx.parquet'
VOCAB_FILE      = './data/vocab.parquet'

def load_aliases() -> pd.DataFrame:
    """Loads the stored aliases

    :returns: aliases as a DataFrame
        (from -> to)
    """

    print('Loading aliases ...')
    aliases = pd.read_parquet(ALIAS_FILE, engine='pyarrow')
    print('Finished loading aliases\n')

    return aliases

def load_adj_list() -> pd.DataFrame:
    """Loads the stored document info

    :returns: document info as a DataFrame
        (docid -> out_links)
    """

    print('Loading adjacency list ...')
    adj_list = pd.read_parquet(ADJ_LIST_FILE, engine='pyarrow')
    print('Finished loading\n')

    return adj_list

def load_doc_info() -> pd.DataFrame:
    """Loads the stored document info

    :returns: document info as a DataFrame
        (docid -> title, url, len, PageRank, auth_score, hub_score)
    """

    print('Loading doc info ...')
    doc_info = pd.read_parquet(DOC_INFO_FILE, engine='pyarrow')
    print('Finished loading\n')

    return doc_info

def load_inv_idx() -> pd.DataFrame:
    """Loads the stored inveted index

    :returns: inveted index as a DataFrame
        (term -> docid -> frequency)
    """

    print('Loading inverted index ...')
    inv_idx = pd.read_parquet(INV_IDX_FILE, engine='pyarrow')
    print('Finished loading\n')

    return inv_idx

def load_vocab() -> pd.DataFrame:
    """Loads the stored vocab

    :returns: vocab as a DataFrame
        (term -> frequency)
    """

    print('Loading vocab ...')
    vocab = pd.read_parquet(VOCAB_FILE, engine='pyarrow')
    print('Finished loading\n')

    return vocab

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the stored data files

    :returns:
        doc info: (docid -> title, url, len, PageRank, auth_score, hub_score)
        inv idx: (term -> docid -> frequency)
        vocab: (term -> frequency)
    """

    doc_info = load_doc_info()
    inv_idx  = load_inv_idx()
    vocab    = load_vocab()

    return (doc_info, inv_idx, vocab)

def parse_text(text: str) -> list[str]:
    """Parse text into a list of terms

    :param text: text to be parsed
    :returns: list of terms in text
    """

    # remove new line chars, non-alphanumeric chars, and nums
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r"'[\w]*|[^\w\s]+|_+", ' ', text)
    text = re.sub(r'\d+[\w]*', ' ', text)

    # remove extra whitespace and strip
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    text = text.lower()

    # tokenize and remove stopwords
    word_tokens = word_tokenize(text)
    filtered = [ _stemmer.stem(w) for w in word_tokens if w not in stop_words ]

    return filtered
