from helper import DOC_INFO_FILE, INV_IDX_FILE, VOCAB_FILE, VOCAB_SIZE
from helper import load_doc_info, load_inv_idx

import pandas as pd
import os

from collections import Counter
from tqdm import tqdm

def create_vocab() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the vocab

    :returns:
        inv idx
        vocab
    """

    print('Creating vocab ...')

    if os.path.exists(VOCAB_FILE):
        os.remove(VOCAB_FILE)

    inv_idx = load_inv_idx()

    col_counter = Counter()

    all_terms = set(inv_idx.index.get_level_values('term'))
    for term in tqdm(all_terms):
        col_counter[term] = inv_idx.loc[term]['frequency'].sum()

    vocab_counter = col_counter.most_common(VOCAB_SIZE)
    terms = []
    freqs = []
    for term, freq in vocab_counter:
        terms.append(term)
        freqs.append(freq)

    vocab = pd.DataFrame({'term': terms, 'frequency': freqs}).set_index('term')
    vocab.to_csv(VOCAB_FILE, header=False)

    print('Finished')

    return inv_idx, vocab

def reduce_and_sort(inv_idx: pd.DataFrame, vocab: pd.DataFrame) -> None:
    """Load, reduce, and store the inverted index and doc info"""

    print('Reducing and sorting ...')

    terms = set(vocab.index)

    # load, reduce, and sort inverted index
    inv_idx = inv_idx.query('term in @terms')
    inv_idx = inv_idx.sort_index()
    inv_idx.to_csv(INV_IDX_FILE, header=False)

    # load, reduce, and sort doc labels
    doc_info = load_doc_info()
    doc_info = doc_info.sort_index()
    doc_info.to_csv(DOC_INFO_FILE, header=False)

    print('Finished')

    return

